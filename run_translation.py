import logging
import os
from typing import List

import pandas as pd

import constants
import utility
from datastructures.LLM import GPTOpenAI, HFLLM
import datastructures.LLM
from datastructures.Codesample import Codesample, get_language_by_suffix, get_target_language
from config import config
from datastructures.TestcaseOutcome import TestcaseOutcome, TestcaseResult
from utility import get_dataset_from_problem_id, translation_codesample_factory

logger = logging.getLogger("my_project_logger")


def run_translations(data, llm_code, prompt_selection=False, llm_models=config["steps"]["run_translation"]):
    for llm_model in llm_models:
        if "openai_" in llm_model:
            llm = GPTOpenAI(llm_model)
            translate_all_samples_for_llm(llm, data, llm_code, prompt_selection)
        else:
            llm = HFLLM(llm_model)
            translate_all_samples_for_llm(llm, data, llm_code, prompt_selection)
    return llm_code


def translate_all_samples_for_llm(llm: datastructures.LLM, data, llm_code, prompt_selection):
    for (dataset, task_id), d in data.items():
        sample: Codesample
        for sample in d.code_samples:
            target_pls = sample.get_target_languages()
            for target_lang in target_pls:
                prompt_crafting: bool = config["steps"]["run_prompt_crafting"]
                existing_translation: Codesample = None

                # early exit if translation is already correct for iterative repair
                if config["steps"]["run_prompt_crafting"] and utility.is_task_solved(dataset, task_id, llm.model,
                                                                                     llm_code, sample.submission_id,
                                                                                     target_lang):
                    continue

                if config["steps"]["run_prompt_crafting"] or config["steps"]["run_backtranslation"]:
                    relevant_iteration = config["prompt_crafting_current_iteration"] - 1 if config["steps"][
                        "run_prompt_crafting"] else 0
                    best_samples = list(filter(
                        lambda cs: cs.programming_language == target_lang and sample.submission_id in cs.submission_id
                                   and cs.get_is_best_sample(),
                        llm_code[(dataset, task_id)][llm.model][relevant_iteration]
                    ))

                    if len(best_samples) == 0:
                        logger.error(
                            f"Prompt crafting OR backtranslation is run but no best samples found for: {dataset}/{task_id}/{sample.submission_id} for {llm.model} for {target_lang} as target_lang")
                        if relevant_iteration > 0:
                            fallback_samples = list(filter(lambda
                                                               cs: cs.programming_language == target_lang and sample.submission_id in cs.submission_id,
                                                           llm_code[(dataset, task_id)][llm.model][relevant_iteration]))
                            if len(fallback_samples) > 0:
                                existing_translation = fallback_samples[0]
                                logger.error(f"Found fallback code sample.")
                            else:
                                logger.error("No fallback code sample found.")
                                raise FileNotFoundError()
                    else:
                        existing_translation = best_samples[0]

                if config["steps"]["run_backtranslation"]:
                    target_lang = sample.programming_language

                if prompt_selection:
                    translate_one_sample_for_llm(dataset, task_id, llm, target_lang, sample, llm_code,
                                                 prompt_crafting, existing_translation, "selection")

                else:
                    translate_one_sample_for_llm(dataset, task_id, llm, target_lang, sample, llm_code,
                                                 prompt_crafting, existing_translation, "default", data)


def translate_one_sample_for_llm(dataset: str, task_id: str, llm: datastructures.LLM, target_lang: str,
                                 code_sample: Codesample, llm_code,
                                 prompt_crafting: bool = False, existing_translation: Codesample = None,
                                 prompt_id: str = "default", data: dict = None):
    prompt_crafting_iteration = config["prompt_crafting_current_iteration"]
    # only translate several times if it's the first translation (i.e., not during prompt crafting approach)
    num_iterations = config["number_of_translations_per_task"] if prompt_crafting_iteration == 0 else 1

    if not config["steps"]["run_backtranslation"]:
        # do the translations (not prompt crafting but actual pipeline) already exist? in that case, skip translation
        do_translations_exist = [os.path.exists(
            utility.get_code_sample_path(code_sample.submission_id, dataset, target_lang, task_id, llm.model, it_number,
                                         "default", prompt_crafting_iteration)) for it_number in
            range(config["number_of_translations_per_task"])]
        if all(do_translations_exist):
            return

    codes = llm.translate(code_sample, target_lang, dataset, task_id, num_iterations,
                          prompt_crafting, existing_translation, prompt_id, data)

    translated_code_samples = []

    if prompt_id == "selection":
        for pid in codes:
            for it_number, code in enumerate(codes[pid]):
                translated_code_sample = translation_codesample_factory(code_sample.submission_id, code, dataset,
                                                                        target_lang, task_id, llm.model, it_number, pid,
                                                                        prompt_crafting_iteration)
                translated_code_samples.append(translated_code_sample)
    else:
        for it_number, code in enumerate(codes):
            submission_id = code_sample.submission_id
            if config["steps"]["run_backtranslation"]:
                submission_id = code_sample.submission_id + "!" + existing_translation.programming_language  # we need to note down the intermediate PL in backtranslation
            translated_code_sample = translation_codesample_factory(submission_id, code, dataset,
                                                                    target_lang, task_id, llm.model, it_number,
                                                                    "default",
                                                                    prompt_crafting_iteration)
            translated_code_samples.append(translated_code_sample)

    for tcs in translated_code_samples:
        if (dataset, task_id) not in llm_code.keys():
            llm_code[(dataset, task_id)] = {}
        if llm.model not in llm_code[(dataset, task_id)].keys():
            llm_code[(dataset, task_id)] = {llm.model: {}}
        if prompt_crafting_iteration not in llm_code[(dataset, task_id)][llm.model]:
            llm_code[(dataset, task_id)][llm.model][prompt_crafting_iteration] = []

        llm_code[(dataset, task_id)][llm.model][prompt_crafting_iteration].append(tcs)


def load_from_disk(data, tcos, code_smells):
    llm_code = {}
    if tcos is not None:
        tcos_paths = tcos["Path"].tolist()
        tcos_paths_dedup = list(set(tcos["Path"].tolist()))
        tcos_exist = True
    else:
        tcos_exist = False

    if code_smells is not None:
        cs_paths = code_smells["path"].tolist()
        cs_exist = True
    else:
        cs_exist = False

    for (dataset, task_id), _ in data.items():
        for llm_model in config["steps"]["read_translations_from_disk"]:
            base_path = os.path.join(config["base_path_data"], dataset, "translations", task_id, llm_model)
            try:
                iterations = os.listdir(base_path)
                for it in iterations:
                    if (not it.isdigit()) or int(it) > config['prompt_crafting_current_iteration']:
                        continue
                    if not int(it) == config["prompt_crafting_current_iteration"]:
                        continue
                    path = os.path.join(base_path, it)
                    for root, dirs, files in os.walk(path):
                        if len(files) == 0:
                            continue
                        task_id = root.split(os.path.sep)[-4]
                        llm_model = root.split(os.path.sep)[-3] + "/" + root.split(os.path.sep)[-2]
                        iteration = int(root.split(os.path.sep)[-1])
                        if (dataset, task_id) not in llm_code:
                            llm_code[(dataset, task_id)] = {}
                        if llm_model not in llm_code[(dataset, task_id)].keys():
                            llm_code[(dataset, task_id)][llm_model] = {}
                        if int(iteration) not in llm_code[(dataset, task_id)][llm_model]:
                            llm_code[(dataset, task_id)][llm_model][int(iteration)] = []

                        for f in files:
                            if f.startswith('.'):
                                continue  # ignore .DS_Store
                            target_lang = get_language_by_suffix("." + f.split(".")[-1])
                            version_number = int(f.split("__")[0])
                            prompt_id = f.split("__")[1]
                            submission_id = f.split("__")[2].rsplit(".", 1)[0]
                            file_path = utility.get_code_sample_path(submission_id, dataset, target_lang, task_id,
                                                                     llm_model, version_number, prompt_id, iteration)

                            if llm_model in config["steps"]["preprocess_translations"]:
                                with open(file_path, "r") as file:
                                    code = file.read()
                                    code = utility.preprocess_code(code, target_lang)
                                with open(file_path, "w") as file:
                                    file.write(code)

                            translated_code_sample = utility.cheap_codesample_factory(f, file_path, dataset, task_id,
                                                                                      prompt_id)
                            if tcos_exist:
                                if config["base_path_data_vm"] + f"/{dataset}/translations" + \
                                        file_path.split(f"{dataset}/translations")[1] in tcos_paths_dedup:
                                    series = [config["base_path_data_vm"] + f"/{dataset}/translations" +
                                              file_path.split(f"{dataset}/translations")[1] == p for p in tcos_paths]
                                    intermediate_df = tcos[series]
                                    translated_code_sample.testcase_outcomes = [TestcaseOutcome(tc_id,
                                                                                                TestcaseResult[
                                                                                                    outcome.split(".")[
                                                                                                        1]],
                                                                                                error,
                                                                                                stacktrace) for
                                                                                tc_id, outcome, error, stacktrace in
                                                                                zip(intermediate_df["Testcase_id"],
                                                                                    intermediate_df["Outcome"],
                                                                                    intermediate_df["Error"],
                                                                                    intermediate_df["Stacktrace"])]
                            if cs_exist:
                                if config["base_path_data_vm"] + f"/{dataset}/translations" + \
                                        file_path.split(f"{dataset}/translations")[1] in cs_paths:
                                    cs_result = code_smells[code_smells["path"] == (
                                            config["base_path_data_vm"] + f"/{dataset}/translations" +
                                            file_path.split(f"{dataset}/translations")[1])][
                                        "code_smell_result"].iloc[0]
                                    translated_code_sample.code_smell_result = cs_result

                            llm_code[(dataset, task_id)][llm_model][int(iteration)].append(translated_code_sample)
            except (FileNotFoundError, TypeError) as e:
                if config['steps']['run_prompt_crafting'] and config['prompt_crafting_current_iteration'] != 0:
                    logger.error('Please either disable prompt crafting or set current iteration to 0!')
                    exit(-1)

                logger.error("Code sample not found!")
                logger.error(e)

    return llm_code


def load_from_disk_prompt_crafting(data: dict, tcos: pd.DataFrame, code_smells: pd.DataFrame, iteration: int):
    llm_code = {}
    data_keys = list(data.keys())

    code_smells["Problem ID"] = code_smells["path"].apply(lambda e: e.split("/")[8])
    code_smells["LLM"] = code_smells["path"].apply(lambda p: p.split("/")[9] + "/" + p.split("/")[10])
    code_smells["Dataset"] = code_smells["Problem ID"].apply(get_dataset_from_problem_id)
    code_smells["Iteration"] = code_smells["path"].apply(lambda p: int(p.split("/")[-2]))
    tcos["Iteration"] = tcos["Path"].apply(lambda p: int(p.split("/")[-2]))

    tcos_subset = tcos[
        (tcos["Iteration"] == iteration) & (tcos["Llm"].isin(config["steps"]["read_translations_from_disk"]))]
    cs_subset = code_smells[(code_smells["Iteration"] == iteration) & (
        code_smells["LLM"].isin(config["steps"]["read_translations_from_disk"]))]

    for dataset in constants.list_of_datasets:
        is_best_true_set_for = 0
        tcos_subset_d = tcos_subset[tcos_subset["Dataset"] == dataset]
        cs_subset_d = cs_subset[cs_subset["Dataset"] == dataset]
        relevant_keys = list(filter(lambda key: key[0] == dataset, data_keys))
        for k in relevant_keys:
            task_id = k[1]
            tcos_subset_k = tcos_subset_d[tcos_subset_d["Problem ID"] == task_id]
            cs_subset_k = cs_subset_d[cs_subset_d["Problem ID"] == task_id]
            d = data[k]
            if (dataset, task_id) not in llm_code:
                llm_code[(dataset, task_id)] = {}

            for llm_model in config["steps"]["read_translations_from_disk"]:
                path = os.path.join(config["base_path_data"], dataset, "translations", task_id, llm_model,
                                    str(iteration))
                files = os.listdir(path)

                if llm_model not in llm_code[(dataset, task_id)].keys():
                    llm_code[(dataset, task_id)][llm_model] = {}
                if int(iteration) not in llm_code[(dataset, task_id)][llm_model]:
                    llm_code[(dataset, task_id)][llm_model][int(iteration)] = []

                for f in files:
                    if f.startswith('.'):
                        continue  # ignore .DS_Store
                    target_lang = get_language_by_suffix("." + f.split(".")[-1])
                    version_number = int(f.split("__")[0])
                    prompt_id = f.split("__")[1]
                    submission_id = f.split("__")[2].rsplit(".", 1)[0]
                    file_path = utility.get_code_sample_path(submission_id, dataset, target_lang, task_id,
                                                             llm_model, version_number, prompt_id, iteration)

                    if llm_model in config["steps"]["preprocess_translations"]:
                        with open(file_path, "r") as file:
                            code = file.read()
                            code = utility.preprocess_code(code, target_lang)
                        with open(file_path, "w") as file:
                            file.write(code)

                    translated_code_sample = utility.cheap_codesample_factory(f, file_path, dataset, task_id,
                                                                              prompt_id)
                    intermediate_df = tcos_subset_k[
                        tcos_subset_k["Path"] == (config["base_path_data_vm"] + f"/{dataset}/translations" +
                                                  file_path.split(f"{dataset}/translations")[1])]
                    translated_code_sample.testcase_outcomes = [TestcaseOutcome(tc_id,
                                                                                TestcaseResult[outcome.split(".")[1]],
                                                                                error,
                                                                                stacktrace) for
                                                                tc_id, outcome, error, stacktrace in
                                                                zip(intermediate_df["Testcase_id"],
                                                                    intermediate_df["Outcome"],
                                                                    intermediate_df["Error"],
                                                                    intermediate_df["Stacktrace"])]
                    is_best = False
                    try:
                        item = cs_subset_k[cs_subset_k["path"] == (
                                config["base_path_data_vm"] + f"/{dataset}/translations" +
                                file_path.split(f"{dataset}/translations")[1])]
                        cs_result = item["code_smell_result"].iloc[0]
                        is_best = item["is_best_cs"].iloc[0]
                    except IndexError:
                        cs_result = constants.cs_parsing_error

                    if is_best:
                        is_best_true_set_for += 1

                    translated_code_sample.set_cs_result(cs_result)
                    translated_code_sample.set_is_best_sample(is_best)

                    llm_code[(dataset, task_id)][llm_model][int(iteration)].append(translated_code_sample)

        logger.info(f"Loaded {dataset}; is_best_count: {is_best_true_set_for}")
    return llm_code


def load_from_disk_minimal(tcos: pd.DataFrame, tcos_paths, code_smells: pd.DataFrame,
                           samples_to_be_translated: List[str], llm: str, dataset: str, iteration: int, task_id: str,
                           n_translations: int):
    llm_code = {}
    llm_code[(dataset, task_id)] = {}
    llm_code[(dataset, task_id)][llm] = {}
    llm_code[(dataset, task_id)][llm][int(iteration)] = []
    tcos_subset = tcos[tcos["Problem ID"] == task_id]
    cs_subset = code_smells[code_smells["Problem ID"] == task_id]
    for s in samples_to_be_translated:
        submission_id = s.split(".")[0]
        for pl in get_target_language(get_language_by_suffix(s.split(".")[1])):
            for i in range(0, n_translations):
                if not iteration == 100:
                    file_path = utility.get_code_sample_path(submission_id, dataset, pl, task_id,
                                                             llm, i, "default", iteration)
                    translated_code_sample = utility.cheap_codesample_factory(s, file_path, dataset, task_id,
                                                                              "default")
                else:
                    local_submission_id = submission_id + f"!{pl}"
                    file_path = utility.get_code_sample_path(local_submission_id, dataset,
                                                             get_language_by_suffix(s.split(".")[1]), task_id,
                                                             llm, i, "default", iteration)
                    translated_code_sample = utility.cheap_codesample_factory(local_submission_id, file_path, dataset,
                                                                              task_id,
                                                                              "default")

                intermediate_df = tcos_subset[
                    tcos_subset["Path"] == config["base_path_data_vm"] + f"/{dataset}/translations" +
                    file_path.split(f"{dataset}/translations")[1]]
                translated_code_sample.testcase_outcomes = [TestcaseOutcome(tc_id,
                                                                            TestcaseResult[outcome.split(".")[1]],
                                                                            error,
                                                                            stacktrace) for
                                                            tc_id, outcome, error, stacktrace in
                                                            zip(intermediate_df["Testcase_id"],
                                                                intermediate_df["Outcome"],
                                                                intermediate_df["Error"],
                                                                intermediate_df["Stacktrace"])]
                try:
                    cs_result = cs_subset[cs_subset["path"] == (
                            config["base_path_data_vm"] + f"/{dataset}/translations" +
                            file_path.split(f"{dataset}/translations")[1])]["code_smell_result"].iloc[0]
                except IndexError:
                    cs_result = constants.cs_parsing_error
                translated_code_sample.set_cs_result(cs_result)

                llm_code[(dataset, task_id)][llm][int(iteration)].append(translated_code_sample)
    return llm_code
