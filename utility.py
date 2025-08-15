import os
import logging
import pathlib
import re
from itertools import chain
from xml.etree import ElementTree

import pandas as pd
import json

import constants
from config import config
from datastructures.Codesample import PySample, JSample, get_suffix, CodeSampleStatus, ScalaSample, RustSample, Codesample


logger = logging.getLogger("my_project_logger")

# os.walk but with depth restriction
def walk_depth_restricted(path, max_depth):
    base_depth = path.rstrip(os.path.sep).count(os.path.sep)
    for root, dirs, files in os.walk(path):
        yield root, dirs[:], files
        cur_depth = root.count(os.path.sep)
        if base_depth + max_depth <= cur_depth:
            del dirs[:]


def load_content_from_file(path: str):
    with open(path) as file:
        content = file.read()
    return content


def cheap_codesample_factory(submission_id: str, path: str, dataset: str, task_id: str, prompt_id: str = "default"):
    extension = path.split('.')[-1]
    cs = None

    if extension == 'py':
        cs = PySample(submission_id, path, dataset, task_id, prompt_id)
    elif extension == 'java':
        cs = JSample(submission_id, path, dataset, task_id, prompt_id)
    elif extension == "scala":
        cs = ScalaSample(submission_id, path, dataset, task_id, prompt_id)
    elif extension == "rs":
        cs = RustSample(submission_id, path, dataset, task_id, prompt_id)
    else:
        raise NotImplementedError()

    return cs


def translation_codesample_factory(submission_id: str, code: str, dataset: str, programming_language: str, task_id: str,
                                   llm_model: str, number: int, prompt_id: str, iteration: int = 0):
    file_path = get_code_sample_path(submission_id, dataset, programming_language, task_id, llm_model, number, prompt_id,
                                     iteration)
    with open(file_path, "w") as f:
        f.write(code)
    return cheap_codesample_factory(submission_id, file_path, dataset, task_id, prompt_id)


def get_code_sample_path(submission_id: str, dataset: str, programming_language: str, task_id: str, llm_model: str,
                         number: int, prompt_id: str, iteration: int = 0):
    base_path = os.path.join(config["base_path_data"], dataset, "translations", task_id, llm_model, str(iteration))
    os.makedirs(base_path, exist_ok=True)
    return os.path.join(base_path, f"{str(number)}__{prompt_id}__{submission_id}{get_suffix(programming_language)}")


def get_best_translation(dataset: str, task_id: str, llm_model: str, target_lang: str, llm_code, from_iteration: int,
                         submission_id: str, backtranslation: bool = False, source_pl: str = "") -> Codesample:
    if not backtranslation:
        samples = list(filter(
            lambda cs: cs.programming_language == target_lang and submission_id in cs.submission_id, llm_code[(dataset, task_id)][llm_model][from_iteration]
        ))
    else:
        samples = list(filter(
            lambda cs: cs.programming_language == source_pl and submission_id + f"!{target_lang}" == cs.submission_id, llm_code[(dataset, task_id)][llm_model][from_iteration]
        ))
    best_sample: Codesample = samples[0]

    for s in samples:
        if is_other_sample_better(best_sample, s):
            best_sample = s
    best_sample.set_is_best_sample()
    return best_sample


def is_other_sample_better(original_sample: Codesample, other_sample: Codesample):
    if original_sample.code_sample_status == other_sample.code_sample_status:
        return original_sample.get_code_smell_severity() > other_sample.get_code_smell_severity()
    else:
        if original_sample.code_sample_status == CodeSampleStatus.PASSED:
            return False
        elif original_sample.code_sample_status == CodeSampleStatus.FAILED and other_sample.code_sample_status == CodeSampleStatus.PASSED:
            return True
        else:
            return original_sample.get_code_smell_severity() > other_sample.get_code_smell_severity()


def is_task_solved(dataset: str, task_id: str, llm_model: str, llm_code: dict, submission_id: str, target_lang: str):
    samples = llm_code[(dataset, task_id)][llm_model][config["prompt_crafting_current_iteration"] - 1]
    samples = list(filter(lambda s: submission_id in s.path and get_suffix(target_lang) in s.path, samples))

    try:
        if any([cs.is_code_sample_solved() for cs in samples]):
            logging.info(f'Task {dataset}/{task_id}/{submission_id} from llm {llm_model} is considered solved!')
            return True
    except IndexError as e:
        print(e)
    return False


def code_smell_to_str(code_smell, pl: str):
    if code_smell == constants.cs_parsing_error or isinstance(code_smell, str):
        return code_smell

    if pl == constants.scala:
        return ElementTree.tostring(code_smell.getroot(), encoding="unicode")
    else:
        return json.dumps(code_smell)


def save_code_smell_result_data(data):
    paths = []
    code_smell_results = []
    problem_ids = []
    for obj in data.values():
        paths.append(map(lambda cs: cs.path, obj.code_samples))
        code_smell_results.append(map(lambda cs: code_smell_to_str(cs.code_smell_result, cs.programming_language), obj.code_samples))
        problem_ids.append(map(lambda cs: cs.task_id, obj.code_samples))
    paths = list(chain.from_iterable(paths))
    code_smell_results = list(chain.from_iterable(code_smell_results))
    problem_ids = list(chain.from_iterable(problem_ids))

    df = pd.DataFrame(columns=["path", "code_smell_result", "Problem ID"])
    df.path = paths
    df.code_smell_result = code_smell_results
    df["Problem ID"] = problem_ids
    df.to_csv(os.path.join(config["base_path_data"], f"data_code_smell_results_{os.getpid()}.csv"), index=False, quotechar="'", sep='@')


def save_code_smell_result_llm_code(llm_code):
    paths = []
    code_smell_results = []
    is_best_cs = []
    for llms_dict in llm_code.values():
        for iterations_dict in llms_dict.values():
            for l in iterations_dict.values():
                paths.append(map(lambda cs: cs.path, l))
                code_smell_results.append(map(lambda cs: code_smell_to_str(cs.code_smell_result, cs.programming_language), l))
                is_best_cs.append(map(lambda cs: cs.get_is_best_sample(), l))

    paths = list(chain.from_iterable(paths))
    code_smell_results = list(chain.from_iterable(code_smell_results))
    is_best_cs = list(chain.from_iterable(is_best_cs))

    df = pd.DataFrame(columns=["path", "code_smell_result", "is_best_cs"])
    df.path = paths
    df.code_smell_result = code_smell_results
    df.is_best_cs = is_best_cs
    df.to_csv(os.path.join(config["base_path_data"], f"llm_code_smell_results_{os.getpid()}.csv"), index=False, quotechar="'", sep='@')


def load_code_smell_result_data(data):
    new_code_smell = False

    if not os.path.exists(os.path.join(config["base_path_data"], "data_code_smell_results.csv")):
        return True

    df = pd.read_csv(os.path.join(config["base_path_data"], "data_code_smell_results.csv"), sep='@', quotechar="'")
    for obj in data.values():
        for cs in obj.code_samples:
            try:
                cs.code_smell_result = json.loads(df.loc[df["path"] == cs.path, "code_smell_result"].values[0])
            except IndexError:
                cs.run_code_smell_detection()
                new_code_smell = True

    return new_code_smell


def load_code_smell_result_llm_code(llm_code):
    new_code_smell = False

    if not os.path.exists(os.path.join(config["base_path_data"], "llm_code_code_smell_results.csv")):
        return True

    df = pd.read_csv(os.path.join(config["base_path_data"], "llm_code_code_smell_results.csv"), sep='@', quotechar="'")
    for llms_dict in llm_code.values():
        for iterations_dict in llms_dict.values():
            for l in iterations_dict.values():
                for cs in l:
                    try:
                        cs.code_smell_result = json.loads(df.loc[df["path"] == cs.path, "code_smell_result"].values[0])
                    except IndexError:
                        cs.run_code_smell_detection()
                        new_code_smell = True

    return new_code_smell


def get_tco_row_from_data(dataset, problem_id, path, tc_id, outcome, error, stacktrace, llm=None):
    if llm is None:  # indicates a source code sample
        return pd.Series({'Dataset': dataset, 'Problem ID': problem_id, 'Path': path,
                          'Testcase_id': tc_id, 'Outcome': outcome, 'Error': error, 'Stacktrace': stacktrace})
    else:  # we have a llm generated code sample with possible iteration and n_translation values associated
        path_split = path.split(llm)[1]
        path_split = path_split.split(os.path.sep)
        it = int(path_split[1])
        n_translation = int(path_split[2].split('_')[0])
        return pd.Series({'Dataset': dataset, 'Problem ID': problem_id, 'Iteration': it, 'nTranslation': n_translation,
                          'Path': path, 'Testcase_id': tc_id, 'Llm': llm, 'Outcome': outcome, 'Error': error,
                          'Stacktrace': stacktrace})


def combine_testcase_outcome_data_frames(path):
    df_types = os.listdir(path)
    for df_type in df_types:
        if not pathlib.Path(os.path.join(path, df_type)).is_dir():
            continue
        dfs = os.listdir(os.path.join(path, df_type))
        df_list = []
        for d in dfs:
            try:
                df_list.append(pd.read_csv(os.path.join(path, df_type, d)))
            except pd.errors.EmptyDataError:
                pass
        if len(df_list) == 0:
            return

        try:
            dfs = pd.concat(df_list)
            with open(os.path.join('.', f'tco_{df_type}.csv'), "w", newline='') as f:
                dfs.to_csv(f, index=False)
        except ValueError:
            logging.debug(f'Could not combine test case dataframe for {df_type}.'
                          f'Possibly already exists and no new tests were conducted.')


def combine_code_smell_results(path):
    files = os.listdir(path)
    data_files = [f for f in files if 'data_code_smell_results' in f]
    llm_files = [f for f in files if 'llm_code_smell_results' in f]

    dfs = []
    for f in data_files:
        p = os.path.join(path, f)
        dfs.append(pd.read_csv(p, sep='@', quotechar="'"))
        pathlib.Path(p).unlink()
    if not len(dfs) == 0:
        data_df = pd.concat(dfs)
        with open(os.path.join(path, 'data_code_smell_results.csv'), "w", newline='') as f:
            data_df.to_csv(f, index=False, quotechar="'", sep='@')

    dfs = []
    for f in llm_files:
        try:
            p = os.path.join(path, f)
            dfs.append(pd.read_csv(p, sep='@', quotechar="'"))
        except pd.errors.EmptyDataError:
            continue
    if not len(dfs) == 0:
        llm_df = pd.concat(dfs)
        with open(os.path.join(path, 'llm_code_smell_results.csv'), "w", newline='') as f:
            llm_df.to_csv(f, index=False, quotechar="'", sep='@')


def preprocess_code(code: str, target_lang: str):
    csplit = code.split("```" + target_lang)
    if len(csplit) > 1:
        code = csplit[1]
    csplit = code.split("```" + target_lang.capitalize())
    if len(csplit) > 1:
        code = csplit[1]
    if len(code.split("```")) == 2:
        code = code.split("```")[0]
    if len(code.split("```")) == 3:
        code = code.split("```")[1]
    if code.startswith(" ") and not code.startswith("  ") and not code.startswith("    ") and not code.startswith(
            "   "):
        code = code[1:]
    code = re.sub(re.compile(".*End-of-Code.*"), "", code)
    return code


def get_dataset_from_problem_id(problem_id: str):
    if problem_id.startswith("p"):
        return constants.codenet
    elif problem_id.startswith("atcoder") or problem_id.startswith("codeforces"):
        return constants.avatar
    elif problem_id.startswith("HumanEval"):
        return constants.evalplus
    elif problem_id.startswith("EvoEval"):
        return constants.evo_eval
    else:
        return "NOT_IMPLEMENTED"
