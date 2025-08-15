import json
import logging
import os
import time

import openai
import tiktoken
from openai import OpenAI

import constants
from transformers import AutoTokenizer
import re
import vllm

from config import config
from datastructures.Codesample import Codesample
from datastructures.TestcaseOutcome import TestcaseOutcome
from utility import load_content_from_file, code_smell_to_str


logger = logging.getLogger("my_project_logger")
hf_models = {}
tokenizers = {}


def log_translation(model, dataset, task_id, submission, it_number, prompt_crafting, prompt_id, target_lang):
    logger.info(
        f"Translating with {model} and prompt_id={prompt_id} for ds={dataset}, task={task_id}, submission={submission} where it_number={it_number}, prompt_crafting={prompt_crafting} to target_lang={target_lang}")


def get_chat_template(model: str, role: str, prompt: str):
    if "mixtral" in model.lower() or "Magicoder" in model or "granite" in model or "mistral" in model.lower():
        return [{"role": "user", "content": prompt}]
    else:
        return [{"role": "system", "content": role}, {"role": "user", "content": prompt}]


def get_error_message(code_sample: Codesample, data: dict):
    cs_error: str = code_smell_to_str(code_sample.get_cs_result(), code_sample.programming_language)
    exception: TestcaseOutcome | None = code_sample.get_random_exception_test_case()
    compiler_error_tc: TestcaseOutcome | None = code_sample.get_random_compiler_error_test_case()
    timeout_tc: TestcaseOutcome | None = code_sample.get_random_timeout_test_case()
    oom_tco: TestcaseOutcome | None = code_sample.get_random_OOM_test_case()
    failed_tco: TestcaseOutcome | None = code_sample.get_random_failed_test_case()

    if compiler_error_tc is not None:
        return f"The code could not be compiled."
    elif exception is not None:
        if not code_sample.programming_language == constants.python:
            return f"An Exception occurred with the following stacktrace: {exception.stacktrace}"
        else:
            return f"An Exception occurred with the following stacktrace: {exception.error}"
    elif failed_tco is not None or timeout_tc is not None or oom_tco is not None:
        if failed_tco is not None:
            descr = "yielded a wrong result"
            tc = code_sample.get_testcase_for_id(failed_tco.testcase_id, data)
        elif timeout_tc is not None:
            descr = "ran into a timeout"
            tc = code_sample.get_testcase_for_id(timeout_tc.testcase_id, data)
        elif oom_tco is not None:
            descr = "caused an out of memory issue"
            tc = code_sample.get_testcase_for_id(oom_tco.testcase_id, data)
        if tc is None:
            return f"A testcase {descr}."
        else:
            return f"A testcase {descr}. The input of the testcase is ```{tc.input}```.\n\nThe output of the testcase should be {tc.output}."
    elif cs_error is not None:
        return f"The following code smells were detected in the presented code: {cs_error}"
    else:
        logger.error(f"Expected some error but found none for {code_sample.path}")
        return ""


def truncate_prompt(text: str):
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    encoded_str = encoding.encode(text)
    if len(encoded_str) > (15000 - 4096):
        text = encoding.decode(encoded_str[:(15000 - 4096)])
    if len(text) > 60000:
        return text[:60000]
    return text


class LLM:
    def get_translation_prompt(self, task_code_sample: Codesample, target_lang: str, prompt_crafting: bool = False,
                               existing_translation: Codesample = None, prompt_id: str = "default", data: dict = None) -> (str, str):
        raise NotImplementedError()

    def translate(self, code_sample: Codesample, target_lang: str, dataset: str, task_id: str, it_number: int,
                  prompt_crafting: bool = False, existing_translation: Codesample = None, prompt_id: str = "default", data: dict = None):
        raise NotImplementedError()

    def get_model(self):
        return self.model


class GPTOpenAI(LLM):
    def __init__(self, model: str):
        self.client = OpenAI()  # ensure that environment variable with secret key is set; comment if OpenAI not used
        self.model = model

    def get_translation_prompt(self, task_code_sample: Codesample, target_lang: str, prompt_crafting: bool = False,
                               existing_translation: Codesample = None, prompt_id: str = "default", data: dict = None) -> (str, str):
        source_code = truncate_prompt(load_content_from_file(task_code_sample.path))
        source_lang = task_code_sample.programming_language
        ds = task_code_sample.dataset
        if prompt_id == "selection":
            with open(config["prompt_selection_config"], "r") as f:
                prompt_config = json.load(f)
                role, prompt = prompt_config[prompt_id]["role"], prompt_config[prompt_id]["prompt"]
                prompt = prompt.format(source_code=source_code, source_lang=source_lang, target_lang=target_lang)
                return get_chat_template(self.model, role, prompt)
        elif prompt_id == "default" and not prompt_crafting and not config["steps"]["run_backtranslation"]:
            with open(config["prompt_selection_config"], "r") as f_ps_config:
                prompt_config = json.load(f_ps_config)
            with open(config["prompt_selection_assignment"], "r") as f_ps_assignment:
                prompt_assignment = json.load(f_ps_assignment)
                pid = prompt_assignment[self.model][ds][f"{task_code_sample.programming_language}-{target_lang}"]
                role, prompt = prompt_config[pid]["role"], prompt_config[pid]["prompt"]
                prompt = prompt.format(source_code=source_code, source_lang=source_lang, target_lang=target_lang)
                return get_chat_template(self.model, role, prompt)
        elif prompt_crafting and not config["steps"]["run_backtranslation"]:
            role = "You are a skilled programmer and have significant experience in code translation."

            old_code = load_content_from_file(existing_translation.path)
            error_message = get_error_message(existing_translation, data)

            prompt = (f"```{source_code}```\n\n You were supposed to translate the above {source_lang} code to {target_lang}.\n\n"
                      f"Your solution was: ``` {old_code}```\n\n"
                      f"Your solution exhibits the following problem: {error_message}\n\n"
                      f"Fix the mentioned issue.")

            with open(f"prompting_log_{self.model}_{config['prompt_crafting_current_iteration']}_{os.getpid()}.log", "a+") as f:
                f.write(json.dumps({"model": self.model, "cs_path": existing_translation.path, "it": config['prompt_crafting_current_iteration'], "prompt": prompt}))
                f.write("\n")

            return get_chat_template(self.model, role, prompt)
        else:
            # backtranslation
            source_code = load_content_from_file(existing_translation.path)
            source_lang = existing_translation.programming_language
            with open(config["prompt_selection_config"], "r") as f_ps_config:
                prompt_config = json.load(f_ps_config)
            with open(config["prompt_selection_assignment"], "r") as f_ps_assignment:
                prompt_assignment = json.load(f_ps_assignment)
                # if previously translated from PL' -> PL'', then now use the same prompt again for this combination.
                pid = prompt_assignment[self.model][ds][f"{task_code_sample.programming_language}-{existing_translation.programming_language}"]
                role, prompt = prompt_config[pid]["role"], prompt_config[pid]["prompt"]
                prompt = prompt.format(source_code=source_code, source_lang=source_lang, target_lang=task_code_sample.programming_language)
                return get_chat_template(self.model, role, prompt)

    def translate(self, code_sample: Codesample, target_lang: str, dataset: str, task_id: str, it_number: int,
                  prompt_crafting: bool = False, existing_translation: Codesample = None, prompt_id: str = "default", data: dict = None):
        role, prompt = self.get_translation_prompt(code_sample, target_lang, prompt_crafting, existing_translation,
                                                   prompt_id)
        log_translation(self.model, dataset, task_id, code_sample.submission_id, it_number, prompt_crafting, prompt_id,
                        target_lang)
        try:
            completion = self.query_api(role, prompt)
            ret_val = completion.choices[0].message.content
            ret_val = re.sub(r"(```.*$)|(\|Start-of-Code\|)|(\|End-of-Code\|)", "", ret_val, flags=re.M)
            code = ret_val.strip() + "\n"
        except LLMError:
            code = "MAX ATTEMPT EXCEEDED - NO CODE GENERATED"

        logger.debug(code)
        return code

    def query_api(self, role: str, prompt: str):
        successful_translation = False
        current_attempt = 0
        max_attempts = config["LLMs"][self.model]["max_attempts"]

        while (not successful_translation) and current_attempt < max_attempts:
            current_attempt += 1
            logger.debug(f"{self.model}: Prompt is \n{prompt}")
            try:
                completion = self.client.chat.completions.create(
                    model=self.model.split("openai_")[1],
                    messages=[
                        {"role": "system", "content": role},
                        {"role": "user", "content": prompt}
                    ]
                )
                successful_translation = True
            except (openai.APITimeoutError, openai.APIError, openai.APIConnectionError,
                    openai.AuthenticationError, openai.PermissionDeniedError, openai.RateLimitError) as e:
                # check for code 200
                # create statistics
                print(f"OpenAI API request failed: {e}. Retry {current_attempt - 1}")
                continue

            logger.debug(completion)
            logger.debug(completion.choices)

        if not successful_translation:
            raise LLMError()

        return completion


class HFLLM(LLM):
    def __init__(self, model: str):
        self.model = model
        if not self.model in hf_models:
            tokenizer = AutoTokenizer.from_pretrained(self.model)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizers[self.model] = tokenizer

            tp_size = config["num_gpus"]  # Tensor Parallelism
            self.sampling_params = vllm.SamplingParams(temperature=1, top_p=0.92, top_k=50, max_tokens=6144)
            self.sampling_params.stop = [tokenizer.eos_token]
            llm = vllm.LLM(model=self.model, gpu_memory_utilization=0.9,
                           tensor_parallel_size=tp_size, dtype="float16")
            hf_models[self.model] = llm

    def get_translation_prompt(self, task_code_sample: Codesample, target_lang: str, prompt_crafting: bool = False,
                               existing_translation: Codesample = None, prompt_id: str = "default", data: dict = None) -> (str, str):
        source_code = load_content_from_file(task_code_sample.path)
        source_lang = task_code_sample.programming_language
        ds = task_code_sample.dataset
        if prompt_id == "selection":
            with open(config["prompt_selection_config"], "r") as f:
                prompt_config = json.load(f)
                role, prompt = prompt_config[prompt_id]["role"], prompt_config[prompt_id]["prompt"]
                prompt = prompt.format(source_code=source_code, source_lang=source_lang, target_lang=target_lang)
                return get_chat_template(self.model, role, prompt)
        elif prompt_id == "default" and not prompt_crafting and not config["steps"]["run_backtranslation"]:
            with open(config["prompt_selection_config"], "r") as f_ps_config:
                prompt_config = json.load(f_ps_config)
            with open(config["prompt_selection_assignment"], "r") as f_ps_assignment:
                prompt_assignment = json.load(f_ps_assignment)
                pid = prompt_assignment[self.model][ds][f"{task_code_sample.programming_language}-{target_lang}"]
                role, prompt = prompt_config[pid]["role"], prompt_config[pid]["prompt"]
                prompt = prompt.format(source_code=source_code, source_lang=source_lang, target_lang=target_lang)
                return get_chat_template(self.model, role, prompt)
        elif prompt_crafting and not config["steps"]["run_backtranslation"]:
            role = "You are a skilled programmer and have significant experience in code translation."

            old_code = load_content_from_file(existing_translation.path)
            error_message = get_error_message(existing_translation, data)

            prompt = (f"```{source_code}```\n\n You were supposed to translate the above {source_lang} code to {target_lang}.\n\n"
                      f"Your solution was: ``` {old_code}```\n\n"
                      f"Your solution exhibits the following problem: {error_message}\n\n"
                      f"Fix the mentioned issue.")

            with open(f"prompting_log_{self.model.split('/')[1]}_{config['prompt_crafting_current_iteration']}_{os.getpid()}.log", "a+") as f:
                f.write(json.dumps({"model": self.model, "cs_path": existing_translation.path, "it": config['prompt_crafting_current_iteration'], "prompt": prompt}))
                f.write("\n")

            return get_chat_template(self.model, role, prompt)
        else:
            # backtranslation
            source_code = load_content_from_file(existing_translation.path)
            source_lang = existing_translation.programming_language
            with open(config["prompt_selection_config"], "r") as f_ps_config:
                prompt_config = json.load(f_ps_config)
            with open(config["prompt_selection_assignment"], "r") as f_ps_assignment:
                prompt_assignment = json.load(f_ps_assignment)
                # if previously translated from PL' -> PL'', then now use the same prompt again for this combination.
                pid = prompt_assignment[self.model][ds][f"{task_code_sample.programming_language}-{existing_translation.programming_language}"]
                role, prompt = prompt_config[pid]["role"], prompt_config[pid]["prompt"]
                prompt = prompt.format(source_code=source_code, source_lang=source_lang, target_lang=task_code_sample.programming_language)
                return get_chat_template(self.model, role, prompt)

    def translate(self, code_sample: Codesample, target_lang: str, dataset: str, task_id: str, it_number: int,
                  prompt_crafting: bool = False, existing_translation: Codesample = None, prompt_id: str = "default", data: dict = None):
        if prompt_id == "selection":
            prompts = [self.get_translation_prompt(code_sample, target_lang, prompt_crafting,
                                                   existing_translation, f"p{idx % 8 + 1}")
                       for idx in range((8 * 3))]
        elif prompt_id == "default" and prompt_crafting is False and not config["steps"]["run_backtranslation"]:
            prompts = [self.get_translation_prompt(code_sample, target_lang, prompt_crafting, existing_translation, "default")] * it_number
        elif prompt_crafting is True and not config["steps"]["run_backtranslation"]:
            prompts = [self.get_translation_prompt(code_sample, target_lang, prompt_crafting, existing_translation, "default", data)]
        else:
            # backtranslation
            prompts = [self.get_translation_prompt(code_sample, target_lang, True, existing_translation,"default", data)] * config["number_of_translations_per_task"]

        log_translation(self.model, dataset, task_id, code_sample.submission_id, it_number, prompt_crafting, prompt_id,
                        target_lang)
        try:
            start = time.time()
            completion = self.query_api(prompts, prompt_id)
            duration = time.time() - start
            with open(config["time_logging_file"], "a") as f:
                f.write(f"{self.model};{code_sample.dataset};{code_sample.programming_language};{target_lang};{config['prompt_crafting_current_iteration']};{duration}\n")
            logger.info(f"translation took {duration}s")
        except LLMError:
            code = "MAX ATTEMPT EXCEEDED - NO CODE GENERATED"

        return completion

    def query_api(self, prompts, prompt_id: str):
        successful_translation = False
        current_attempt = 0
        max_attempts = config["LLMs"][self.model]["max_attempts"]

        while (not successful_translation) and current_attempt < max_attempts:
            current_attempt += 1

            prompts = [tokenizers[self.model].apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                       for messages in prompts]

            outputs = hf_models[self.model].generate(prompts, self.sampling_params)
            generated_text = [output.outputs[0].text for output in outputs]

            if prompt_id == "selection":
                translations = {}
                for idx, out in enumerate(generated_text):
                    pid = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"][idx % 8]
                    if pid not in translations:
                        translations[pid] = []
                    translations[pid].append(out)
                    logger.debug(f"generated_text for idx={idx} is : {out}")
            else:
                translations = generated_text

            logger.debug(f"Translations is: {translations}")
            successful_translation = True

        if not successful_translation:
            raise LLMError()

        return translations


class LLMError(Exception):
    pass
