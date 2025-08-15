import logging
import math
import os
import random
import shutil
import time
from multiprocessing import Pool

import pandas as pd

import prepare_dataset
import run_code_smell_detection
import run_translation
import test_suit
import utility
from config import config

logging.basicConfig(filename=f"general_logger_main_{os.getpid()}.log")  # inspired by https://stackoverflow.com/questions/73290115/python-logger-prints-everything-twice
my_logger = logging.getLogger("my_project_logger")
my_logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter("%(levelname)s - %(asctime)s - %(message)s"))
file_handler = logging.FileHandler(f"translation_main_{os.getpid()}.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
my_logger.addHandler(stream_handler)
my_logger.addHandler(file_handler)


def pipeline(data, tcos, code_smells):
    if config["steps"]["run_tests"] or config["steps"]["run_tests_source"]:
        # docker is only needed for running tests
        import docker
        pd = os.path.join(config["base_path_testing_docker"], str(os.getpid()))
        os.makedirs(pd, exist_ok=True)
        shutil.copy("run_test_in_docker.py", pd)
        client = docker.from_env()
        client.containers.create("testing:t6", "sh", mem_limit="500m", volumes={pd: {'bind': '/usr/src/app', 'mode': 'rw'}}, working_dir="/usr/src/app", name=f"testing{os.getpid()}", tty=True)
        c = client.containers.get(f"testing{os.getpid()}")
        c.start()
    data = dict(data)

    llm_code = {}.fromkeys(data.keys(), {})

    # Run code translation for all LLMs specified in conf

    if "run_prompt_crafting" in config["steps"] and config["steps"]["run_prompt_crafting"]:
        llm_code = run_translation.load_from_disk_prompt_crafting(data, tcos, code_smells, config["prompt_crafting_current_iteration"] - 1)
        llm_code = run_translation.run_translations(data, llm_code)
    elif config["steps"]["run_backtranslation"]:
        llm_code = run_translation.load_from_disk_prompt_crafting(data, tcos, code_smells, 0)
        llm_code = run_translation.run_translations(data, llm_code)
    elif "run_translation" in config["steps"]:
        llm_code = run_translation.run_translations(data, llm_code)
    else:
        llm_code = run_translation.load_from_disk(data, tcos, code_smells)

    if config["steps"]["run_tests_source"]:
        test_suit.run_tests(data, llm_code, True)
        test_suit.save_test_case_outcomes(data, llm_code)

    # Run tests if specified in conf or if new translations have been generated
    if config["steps"]["run_tests"]:
        test_suit.run_tests(data, llm_code)
        test_suit.save_test_case_outcomes(data, llm_code)

    # Run code smell detection if specified in conf
    if config["steps"]["run_code_smell_detection"]:
        run_code_smell_detection.run_code_smell_detection(data, llm_code)

    if config["steps"]["run_tests"] or config["steps"]["run_tests_source"]:
        # stop docker if it was started
        c.stop()


def prompt_crafting_pipeline(data, llm_code):
    for iteration in range(config['prompt_crafting_current_iteration'] + 1,
                           config["prompt_crafting_iterations"] + 1):
        config["prompt_crafting"] = True
        config["prompt_crafting_current_iteration"] = iteration
        if "run_translation" not in config["steps"]:
            config["steps"]["run_translation"] = config["steps"]["read_translations_from_disk"]
        llm_code = run_translation.run_translations(data, llm_code)
        test_suit.run_tests(data, llm_code)
        run_code_smell_detection.run_code_smell_detection(data, llm_code)

    test_suit.save_llm_testcase_outcomes(llm_code)


def main():
    os.makedirs('./tcos/src', exist_ok=True)
    os.makedirs('./tcos/llm', exist_ok=True)

    start = time.time()

    data = prepare_dataset.load_to_cache("translation")
    items = list(data.items())
    random.shuffle(items)
    n_procs = 1
    chunksize = math.ceil(len(data) / n_procs)
    chunks = [items[i:i + chunksize] for i in range(0, len(items), chunksize)]

    # load existing test case results and distribute only the relevant parts across the processes to save memory
    if os.path.exists(os.path.join(config["tco_path"], "tco_llm.csv")):
        tcos = pd.read_csv(os.path.join(config["tco_path"], "tco_llm.csv"))
        tco_chunks = [tcos[tcos["Problem ID"].isin(list(map(lambda e: e[0][1], c)))] for c in chunks]
    else:
        tco_chunks = [None for _ in chunks]

    # load existing code smell results and distribute only the relevant parts across the processes to save memory
    if os.path.exists(os.path.join(config["base_path_data"], "llm_code_smell_results.csv")):
        cs_llm_df = pd.read_csv(os.path.join(config["base_path_data"], "llm_code_smell_results.csv"), sep='@', quotechar="'")
        cs_chunks = [cs_llm_df[cs_llm_df["path"].str.contains("|".join(list(map(lambda e: e[0][1], c))))] for c in chunks]
    else:
        cs_chunks = [None for _ in chunks]

    n_procs = len(chunks)
    print(f'Spawning {n_procs} processes from {os.getpid()}')
    with Pool(n_procs) as p:
        p.starmap(pipeline, zip(chunks, tco_chunks, cs_chunks))

    # combine test case outcomes of each process to one single dataframe
    if config['steps']['run_tests'] or config['steps']['run_prompt_crafting']:
        utility.combine_testcase_outcome_data_frames(os.path.join('.', 'tcos'))

    if config['steps']['run_code_smell_detection']:
        utility.combine_code_smell_results(config["base_path_data"])

    print(f'pipeline took {time.time()-start}s')


if __name__ == "__main__":
    main()
