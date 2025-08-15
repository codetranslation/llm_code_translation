import os
import logging
import pandas as pd
from typing import List

import utility
from config import config
from datastructures.Testcase import Testcase
from datastructures.TestcaseOutcome import TestcaseOutcome


logger = logging.getLogger("my_project_logger")


def run_tests(data, llm_code, run_source_tc=False):
    total_len = len(llm_code.items())
    for idx, (meta, code_samples_per_llm) in enumerate(llm_code.items()):
        logger.info(f"Now llm code item {idx}/{total_len} for process id {os.getpid()}")
        src = data[meta]
        # retrieve test cases out of data container
        testcases: List[Testcase] = src.testcases
        # perform tests on source
        if run_source_tc:
            for code_sample in src.code_samples:
                code_sample.evaluate_testcases(testcases)
        else:
            # perform tests on all translations
            for llm, code_samples_per_it in code_samples_per_llm.items():
                for it, code_samples in code_samples_per_it.items():
                    for code_sample in code_samples:
                        code_sample.evaluate_testcases(testcases)


def save_test_case_outcomes(data, llm_code):
    content = []
    for k, code_sample in data.items():
        dataset = code_sample.dataset
        problem_id = code_sample.problem_id
        for cs in code_sample.code_samples:
            for tco in cs.testcase_outcomes:
                content.append(utility.get_tco_row_from_data(dataset, problem_id, cs.path, tco.testcase_id,
                                                             tco.testcase_outcome, tco.error, tco.stacktrace))

    df = pd.DataFrame(content)
    with open(f'./tcos/src/tco_{os.getpid()}.csv', "w", newline='') as f:
        df.to_csv(f, index=False)
    save_llm_testcase_outcomes(llm_code)


def save_llm_testcase_outcomes(llm_code):
    path = f'./tcos/llm/tco_{os.getpid()}.csv'

    content = []

    for k, llm_dict in llm_code.items():
        dataset = k[0]
        problem_id = k[1]
        for llm, it_dict in llm_dict.items():
            for it, code_samples in it_dict.items():
                for n_translation, code_sample in enumerate(code_samples):
                    for tco in code_sample.testcase_outcomes:
                        content.append(utility.get_tco_row_from_data(dataset, problem_id, code_sample.path,
                                                                     tco.testcase_id, tco.testcase_outcome, tco.error,
                                                                     tco.stacktrace, llm))
    df = pd.DataFrame(content)
    with open(path, "w", newline='') as f:
        df.to_csv(f, index=False)


def load_test_case_outcomes(data, llm_code):
    if not os.path.exists('./tco_src.csv') or not os.path.exists('./tco_llm.csv'):
        return True
    df_data = pd.read_csv('./tco_src.csv')
    df_llm = pd.read_csv('./tco_llm.csv')

    df_llm = df_llm[df_llm.Iteration <= config['prompt_crafting_current_iteration']]

    new_test_case_results = False

    _, problem_ids = zip(*list(data.keys()))
    df_data = df_data[df_data['Problem ID'].isin(problem_ids)]
    _, problem_ids = zip(*list(data.keys()))
    df_llm = df_llm[df_llm['Problem ID'].isin(problem_ids)]

    # map test case outcomes
    for obj in data.values():
        for cs in obj.code_samples:
            tmp_df = df_data[df_data.Dataset == cs.dataset]
            tmp_df = tmp_df[tmp_df['Problem ID'] == cs.submission_id]
            if len(tmp_df) > 0:
                for _, tco in tmp_df.iterrows():
                    cs.testcase_outcomes.append(TestcaseOutcome(testcase_id=tco.Testcase_id,
                                                                testcase_outcome=tco.Outcome, error=tco.Error,
                                                                stacktrace=tco.Stacktrace))
            else:
                cs.evaluate_testcases(obj.testcases)
                new_test_case_results = True

    for meta, llms_dict in llm_code.items():
        for iterations_dict in llms_dict.values():
            for it, l in iterations_dict.items():
                tmp_df = df_llm[df_llm.Iteration == it]
                for cs in l:
                    problem_id = cs.submission_id.split('.')[0]
                    n_translation = int(problem_id.split('__')[0])
                    problem_id = problem_id.split('__')[-1]
                    tmp_df = tmp_df[tmp_df.Dataset == cs.dataset]
                    tmp_df = tmp_df[tmp_df['Problem ID'] == problem_id]
                    tmp_df = tmp_df[tmp_df['nTranslation'] == n_translation]
                    if len(tmp_df) > 0:
                        for _, tco in tmp_df.iterrows():
                            cs.testcase_outcomes.append(TestcaseOutcome(testcase_id=tco.Testcase_id,
                                                                        testcase_outcome=tco.Outcome, error=tco.Error,
                                                                        stacktrace=tco.Stacktrace))
                    else:
                        src = data[meta]
                        # retrieve test cases out of data container
                        testcases: List[Testcase] = src.testcases
                        cs.evaluate_testcases(testcases)
                        new_test_case_results = True

    return new_test_case_results
