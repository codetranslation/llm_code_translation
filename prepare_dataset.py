import logging
import pandas as pd
import os
import ast

import utility
from config import config
from datastructures.Data import Data
from datastructures.Testcase import Testcase
from datastructures.Codesample import get_suffix
import constants

data: dict = {}


def load_to_cache(subset: str):
    if constants.evo_eval in config["steps"]["prepare_datasets"]:
        load_to_cache_eval(subset, "evoeval")
    if constants.evalplus in config["steps"]["prepare_datasets"]:
        load_to_cache_eval(subset, "evalplus")
    if constants.codenet in config["steps"]["prepare_datasets"]:
        load_to_cache_codenet(subset)
    if constants.avatar in config["steps"]["prepare_datasets"]:
        load_to_cache_avatar(subset)
    return data


def load_to_cache_eval(subset: str, dataset: str):
    base_path_data = os.path.join(config["base_path_data"], dataset)
    data_subset = pd.read_csv(os.path.join(base_path_data, f"{subset}_data_{dataset}.csv"))

    # neglect can be used to discard elements which, e.g., do not pass tests in the testing environment
    data_subset = data_subset[~data_subset["neglect"]]
    for s in data_subset["id"]:
        problem_id = s.split(".")[0]

        # only_specific_tasks is a performance optimisation if the pipeline should refer to only part of the dataset
        if config["steps"]["only_specific_tasks"] and problem_id not in config["steps"]["specific_tasks"][dataset]:
            continue

        cs_python = utility.cheap_codesample_factory(submission_id=problem_id,
                                                     path=os.path.join(base_path_data, "data", s),
                                                     dataset=dataset, task_id=problem_id)

        # load test cases
        tc_file = f'tc_{problem_id}.txt'
        with open(os.path.join(base_path_data, 'test_cases', 'in', tc_file), 'r', encoding='utf-8') as f:
            f_in = f.readlines()
        with open(os.path.join(base_path_data, 'test_cases', 'out', tc_file), 'r', encoding='utf-8') as f:
            f_out = f.readlines()

        tcs = []

        for tc_id, (tc_in, tc_out) in enumerate(zip(f_in, f_out)):
            tin = '\n'.join(tc_in.split("NLITC"))
            tout = '\n'.join(tc_out.split("NLITC"))
            tcs.append(Testcase(testcase_id=f'tc_{tc_id}', input=tin, output=tout))

        description = os.path.join(base_path_data, 'problem_description', s.replace('py', 'txt'))
        with open(description, 'r', encoding='utf-8') as f:
            description = f.read()

        data[(dataset, problem_id)] = Data(dataset=dataset, problem_id=problem_id,
                                           task_description=description, testcases=tcs,
                                           code_samples=[cs_python])

    logging.debug(data)


def load_to_cache_avatar(subset: str):
    dataset = constants.avatar
    base_path_data = os.path.join(config["base_path_data"], dataset)
    data_subset = pd.read_csv(os.path.join(base_path_data, f"{subset}_data_{dataset}.csv"))

    for idx, row in data_subset.iterrows():
        avatar_id: str = row["avatar_id"]

        if config["steps"]["only_specific_tasks"] and avatar_id not in config["steps"]["specific_tasks"][dataset]:
            continue

        if avatar_id.startswith("atcoder"):
            atcoder_id, level = avatar_id.split("_")[1], avatar_id.split("_")[2]
            codesamples = []
            for source_pl in ["java", "python"]:
                ext = get_suffix(source_pl)
                for submission in ast.literal_eval(row[source_pl]):
                    submission_id = submission
                    if submission_id + get_suffix(source_pl) in row["neglect"]:
                        continue
                    path = os.path.join(base_path_data, "data", atcoder_id, level, source_pl, submission_id + ext)
                    cs = utility.cheap_codesample_factory(submission_id=submission_id, path=path,
                                                          dataset=constants.avatar, task_id=avatar_id)
                    codesamples.append(cs)

            if len(codesamples) == 0:
                continue

            tcs = []
            testcases_path = os.path.join(base_path_data, 'test_cases', atcoder_id, level)
            testcases_in = os.listdir(os.path.join(testcases_path, 'in'))
            testcases_in = list(filter(lambda e: not e.startswith("."), testcases_in))
            testcases_out = os.listdir(os.path.join(testcases_path, 'out'))
            testcases_out = list(filter(lambda e: not e.startswith("."), testcases_out))

            testcases_in.sort()
            testcases_out.sort()

            for i in range(len(testcases_in)):
                t_in = utility.load_content_from_file(os.path.join(testcases_path,
                                                                   'in', testcases_in[i], testcases_in[i])).strip()
                t_out = utility.load_content_from_file(os.path.join(testcases_path,
                                                                    'out', testcases_out[i], testcases_out[i])).strip()
                t_id = '_'.join(['tc', atcoder_id, testcases_in[i].split('.')[0]])
                tcs.append(Testcase(testcase_id=t_id, input=t_in, output=t_out))

            data[(constants.avatar, avatar_id)] = Data(dataset=constants.avatar,
                                                       problem_id=avatar_id,
                                                       task_description='',
                                                       testcases=tcs,
                                                       code_samples=codesamples)
        else:
            codeforces_id = avatar_id.replace("codeforces_", "")
            codesamples = []
            for source_pl in ["java", "python"]:
                ext = get_suffix(source_pl)
                for submission in ast.literal_eval(row[source_pl]):
                    submission_id = submission
                    if submission_id + get_suffix(source_pl) in row["neglect"]:
                        continue
                    path = os.path.join(base_path_data, "data", codeforces_id, source_pl, submission_id + ext)
                    cs = utility.cheap_codesample_factory(submission_id=submission_id, path=path,
                                                          dataset=constants.avatar, task_id=avatar_id)
                    codesamples.append(cs)

            if len(codesamples) == 0:
                continue

            tcs = []
            testcases_path = os.path.join(base_path_data, 'test_cases', codeforces_id, "samples")
            testcases_folders = os.listdir(testcases_path)
            testcases_in = list(filter(lambda e: "input" in e and not e.startswith("."), testcases_folders))
            testcases_out = list(filter(lambda e: "output" in e and not e.startswith("."), testcases_folders))

            testcases_in.sort()
            testcases_out.sort()

            for i in range(len(testcases_in)):
                t_in = utility.load_content_from_file(os.path.join(testcases_path,
                                                                   testcases_in[i], testcases_in[i])).strip()
                t_out = utility.load_content_from_file(
                    os.path.join(testcases_path, testcases_out[i], testcases_out[i])).strip()
                t_id = '_'.join(['tc', avatar_id, testcases_in[i].split('_')[0]])
                tcs.append(Testcase(testcase_id=t_id, input=t_in, output=t_out))

            data[(constants.avatar, avatar_id)] = Data(dataset=constants.avatar,
                                                       problem_id=avatar_id,
                                                       task_description='',
                                                       testcases=tcs,
                                                       code_samples=codesamples)

    logging.debug(data)


def load_to_cache_codenet(subset: str):
    dataset = constants.codenet
    base_path_data = os.path.join(config["base_path_data"], dataset)
    data_subset = pd.read_csv(os.path.join(base_path_data, f"{subset}_data_{dataset}.csv"))

    for idx, row in data_subset.iterrows():
        problem_id = row["id"]

        if config["steps"]["only_specific_tasks"] and problem_id not in config["steps"]["specific_tasks"][dataset]:
            continue

        codesamples = []
        for source_pl in ["java", "python", "rust", "scala"]:
            ext = get_suffix(source_pl)
            for submission in ast.literal_eval(row[source_pl]):
                if submission + get_suffix(source_pl) in row["neglect"]:
                    continue
                path = os.path.join(base_path_data, "data", row["id"], source_pl, submission + ext)
                task_id = f"{problem_id}_{submission}"

                cs = utility.cheap_codesample_factory(submission_id=submission, path=path, dataset=constants.codenet,
                                                      task_id=task_id)
                codesamples.append(cs)

        if len(codesamples) == 0:
            continue

        # fetch corresponding testcases
        testcases_path = os.path.join(base_path_data, 'test_cases', problem_id)
        tc_in = utility.load_content_from_file(os.path.join(testcases_path, 'input.txt'))
        tc_out = utility.load_content_from_file(os.path.join(testcases_path, 'output.txt'))
        testcases = [Testcase(f'tc_{problem_id}_0', tc_in, tc_out)]  # only one tc in codenet

        data[(dataset, problem_id)] = Data(dataset=dataset, problem_id=problem_id, task_description='',
                                           testcases=testcases, code_samples=codesamples)

    logging.debug(data)
