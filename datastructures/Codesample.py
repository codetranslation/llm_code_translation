import logging
import os
import shutil
import random
import json
import xml.etree.ElementTree
from enum import Enum
from typing import List
import xml.etree.ElementTree as ET
import docker

import constants
from config import config
from datastructures.Testcase import Testcase
from datastructures.TestcaseOutcome import TestcaseResult, TestcaseOutcome


logger = logging.getLogger("my_project_logger")


def get_suffix(programming_language):
    if programming_language == constants.python:
        return constants.python_ext
    elif programming_language == constants.java:
        return constants.java_ext
    elif programming_language == constants.scala:
        return constants.scala_ext
    elif programming_language == constants.rust:
        return constants.rust_ext
    else:
        raise NotImplementedError()


def get_language_by_suffix(suffix: str):
    if not suffix.startswith("."):
        suffix = "." + suffix
    if suffix == constants.python_ext:
        return constants.python
    elif suffix == constants.java_ext:
        return constants.java
    elif suffix == constants.scala_ext:
        return constants.scala
    elif suffix == constants.rust_ext:
        return constants.rust
    else:
        raise NotImplementedError()


def get_target_language(source_language: str):
    return filter(lambda e: e != source_language, config["programming_languages"])


def clippy_weights(level: str) -> int:
    if level == "warning":
        return 1
    if level == "error":
        return 2


def get_filename_testing(programming_language):
    if programming_language == constants.python:
        filename = "main.py"
    elif programming_language == constants.java:
        filename = "Main.java"
    elif programming_language == constants.scala:
        filename = "Main.scala"
    elif programming_language == constants.rust:
        filename = "main.rs"
    else:
        logger.error("Not implemented programming language passed.")
        raise NotImplementedError()
    return filename


class Codesample:
    def __init__(self, submission_id: str, path: str, dataset: str, task_id: str, prompt_id: str = "default"):
        self.submission_id = submission_id
        self.path = path
        self.dataset = dataset
        self.task_id = task_id
        self.programming_language = ""
        self.code_smell_result = ""
        self.code_sample_status = CodeSampleStatus.NOT_EVALUATED
        self.testcase_outcomes = []
        self.fraction_passed = 0
        self.is_best_sample = False
        self.prompt_id = prompt_id

    def get_testcase_for_id(self, tc_id: str, data: dict):
        d = data[(self.dataset, self.task_id)]
        tcs = list(filter(lambda tc: tc.testcase_id == tc_id, d.testcases))
        if len(tcs) == 0:
            return None
        else:
            return tcs[0]

    def get_target_languages(self):
        return get_target_language(self.programming_language)

    def update_test_case_stats(self, n_tcs):
        self.fraction_passed = sum(
            [1 for tco in self.testcase_outcomes if tco.testcase_outcome == TestcaseResult.PASSED]) / n_tcs

        if self.fraction_passed == 0:
            self.code_sample_status = CodeSampleStatus.FAILED
        elif self.fraction_passed == 1:
            self.code_sample_status = CodeSampleStatus.PASSED
        else:
            self.code_sample_status = CodeSampleStatus.PARTIALLY_PASSED

    def get_random_failed_test_case(self):
        failed = [tco for tco in self.testcase_outcomes if tco.testcase_outcome == TestcaseResult.FAILED]
        if len(failed) > 0:
            return random.choice(failed)
        return None

    def get_random_exception_test_case(self):
        exception = [tco for tco in self.testcase_outcomes if tco.testcase_outcome == TestcaseResult.EXCEPTION]
        if len(exception) > 0:
            return random.choice(exception)
        return None

    def get_random_OOM_test_case(self):
        exception = [tco for tco in self.testcase_outcomes if tco.testcase_outcome == TestcaseResult.OOM]
        if len(exception) > 0:
            return random.choice(exception)
        return None

    def get_random_compiler_error_test_case(self):
        exception = [tco for tco in self.testcase_outcomes if tco.testcase_outcome == TestcaseResult.COMPILER_ERROR]
        if len(exception) > 0:
            return random.choice(exception)
        return None

    def get_random_timeout_test_case(self):
        exception = [tco for tco in self.testcase_outcomes if tco.testcase_outcome == TestcaseResult.TIMEOUT]
        if len(exception) > 0:
            return random.choice(exception)
        return None

    def get_code_smell_severity(self):
        pass

    def is_code_sample_solved(self):
        pass

    def set_is_best_sample(self, status=True):
        self.is_best_sample = status

    def get_is_best_sample(self):
        return self.is_best_sample

    def set_cs_result(self, inp):
        pass

    def get_cs_result(self):
        return self.code_smell_result

    def log_test_execution(self):
        logger.info(
            f"{self.fraction_passed} passed acc. to test execution for for ds={self.dataset}, task={self.task_id}, submission={self.submission_id}, prompt_id={self.prompt_id}, pl={self.programming_language}")

    def log_code_smell_detection(self, ret_value):
        logger.info(
            f"ret_value={ret_value} for code smell detection for for ds={self.dataset}, task={self.task_id}, submission={self.submission_id}, prompt_id={self.prompt_id}, pl={self.programming_language}, path={self.path}")

    def log_docker_test_execution(self, ret_value, output):
        logger.info(
            f"ret_value={ret_value}, output={output} for ds={self.dataset}, task={self.task_id}, submission={self.submission_id}, prompt_id={self.prompt_id}, pl={self.programming_language}, path={self.path}; pid={os.getpid()}")

    def evaluate_tcs_docker(self, programming_language: str, tcs: List[Testcase]):
        filename = get_filename_testing(programming_language)
        pd = os.path.join(config["base_path_testing_docker"], str(os.getpid()))
        tcs_json = list(map(lambda tc: {"id": tc.testcase_id, "input": tc.input, "output": tc.output}, tcs))
        shutil.copyfile(self.path, os.path.join(pd, filename))
        with open(os.path.join(pd, "tcs.json"), "w") as f:
            json.dump(tcs_json, f)

        client = docker.from_env()
        c = client.containers.get(f"testing{os.getpid()}")
        ret_value, result = c.exec_run(f"timeout -k 10 1200 python3 run_test_in_docker.py {programming_language} {self.submission_id} {self.path} {self.dataset}", stdout=True)
        try:
            result = result.decode("utf-8")
        except UnicodeDecodeError:
            ret_value = 100
        self.log_docker_test_execution(ret_value, result)

        if ret_value == 0:
            with open(os.path.join(pd, "result.json"), "r") as f:
                tcos = json.load(f)
            tcos = list(
                map(lambda tco: TestcaseOutcome(tco["testcase_id"], TestcaseResult(tco["testcase_outcome"]), tco["error"],
                                                tco["stacktrace"]), tcos))

            os.remove(os.path.join(pd, filename))
        else:
            if ret_value == 124:
                tcr = TestcaseResult.TIMEOUT
            elif ret_value == 137:
                tcr = TestcaseResult.OOM
            else:
                tcr = TestcaseResult.EXCEPTION
            tcos = list(map(lambda tc: TestcaseOutcome(tc.testcase_id, tcr, "Overall timeout exceeded / OOM / Other", ""), tcs))
        self.testcase_outcomes = tcos
        self.update_test_case_stats(len(tcs))
        self.log_test_execution()


class PySample(Codesample):
    def __init__(self, submission_id: str, path: str, dataset: str, task_id: str, prompt_id: str = "default"):
        super().__init__(submission_id, path, dataset, task_id, prompt_id)
        self.programming_language = constants.python

    # execute TestCase and return TestCaseResult
    def evaluate_testcases(self, tcs: List[Testcase]):
        if len(self.testcase_outcomes) != 0:
            return

        self.evaluate_tcs_docker(constants.python, tcs)

    def run_code_smell_detection(self):
        if not self.code_smell_result == "":
            return
        # trigger PyLint execution
        client = docker.from_env()
        c = client.containers.get(f"testing{os.getpid()}")

        filename = get_filename_testing(self.programming_language)
        pd = os.path.join(config["base_path_testing_docker"], str(os.getpid()))
        shutil.copyfile(self.path, os.path.join(pd, filename))

        ret_value, result = c.exec_run(
            f"timeout -k 10 60 pylint --recursive=y {filename} -j 0 --output-format=json",
            stdout=True)
        try:
            result = result.decode("utf-8")
        except UnicodeDecodeError:
            self.code_smell_result = constants.cs_parsing_error
            self.log_code_smell_detection(ret_value)
            return

        if ret_value == 32:
            logger.warning(f"Code smell detection (Pylint) failed for {self.submission_id}; error: {result}, "
                           f"return value: {ret_value}")
            self.code_smell_result = constants.cs_parsing_error
        else:
            try:
                self.code_smell_result = json.loads(result)  # string to json
            except json.decoder.JSONDecodeError:
                self.code_smell_result = constants.cs_parsing_error
            self.log_code_smell_detection(ret_value)

    def set_cs_result(self, inp: str):
        try:
            self.code_smell_result = json.loads(inp)  # string to json
        except json.decoder.JSONDecodeError:
            self.code_smell_result = constants.cs_parsing_error

    def get_code_smell_severity(self):
        if self.code_smell_result == constants.cs_parsing_error:
            return 10
        return len(self.code_smell_result)

    def is_code_sample_solved(self):
        return self.get_code_smell_severity() == 0 and (self.code_sample_status == CodeSampleStatus.PASSED)

    def get_code_smell_violations(self):
        violations = []
        for code_smell in self.code_smell_result:
            violations.append(f'Line {code_smell["line"]}: {code_smell["message"]}')
        return violations


class JSample(Codesample):
    def __init__(self, submission_id: str, path: str, dataset: str, task_id: str, prompt_id: str = "default"):
        super().__init__(submission_id, path, dataset, task_id, prompt_id)
        self.programming_language = "java"

    # execute TestCase and return TestCaseResult
    def evaluate_testcases(self, tcs: List[Testcase]):
        if len(self.testcase_outcomes) != 0:
            return

        self.evaluate_tcs_docker(constants.java, tcs)

    def run_code_smell_detection(self):
        if not self.code_smell_result == "":
            return
        # trigger PMD execution
        client = docker.from_env()
        c = client.containers.get(f"testing{os.getpid()}")

        filename = get_filename_testing(self.programming_language)
        pd = os.path.join(config["base_path_testing_docker"], str(os.getpid()))
        shutil.copyfile(self.path, os.path.join(pd, filename))

        ret_value, result = c.exec_run(
            f"timeout -k 10 60 /root/pmd-bin-7.1.0/bin/pmd check -d {filename} -R rulesets/java/quickstart.xml -f json",
            stdout=True)
        try:
            result = result.decode("utf-8")
        except UnicodeDecodeError:
            self.code_smell_result = constants.cs_parsing_error
            self.log_code_smell_detection(ret_value)
            return
        self.log_code_smell_detection(ret_value)

        if ret_value == 1 or ret_value == 2:
            logger.warning(f"Code smell detection (PMD) failed for {self.submission_id}; error: {result}, "
                           f"return value: {ret_value}")
            self.code_smell_result = constants.cs_parsing_error
        else:
            try:
                result = "\n".join(result.split("\n")[2:])
                self.code_smell_result = json.loads(result)  # string to json
            except (json.decoder.JSONDecodeError, IndexError):
                self.code_smell_result = constants.cs_parsing_error

    def set_cs_result(self, inp: str):
        try:
            self.code_smell_result = json.loads(inp)  # string to json
        except json.decoder.JSONDecodeError:
            self.code_smell_result = constants.cs_parsing_error

    def get_code_smell_severity(self):
        if self.code_smell_result == constants.cs_parsing_error:
            return 10
        try:
            return len(self.code_smell_result['files'][0]['violations'])
        except IndexError:
            return 10
        except TypeError:
            logger.warning(f"Code smell results were of invalid type. Found {self.code_smell_result} for {self.path} (PL = {self.programming_language}=")
            return 10

    def is_code_sample_solved(self):
        return self.get_code_smell_severity() == 0 and self.code_sample_status == CodeSampleStatus.PASSED

    def get_code_smell_violations(self):
        violations = []
        for v in self.code_smell_result['files'][0]['violations']:
            violations.append(f'Line {v["beginline"]} to {v["endline"]}: {v["description"]}')
        return violations


class ScalaSample(Codesample):
    def __init__(self, submission_id: str, path: str, dataset: str, task_id: str, prompt_id: str = "default"):
        super().__init__(submission_id, path, dataset, task_id, prompt_id)
        self.programming_language = constants.scala

    # execute TestCase and return TestCaseResult
    def evaluate_testcases(self, tcs: List[Testcase]):
        if len(self.testcase_outcomes) != 0:
            return

        self.evaluate_tcs_docker(constants.scala, tcs)

    def run_code_smell_detection(self):
        if not self.code_smell_result == "":
            return

        client = docker.from_env()
        c = client.containers.get(f"testing{os.getpid()}")

        filename = get_filename_testing(self.programming_language)
        pd = os.path.join(config["base_path_testing_docker"], str(os.getpid()))
        shutil.copyfile(self.path, os.path.join(pd, filename))

        ret_value, result = c.exec_run(
            f"timeout -k 10 60 /root/scalastyle --xmlOutput /usr/src/app/scalastyle_output.xml /usr/src/app/{filename}",
            stdout=True)
        try:
            result = result.decode("utf-8")
        except UnicodeDecodeError:
            self.code_smell_result = constants.cs_parsing_error
            self.log_code_smell_detection(ret_value)
            return

        if ret_value == 0:
            try:
                self.code_smell_result = ET.parse(os.path.join(pd, "scalastyle_output.xml"))
            except xml.etree.ElementTree.ParseError:
                self.code_smell_result = constants.cs_parsing_error
        else:
            self.code_smell_result = constants.cs_parsing_error
            logger.warning(f"Code smell detection (scalastyle) failed for {self.submission_id}; error: {result}, "
                           f"return value: {ret_value}")
        self.log_code_smell_detection(ret_value)

    def set_cs_result(self, inp: str):
        try:
            self.code_smell_result = ET.ElementTree(ET.fromstring(inp))
        except xml.etree.ElementTree.ParseError:
            self.code_smell_result = constants.cs_parsing_error

    def get_code_smell_severity(self):
        if self.code_smell_result == constants.cs_parsing_error:
            return 10
        try:
            return len([e.attrib["severity"] for e in self.code_smell_result.iter() if e.tag == "error"])
        except IndexError:
            return 10

    def is_code_sample_solved(self):
        return self.get_code_smell_severity() == 0 and self.code_sample_status == CodeSampleStatus.PASSED

    def get_code_smell_violations(self):
        return [str(e.attrib) for e in self.code_smell_result.iter() if e.tag == "error"]


class RustSample(Codesample):
    def __init__(self, submission_id: str, path: str, dataset: str, task_id: str, prompt_id: str = "default"):
        super().__init__(submission_id, path, dataset, task_id, prompt_id)
        self.programming_language = constants.rust

    # execute TestCase and return TestCaseResult
    def evaluate_testcases(self, tcs: List[Testcase]):
        if len(self.testcase_outcomes) != 0:
            return

        self.evaluate_tcs_docker(constants.rust, tcs)

    def run_code_smell_detection(self):
        if not self.code_smell_result == "":
            return

        client = docker.from_env()
        c = client.containers.get(f"testing{os.getpid()}")

        filename = get_filename_testing(self.programming_language)
        pd = os.path.join(config["base_path_testing_docker"], str(os.getpid()))
        if not os.path.exists(os.path.join(pd, "rcargo")):
            ret_value, result = c.exec_run("cargo new rcargo", stdout=True)
            result = result.decode("utf-8")
            if not ret_value == 0:
                logger.error("ERROR: Could not create cargo env; res = {result}")
                return -1
            ret_value, result = c.exec_run("chmod 0777 rcargo -R", stdout=True)
            result = result.decode("utf-8")
            if not ret_value == 0:
                logger.error("ERROR: Could not create cargo env; res = {result}")
                return -1

        shutil.copyfile(self.path, os.path.join(pd, "rcargo", "src", filename))

        ret_value, result = c.exec_run(
            f'sh -c "timeout -k 10 60 cargo clippy --message-format=json | tee current_clippy_out_rust.json"', stdout=True, workdir="/usr/src/app/rcargo/")
        try:
            result = result.decode("utf-8")
        except UnicodeDecodeError:
            self.code_smell_result = constants.cs_parsing_error
            self.log_code_smell_detection(ret_value)
            return

        if not ret_value == 0:
            logger.warning(f"Code smell detection (rust / clippy) failed for {self.submission_id}; error: {result}, "
                           f"return value: {ret_value}")

        self.log_code_smell_detection(ret_value)

        if ret_value == 0:
            try:
                with open(os.path.join(pd, "rcargo", "current_clippy_out_rust.json"), "r") as f:
                    output = [json.loads(line) for line in f.readlines()]
                self.code_smell_result = output
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                self.code_smell_result = constants.cs_parsing_error
        else:
            self.code_smell_result = constants.cs_parsing_error

    def set_cs_result(self, inp: str):
        try:
            self.code_smell_result = json.loads(inp)
        except json.decoder.JSONDecodeError:
            self.code_smell_result = constants.cs_parsing_error

    def get_code_smell_severity(self):
        if self.code_smell_result == constants.cs_parsing_error:
            return 10
        try:
            return len([smell["message"]["level"] for smell in self.code_smell_result if "message" in smell and "level" in smell["message"]])
        except IndexError:
            return 10

    def is_code_sample_solved(self):
        return self.get_code_smell_severity() == 0 and self.code_sample_status == CodeSampleStatus.PASSED

    def get_code_smell_violations(self):
        return [smell["message"]["code"]["code"] for smell in self.code_smell_result if
         "message" in smell and "code" in smell["message"]]


class CodeSampleStatus(Enum):
    NOT_EVALUATED = 1
    PASSED = 2
    PARTIALLY_PASSED = 3
    FAILED = 4
