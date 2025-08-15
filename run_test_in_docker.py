import os
import subprocess
import json
import sys
from enum import Enum
import logging
import math

logging.basicConfig(filename="testing_logger.log", level=logging.DEBUG)


class TestcaseResult(Enum):
    NOT_EVALUATED = 1
    PASSED = 2
    FAILED = 3
    EXCEPTION = 4
    COMPILER_ERROR = 5


def main():
    pl = sys.argv[1]
    submission_id = sys.argv[2]
    pa = sys.argv[3]
    ds = sys.argv[4]
    with open("tcs.json", "r") as f:
        tcs = json.load(f)

    ret_value, stderr = compile_code(pl)
    if not ret_value == 0:
        tcos = [{"testcase_id": tc["id"], "testcase_outcome": TestcaseResult.COMPILER_ERROR.value, "error": stderr,
                     "stacktrace": stderr} for tc in tcs]
        with open("result.json", "w") as f:
            if len(json.dumps(tcos)) > 1024 * 1024 * 1024:
                return 137
            else:
                json.dump(tcos, f)
        return 11

    tcos = []
    for tc in tcs:
        error = None
        stacktrace = None
        outcome = TestcaseResult.NOT_EVALUATED

        tc_inp = get_input(pl, tc)

        p = subprocess.Popen(get_exec_cmd(pl),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:
            timeout = False
            output, stderr = p.communicate(tc_inp, timeout=20)
        except subprocess.TimeoutExpired as e:
            p.kill()
            output = stderr = "ERROR: TIMEOUT"
            timeout = True
        if p.returncode == 0 and not timeout:
            if ds == "avatar":
                output = output.replace(" ", "")
                tc["output"] = tc["output"].replace(" ", "")
            if compare_output(tc["output"], output):
                logging.debug(f'Correct TC result for {tc["id"]} for submission {submission_id} at {pa}')
                outcome = TestcaseResult.PASSED
            else:
                logging.debug(f'Wrong TC result: Is: "{output.strip()}", Should: "{tc["output"].strip()}", For: {tc["id"]} for submission {submission_id} at {pa}')
                outcome = TestcaseResult.FAILED
        else:
            logging.error(f'Could not execute test case! {tc["id"]} for submission {submission_id}'
                          f' at {pa}. Reason: {stderr}')
            error, stacktrace = parse_error_stacktrace(pl, stderr)
            outcome = TestcaseResult.EXCEPTION
            if "compilation failed" in error:
                outcome = TestcaseResult.COMPILER_ERROR
        tcos.append({"testcase_id": tc["id"], "testcase_outcome": outcome.value, "error": error,
                     "stacktrace": stacktrace})

        if not outcome == TestcaseResult.PASSED:
            break  # early exit

    with open("result.json", "w") as f:
        if len(json.dumps(tcos)) > 1024 * 1024 * 1024:
            return 137
        else:
            json.dump(tcos, f)

    cleanup(pl)


def get_input(pl, tc):
    if pl == "python":
        return tc["input"]
    elif pl == "java" or pl == "scala" or pl == "rust":
        try:
            java_in = '\n'.join([eval(i) for i in tc.input.split('\n')])  # ensure correct transmission of strings
        except:
            java_in = tc["input"]  # fallback in case of int
        if java_in == '':
            java_in = '\n'  # avoid scanner has no line error
        return java_in


def parse_error_stacktrace(pl, stderr):
    if pl == "python":
        stacktrace = ''.join(
            [s for s in stderr.split('File') if not s.strip().startswith('"') or s.strip().startswith('".')])
        try:
            error = stderr.split('\n')[-2]
        except IndexError as _:
            error = stderr
        return error, stacktrace
    elif pl == "java":
        if stderr.startswith('Exception'):
            stderr = stderr.strip('Exception')
            stderr = stderr.split(':', maxsplit=1)
            error = stderr[0].split(' ')[-1].strip()
            stacktrace = stderr[1].strip()
        else:
            error = stderr.split('error:')[-1].strip()
            stacktrace = '\n'.join(stderr.split('\n')[:-1]).strip()
        return error, stacktrace
    elif pl == "scala":
        return stderr, stderr
    elif pl == "rust":
        return stderr.split(":")[0], stderr


def get_exec_cmd(pl):
    if pl == "python":
        return ['python3', '-u', 'main.py']
    elif pl == "java":
        return ['java', 'Main.java']
    elif pl == "scala":
        return ["scala", "-classpath", "scala_comp", "Main"]
    elif pl == "rust":
        return ["./main"]


def compile_code(pl):
    if pl == "scala":
        os.makedirs("scala_comp", exist_ok=True)
        p = subprocess.Popen(["scalac", "../Main.scala"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd="./scala_comp/")
        output, stderr = p.communicate()
        if "error" in stderr or "error" in output:
            return -1, stderr
        else:
            return 0, ""
    elif pl == "rust":
        p = subprocess.Popen(['rustc', 'main.rs'],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, stderr = p.communicate()
        if "error" in stderr:
            return -1, stderr
        else:
            return 0, ""
    else:
        return 0, ""


def compare_output(true_output, to_be_checked):
    if to_be_checked.strip() == true_output.strip():
        return True
    else:
        try:
            true_output = float(true_output)
            to_be_checked = float(to_be_checked)
            return math.isclose(true_output, to_be_checked, rel_tol=1e-5)
        except ValueError:
            return False
    return False


def cleanup(pl):
    if pl == "scala":
        try:
            [os.remove(os.path.join("scala_comp", f)) for f in os.listdir("scala_comp")]
        except IsADirectoryError:
            pass
    elif pl == "rust":
        os.remove("main")


if __name__ == "__main__":
    main()
