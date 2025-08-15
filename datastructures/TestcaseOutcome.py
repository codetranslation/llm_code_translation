from enum import Enum


class TestcaseResult(Enum):
    NOT_EVALUATED = 1
    PASSED = 2
    FAILED = 3
    EXCEPTION = 4
    COMPILER_ERROR = 5
    TIMEOUT = 6
    OOM = 7


class TestcaseOutcome:
    def __init__(self, testcase_id: str, testcase_outcome: TestcaseResult, error=None, stacktrace=None):
        self.testcase_id = testcase_id
        self.testcase_outcome = testcase_outcome
        self.error = error
        self.stacktrace = stacktrace

    testcase_id: str
    testcase_outcome: TestcaseResult
    error: str
    stacktrace: str
