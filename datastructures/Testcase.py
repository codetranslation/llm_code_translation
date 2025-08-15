from datastructures.TestcaseOutcome import TestcaseResult


class Testcase:
    def __init__(self, testcase_id: str, input: str, output: str, path: bool = False):
        import utility
        self.testcase_id = testcase_id
        if path:
            self.input_path = input
            self.output_path = output
            self.input = utility.load_content_from_file(input)
            self.output = utility.load_content_from_file(output)
        else:
            self.input = input
            self.output = output
        self.passed = TestcaseResult.NOT_EVALUATED

    testcase_id: str
    input_path: str
    input: str
    output_path: str
    output: str
    passed: TestcaseResult
