from typing import List

from datastructures.Testcase import Testcase
from datastructures.Codesample import Codesample

class Data:
    def __init__(self, dataset: str, problem_id: str, task_description: str,
                 testcases: List[Testcase], code_samples: List[Codesample]):

        self.dataset = dataset
        self.problem_id = problem_id
        self.task_description = task_description
        self.testcases = testcases
        self.code_samples = code_samples

    dataset: str
    problem_id: str
    task_description: str
    testcases: List[Testcase]
    code_samples: List[Codesample]
