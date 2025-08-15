from utility import save_code_smell_result_data, save_code_smell_result_llm_code
from itertools import chain


def run_code_smell_detection(data, llm_code):
    input_code_samples = list(chain.from_iterable([obj.code_samples for (_, _), obj in data.items()]))
    for cs in input_code_samples:
        cs.run_code_smell_detection()
    for (dataset, task_id), obj in llm_code.items():
        for llm_model, code_samples_per_it in obj.items():
            for it, code_samples in code_samples_per_it.items():
                for code_sample in code_samples:
                    code_sample.run_code_smell_detection()

    # save code smells
    save_code_smell_result_data(data)
    save_code_smell_result_llm_code(llm_code)
