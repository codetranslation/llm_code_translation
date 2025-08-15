import constants

config = {
    'datasets': {'evalplus': {'random_sample_size_per_pl_max_translation': 200, 'random_sample_size_per_pl_max_prompt_choosing': 10},
                 'evoeval': {'random_sample_size_per_pl_max_translation': 200, 'random_sample_size_per_pl_max_prompt_choosing': 10},
                 'avatar': {'random_sample_size_per_pl_max_translation': 200, 'random_sample_size_per_pl_max_prompt_choosing': 10},
                 'codenet': {'random_sample_size_per_pl_max_translation': 200, 'random_sample_size_per_pl_max_prompt_choosing': 10}},
    'LLMs': {'deepseek-ai/deepseek-coder-33b-instruct': {'max_attempts': 3}, 'mistralai/Mixtral-8x7B-Instruct-v0.1': {'max_attempts': 3}, 'ibm-granite/granite-34b-code-instruct': {'max_attempts': 3}, 'ise-uiuc/Magicoder-S-DS-6.7B': {'max_attempts': 3}, 'meta-llama/CodeLlama-70b-Instruct-hf': {'max_attempts': 3}, 'meta-llama/Meta-Llama-3-8B-Instruct': {'max_attempts': 3}, 'meta-llama/Meta-Llama-3-70B-Instruct': {'max_attempts': 3}, 'mistralai/Codestral-22B-v0.1': {'max_attempts': 3}},
    'steps': {
        'prepare_datasets': ['codenet', 'avatar', 'evalplus', 'evoeval'],  # datasets to be loaded; should at least contain one entry
        'only_specific_tasks': False,
        'specific_tasks': {
            'codenet': [],
            'avatar': [],
            "evalplus": [],
            "evoeval": [],
            "manual": []
        },
        'run_translation': [],  # models used for translation; should at least contain one entry if translations are to be run (otherwise set to False)
        'preprocess_translations': [],  # to parse code from LLM reponse; add LLMs for which preprocessing should be conducted
        'read_translations_from_disk': [], # read translations for these models from disk if available; either run_translation or read_translations_from_disk should be set
        'run_tests': False,
        'run_tests_source': False,
        'run_code_smell_detection': False,
        'run_prompt_crafting': False,
        'run_backtranslation': False,
        'run_manual_analysis': False
    },
    'programming_languages': [constants.java, constants.python, constants.scala, constants.rust],
    'number_of_translations_per_task': 3,
    'prompt_crafting_iterations': 3,
    'prompt_crafting_current_iteration': 0,  # increment for each prompt crafting iteration
    'manual_analysis_source': False,
    'prompt_selection_config': 'prompt_selection_prompts.json',
    'prompt_selection_assignment': 'prompt_selection_assignment.json',
    'time_logging_file': 'time_log.csv',
    'base_path_data': '',  # to be set
    'base_path_testing_docker': '',  # to be set
    'tco_path': '',  # to be set
    'base_path_data_vm': "",  # to be set
    'num_gpus': 2  # to be set in dependence on LLM and hardware configuration
}
