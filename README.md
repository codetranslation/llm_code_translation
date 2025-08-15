# Large Language Models for Code Translation: An In-Depth Analysis of Code Smells and Functional Correctness

Artifact repository for the paper titled "Large Language Models for Code Translation: An In-Depth Analysis of Code Smells and Functional Correctness" which is currently under review.
This repository contains the code for reproducing the results presented in our paper.

## Install
To get started, you can clone the repository:

```git clone https://github.com/codetranslation/llm_code_translation```

Make sure to have a suitable Python environment in place. We used Python 3.10. Then, install the requirements:

```pip3 install -r requirements.txt```

## Structure
`config.py` steers the steps to be executed in the pipeline. Currently, we support [CodeNet](https://github.com/IBM/Project_CodeNet), [Avatar](https://github.com/wasiahmad/AVATAR), [EvalPlus](https://github.com/evalplus/evalplus) and [EvoEval](https://github.com/evo-eval/evoeval) as datasets. Further datasets can be added by adapting the `config.py`, `constants.py` and `prepare_dataset.py`.

The LLMs in the `LLMs`-dict in `config.py` refer to HuggingFace model identifiers.

### Translation procedure
To run translations, make sure to have `config["steps"]["run_translation"]` set to the according LLMs. The iterative repair and backtranslation procedure can be activated through the config. Also, make sure to set up all necessary paths in the config. For internal reasons, we conducted the translations and the testing on two different machines; thus, there can be two paths specified.

Furthermore, the source data which is leveraged for translations should be in place. For instance, for Avatar, a .csv file in the place `os.path.join(config["base_path_data"], "avatar", f"translation_data_avatar.csv")` in the following format is expected:
```
,avatar_id,java,python
0,atcoder_ABC042_A,"['517674512', '612206405']",['416447439']
```

Accordingly, the script expects the source file to be in `os.path.join(config["base_path_data"], "avatar", "data", "ABC042", "A", "java", "517674512.java")`. Test cases for Avatar are by default expected, for instance, here: `os.path.join(config["base_path_data"], "avatar", "test_cases", "ABC042", "A", "in", "subtask0_sample_01.txt"` and "out", respectively. 

If the config is set up accordingly, trigger the translation via `main.py`.

For the iterative repair and backtranslations, ensure that intermediate results (testing and code smell detection results) are available so that the best translation attempt can be selected.

### Code smell detection and testing
Adapt the config in dependence on the steps to be executed. We execute both the code smell detection and the test suit in a Docker environment. The docker container is built from `ubuntu:22.04` and Python, Java, Scala, Rust, and PyLint, PMD, Scalastyle, and Clippy need to be installed.

Then, `main.py` can be used to trigger the code smell detection and / or the test suit execution. The image name needs to passed in `main.py` accordingly. 