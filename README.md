# [STG0008079] Improving Decision-Making Abilities of Generative Large Language Models by Combining Multiple Points of View
# Datasets and Scripts for LLM Decision-Making Evaluation

## Authors: Alexander Kvalvaag & Viacheslav Nikolaichuk
##  Course coordinator: Petra Galuscakova

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/vinaysetty/1cc32e6b43d911995bf07adb1cce4e89/raw/coverage.template-project.main.json)

# Testing and GitHub actions

Using `pre-commit` hooks, `flake8`, `black`, `mypy`, `docformatter`, `pydocstyle` and `pytest` are locally run on every commit. For more details on how to use `pre-commit` hooks see [here](https://github.com/iai-group/guidelines/tree/main/python#install-pre-commit-hooks).

Similarly, Github actions are used to run `flake8`, `black`, `mypy`, `docformatter`, `pydocstyle` and `pytest` on every push and pull request and merge to main. The `pytest` also runs coverage. GitHub actions are explained in detail [here](https://github.com/iai-group/guidelines/blob/main/github/Actions.md).

For GitHub actions/CI Setup see [here](docs/CI_setup.md)

# Instructions

This repository contains datasets, Python scripts, and evaluation results for assessing the decision-making capabilities of Large Language Models (LLMs). It supports a study investigating whether multi-perspective prompting can enhance LLM reasoning, evaluated using BERTScore, DeepEval, and human assessments. The repository includes JSON datasets with decision-making tasks, scripts for response collection and evaluation, and organized results for benchmarking and analysis.

## Features

- **Datasets**: JSON files containing decision-making tasks of varying complexity.
- **Scripts**:
  - `GetResponses.py`: Collects LLM responses from APIs or local models.
  - `DeepEval.py`: Evaluates responses using the DeepEval framework.
  - `BERTScore.py`: Evaluates responses using BERTScore metrics.
  - `LowestScoreFinder.py`: Identifies the lowest scores from evaluation results.
  - `collect_mm_stats.py`: Collects stats on which LLMs uses mental models in the responses.
- **Results**: Organized directories with detailed and aggregated evaluation results for BERTScore, DeepEval, and human evaluations, including benchmark tests for metric thresholds.

## Setup

### Prerequisites

*   Python 3.8+
*   Access to LLM APIs (OpenAI, Google Gemini, DeepSeek) and/or local models via Ollama.
*   API Keys for the respective services.
*   (Optional but Recommended) A GPU for faster execution, especially for BERTScore evaluation and running larger local models.

## Install required packages
```console
pip install -r requirements.txt
```

Note: The requirements.txt includes development tools like black, flake8, mypy, etc. These are not strictly necessary to run the core scripts but were used during development.

## Configuration File (config.json)

The scripts/GetResponses.py script requires a config.json file in the root directory to define how to access the different LLMs (both local and API-based).

1. Create the file: Copy the provided config.json.template to config.json.

```console
cp config.json.template config.json
```

2. Edit the file: Open config.json and fill in your details.

## Usage

### 1. Generating LLM Responses
   
Use GetResponses.py to query the selected LLMs (defined in config.json) with the questions from the dataset.

```console
python scripts/GetResponses.py --input datasets/dataset.json --temperature 1.0 --use-description --models gpt-4o # Specify one LLM or more
```

### Evaluating with DeepEval

```console
python scripts/DeepEval.py --responses LLM-Responses-with-description/<filename.json> --metrics answer_relevancy faithfulness
```

### Evaluating with BERTScore

```console
python scripts/BERTScore.py --responses LLM-Responses-with-description/<filename.json>
```
### Finding Lowest Scores

```console
python scripts/LowestScoreFinder.py
```

### Showing LLM mental model usage stats

```console
python scripts/collect_mm_stats.py
```

# Dataset

The primary dataset (data/dataset.json) contains 40 scenarios with 131 questions total. Each scenario includes:

    ID

    Context

    Array of questions

Each question includes:

    Question ID 
    
    Question Text

    Type, and subjective Difficulty

    A single Ground Truth answer (used for automated evaluation)

    Array of answers (model_answers)

Each answer includes:

    Assigned Mental Models (for prompting)

    Illustrative reference answers


The data/test.json file was used specifically for generating the metric threshold benchmark results. It includes scenario 22 and 24, answers were changed to test different outputs.

# Results

Pre-computed results from the experiments described in the thesis are stored in the Results/ directory and then in the various Results-* and LLM-* directories.

    LLM-Responses-with-description/: Raw outputs from the LLMs for the main dataset.

    Results-*[Detailed|Aggregated]/: Evaluation scores (BERTScore, DeepEval, Human) stored in detailed (per-response) and aggregated formats.

    Lowest-Scores/: Lists of responses identified as having the lowest scores by LowestScoreFinder.py.
