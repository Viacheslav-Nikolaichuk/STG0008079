import json
from pathlib import Path
from collections import defaultdict

BASE_RESULTS_DIR = Path("Results")
BERTCORE_INPUT_DIR = BASE_RESULTS_DIR / "Results-Bertscore-Detailed"
DEEPEVAL_INPUT_DIR = BASE_RESULTS_DIR / "Results-DeepEval-Detailed"
OUTPUT_DIR = BASE_RESULTS_DIR / "Average-Scores"

BERTCORE_SCORE_KEYS = ["precision", "recall", "f1"]
DEEPEVAL_METRIC_KEYS = {
    "answer_relevancy": "answer_relevancy",
    "faithfulness": "faithfulness",
}

def get_score_from_result(result, file_type, score_key_or_metric_name=None):
    if file_type == "bertscore":
        return result.get(score_key_or_metric_name)
    elif file_type == "deepeval":
        if score_key_or_metric_name and score_key_or_metric_name in result and isinstance(result[score_key_or_metric_name], dict):
            return result[score_key_or_metric_name].get("score")
    return None

def calculate_average_scores_for_file(file_path, file_type, score_key_or_deepeval_metric_name=None):
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None

    if "results" not in data or not isinstance(data["results"], list):
        return None

    scores_by_category = {
        "no_mental_model": [], "one_mental_model": [],
        "two_mental_models": [], "think_hard_prompt": [],
    }

    for result in data["results"]:
        model_value = result.get("model", "")
        score = get_score_from_result(result, file_type, score_key_or_deepeval_metric_name)

        if score is None or not isinstance(score, (int, float)):
            continue

        if model_value == "": scores_by_category["no_mental_model"].append(score)
        elif model_value == "Think hard prompt": scores_by_category["think_hard_prompt"].append(score)
        elif "+" in model_value: scores_by_category["two_mental_models"].append(score)
        else: scores_by_category["one_mental_model"].append(score)

    average_scores_for_metric = {}
    for category, scores in scores_by_category.items():
        average_scores_for_metric[category] = sum(scores) / len(scores) if scores else 0.0

    return average_scores_for_metric if any(average_scores_for_metric.values()) else None

def process_files_in_directory(input_dir, file_type, all_averages_by_metric):
    if not input_dir.is_dir(): return

    for file_path in input_dir.glob("*.json"):
        if "lastquestions" in file_path.name.lower(): continue

        stem_parts = file_path.stem.split('_')
        model_name_from_file = "unknown_model"

        if file_type == "bertscore":
            if len(stem_parts) > 2 and stem_parts[0] == "bertscore" and stem_parts[1] == "detailed":
                model_name_parts = []
                for i in range(2, len(stem_parts)):
                    if stem_parts[i].startswith("temp"): break
                    model_name_parts.append(stem_parts[i])
                if model_name_parts: model_name_from_file = "_".join(model_name_parts)
        elif file_type == "deepeval":
            if len(stem_parts) > 2 and stem_parts[0] == "deepeval" and stem_parts[1] == "detailed":
                model_name_parts = []
                for i in range(2, len(stem_parts)):
                    if stem_parts[i].startswith("temp"): break
                    model_name_parts.append(stem_parts[i])
                if model_name_parts: model_name_from_file = "_".join(model_name_parts)

        if model_name_from_file == "unknown_model" and len(stem_parts) > 2:
            if (stem_parts[0] in ["bertscore", "deepeval"] and stem_parts[1] == "detailed"):
                 model_name_from_file = stem_parts[2]

        if model_name_from_file == "unknown_model": continue

        if file_type == "bertscore":
            for score_key in BERTCORE_SCORE_KEYS:
                averages = calculate_average_scores_for_file(file_path, file_type, score_key)
                if averages:
                    output_metric_name = f"bertscore_{score_key}"
                    all_averages_by_metric[output_metric_name][model_name_from_file] = averages
        elif file_type == "deepeval":
            for deepeval_output_key, deepeval_input_key in DEEPEVAL_METRIC_KEYS.items():
                averages = calculate_average_scores_for_file(file_path, file_type, deepeval_input_key)
                if averages:
                    output_metric_name = f"deepeval_{deepeval_output_key}"
                    all_averages_by_metric[output_metric_name][model_name_from_file] = averages

def save_averages_to_json(data, output_filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_filename
    try:
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass 

def main():
    all_averages_by_metric = defaultdict(lambda: defaultdict(dict))
    process_files_in_directory(BERTCORE_INPUT_DIR, "bertscore", all_averages_by_metric)
    process_files_in_directory(DEEPEVAL_INPUT_DIR, "deepeval", all_averages_by_metric)

    for metric_type_key, data_for_metric in all_averages_by_metric.items():
        if data_for_metric:
            save_averages_to_json(data_for_metric, f"average_{metric_type_key}.json")

    if not any(all_averages_by_metric.values()):
        print("No data processed or found.")
    else:
        print(f"Processing complete. Output files are in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()