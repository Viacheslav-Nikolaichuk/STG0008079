import json
from pathlib import Path

BERTCORE_DIR = Path("Results/Results-Bertscore-Detailed")
DEEPEVAL_DIR = Path("Results/Results-DeepEval-Detailed")
OUTPUT_DIR = Path("Results/Lowest-Scores")
N_LOWEST_PER_FILE = 15
LOWEST_N_OVERALL_TO_SAVE = 50

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

per_file_lowest_f1 = {}
per_file_lowest_relevancy = {}
per_file_lowest_faithfulness = {}

all_f1_entries = []
all_relevancy_entries = []
all_faithfulness_entries = []


def extract_common_data(item, filename):
    """Helper function to extract common fields from a result item."""
    return {
        "model": item.get("model", ""),
        "query_id": item.get("query_id"),
        "scenario_id": item.get("scenario_id"),
        "answer": item.get("answer"),
        "reference": item.get("reference"),
        "source_file": filename,
    }


print(f"Processing Bertscore files from: {BERTCORE_DIR}")
for filepath in BERTCORE_DIR.glob("*.json"):
    print(f"  Reading: {filepath.name}")
    current_file_f1_items = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("results", []):
                f1 = item.get("f1")
                if f1 is not None:
                    entry_data = extract_common_data(item, filepath.name)
                    entry_data["f1_score"] = f1
                    current_file_f1_items.append(entry_data)
                    all_f1_entries.append(entry_data)

        current_file_f1_items.sort(key=lambda x: x["f1_score"])
        per_file_lowest_f1[filepath.name] = current_file_f1_items[
            :N_LOWEST_PER_FILE
        ]

    except json.JSONDecodeError:
        print(f"    Error decoding JSON from {filepath.name}")
    except Exception as e:
        print(f"    An error occurred while processing {filepath.name}: {e}")

print(f"\nProcessing DeepEval files from: {DEEPEVAL_DIR}")
for filepath in DEEPEVAL_DIR.glob("*.json"):
    print(f"  Reading: {filepath.name}")
    current_file_relevancy_items = []
    current_file_faithfulness_items = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("results", []):
                common_data = extract_common_data(item, filepath.name)

                answer_relevancy_score = item.get("answer_relevancy", {}).get(
                    "score"
                )
                if answer_relevancy_score is not None:
                    relevancy_entry = common_data.copy()
                    relevancy_entry["answer_relevancy_score"] = (
                        answer_relevancy_score
                    )
                    current_file_relevancy_items.append(relevancy_entry)
                    all_relevancy_entries.append(relevancy_entry)

                faithfulness_score = item.get("faithfulness", {}).get("score")
                if faithfulness_score is not None:
                    faithfulness_entry = common_data.copy()
                    faithfulness_entry["faithfulness_score"] = (
                        faithfulness_score
                    )
                    current_file_faithfulness_items.append(faithfulness_entry)
                    all_faithfulness_entries.append(faithfulness_entry)

        # Get N lowest for the current file for relevancy
        current_file_relevancy_items.sort(
            key=lambda x: x["answer_relevancy_score"]
        )
        per_file_lowest_relevancy[filepath.name] = current_file_relevancy_items[
            :N_LOWEST_PER_FILE
        ]

        # Get N lowest for the current file for faithfulness
        current_file_faithfulness_items.sort(
            key=lambda x: x["faithfulness_score"]
        )
        per_file_lowest_faithfulness[filepath.name] = (
            current_file_faithfulness_items[:N_LOWEST_PER_FILE]
        )

    except json.JSONDecodeError:
        print(f"    Error decoding JSON from {filepath.name}")
    except Exception as e:
        print(f"    An error occurred while processing {filepath.name}: {e}")

output_per_file_f1_path = (
    OUTPUT_DIR / f"per_file_lowest_scores_bertscore_f1.json"
)
print(
    f"\nWriting per-file lowest {N_LOWEST_PER_FILE} F1 scores to: {output_per_file_f1_path}"
)
with open(output_per_file_f1_path, "w", encoding="utf-8") as f:
    json.dump(per_file_lowest_f1, f, indent=2)

output_per_file_relevancy_path = (
    OUTPUT_DIR / f"per_file_lowest_scores_answer_relevancy.json"
)
print(
    f"Writing per-file lowest {N_LOWEST_PER_FILE} Answer Relevancy scores to: {output_per_file_relevancy_path}"
)
with open(output_per_file_relevancy_path, "w", encoding="utf-8") as f:
    json.dump(per_file_lowest_relevancy, f, indent=2)

output_per_file_faithfulness_path = (
    OUTPUT_DIR / f"per_file_lowest_scores_faithfulness.json"
)
print(
    f"Writing per-file lowest {N_LOWEST_PER_FILE} Faithfulness scores to: {output_per_file_faithfulness_path}"
)
with open(output_per_file_faithfulness_path, "w", encoding="utf-8") as f:
    json.dump(per_file_lowest_faithfulness, f, indent=2)


print(
    f"\n--- Saving Overall lowest {LOWEST_N_OVERALL_TO_SAVE} Lowest Scores to Files ---"
)

if all_f1_entries:
    all_f1_entries.sort(key=lambda x: x["f1_score"])
    lowest_overall_f1 = all_f1_entries[:LOWEST_N_OVERALL_TO_SAVE]
    output_overall_f1_path = (
        OUTPUT_DIR / f"overall_lowest_scores_bertscore_f1.json"
    )
    print(
        f"Writing overall lowest {LOWEST_N_OVERALL_TO_SAVE} lowest F1 scores to: {output_overall_f1_path}"
    )
    with open(output_overall_f1_path, "w", encoding="utf-8") as f:
        json.dump(lowest_overall_f1, f, indent=2)
else:
    print("No F1 scores found to create overall lowest F1 file.")

if all_relevancy_entries:
    all_relevancy_entries.sort(key=lambda x: x["answer_relevancy_score"])
    lowest_overall_relevancy = all_relevancy_entries[:LOWEST_N_OVERALL_TO_SAVE]
    output_overall_relevancy_path = (
        OUTPUT_DIR / f"overall_lowest_scores_answer_relevancy.json"
    )
    print(
        f"Writing overall lowest {LOWEST_N_OVERALL_TO_SAVE} lowest Answer Relevancy scores to: {output_overall_relevancy_path}"
    )
    with open(output_overall_relevancy_path, "w", encoding="utf-8") as f:
        json.dump(lowest_overall_relevancy, f, indent=2)
else:
    print(
        "No Answer Relevancy scores found to create overall lowest Answer Relevancy file."
    )

if all_faithfulness_entries:
    all_faithfulness_entries.sort(key=lambda x: x["faithfulness_score"])
    lowest_overall_faithfulness = all_faithfulness_entries[
        :LOWEST_N_OVERALL_TO_SAVE
    ]
    output_overall_faithfulness_path = (
        OUTPUT_DIR / f"overall_lowest_scores_faithfulness.json"
    )
    print(
        f"Writing overall lowest {LOWEST_N_OVERALL_TO_SAVE} lowest Faithfulness scores to: {output_overall_faithfulness_path}"
    )
    with open(output_overall_faithfulness_path, "w", encoding="utf-8") as f:
        json.dump(lowest_overall_faithfulness, f, indent=2)
else:
    print(
        "No Faithfulness scores found to create overall lowest Faithfulness file."
    )

print("\nScript finished.")
