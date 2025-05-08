import json
import os
import heapq
import sys
import re

DIRECTORIES_TO_SCAN = [
    "./Results-DeepEval-Detailed",
    "./Results-Bertscore-Detailed",
]

NUM_LOWEST = 5
OUTPUT_FILE_PREFIX = "lowest_scores_"


def generate_output_filename(dir_path, prefix):
    """Generates a clean filename based on the directory path."""
    dir_name = os.path.basename(os.path.abspath(dir_path))
    clean_dir_name = re.sub(r"[^\w\-\_\. ]", "_", dir_name)
    return f"{prefix}{clean_dir_name}.json"


def process_directory(directory_path, num_lowest=NUM_LOWEST):
    """
    Finds the lowest scores for files within a single directory.

    Args:
        directory_path (str): The path to the directory containing JSON files.
        num_lowest (int): The number of lowest scores to find per file.

    Returns:
        dict: A dictionary of results for this directory, or None if errors occur.
    """
    absolute_directory_path = os.path.abspath(directory_path)
    if not os.path.isdir(absolute_directory_path):
        print(
            f"Error: Directory not found at '{absolute_directory_path}' (Resolved from '{directory_path}')."
        )
        return None

    print(f"\n--- Scanning directory: {absolute_directory_path} ---")
    results_for_this_dir = {}
    try:
        filenames = os.listdir(absolute_directory_path)
    except OSError as e:
        print(f"Error accessing directory '{absolute_directory_path}': {e}.")
        return None

    found_json_files_in_dir = False
    for filename in filenames:
        if filename.lower().endswith(".json"):
            found_json_files_in_dir = True
            file_path = os.path.join(absolute_directory_path, filename)
            print(f"  Processing file: {filename}...")
            score_data_list = []

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "results" not in data or not isinstance(
                    data["results"], list
                ):
                    print(
                        f"    Skipping {filename}: Missing or invalid 'results' list."
                    )
                    continue

                is_deepeval_format = (
                    "metrics_used" in data
                    and isinstance(data.get("metrics_used"), list)
                    and data["metrics_used"]
                )

                if is_deepeval_format:
                    metrics_to_check = data["metrics_used"]
                    print(
                        f"    Detected DeepEval format (metrics: {metrics_to_check})."
                    )
                    for result_item in data["results"]:
                        query_id = result_item.get("query_id")
                        question = result_item.get("question")
                        answer = result_item.get("answer")
                        reference = result_item.get("reference")

                        if (
                            question is None
                            or answer is None
                            or reference is None
                        ):
                            print(
                                f"    Warning: Item missing 'question', 'answer', or 'reference'. Skipping."
                            )
                            continue
                        if query_id is None:
                            print(
                                f"    Warning: DeepEval item missing 'query_id'. Question: '{question[:30]}...'"
                            )

                        for metric_name in metrics_to_check:
                            metric_data = result_item.get(metric_name)
                            if isinstance(metric_data, dict):
                                score = metric_data.get("score")
                                if isinstance(score, (int, float)):
                                    score_data_list.append(
                                        {
                                            "score": score,
                                            "question": question,
                                            "answer": answer,
                                            "reference": reference,
                                            "query_id": query_id,
                                        }
                                    )
                else:
                    print(
                        f"    Assuming BertScore format (looking for 'f1' score)."
                    )
                    bertscore_found = False
                    for result_item in data["results"]:
                        question = (
                            result_item.get("question")
                            or f"QueryID: {result_item.get('query_id', 'N/A')}"
                        )
                        answer = result_item.get("answer")
                        reference = result_item.get("reference")
                        f1_score = result_item.get("f1")

                        if answer is None or reference is None:
                            print(
                                f"    Warning: Item missing 'answer' or 'reference'. Skipping."
                            )
                            continue

                        if isinstance(f1_score, (int, float)):
                            bertscore_found = True
                            score_data_list.append(
                                {
                                    "score": f1_score,
                                    "question": question,
                                    "answer": answer,
                                    "reference": reference,
                                }
                            )
                    if not bertscore_found:
                        print(
                            f"    Warning: No 'f1' scores found in assumed BertScore file {filename}."
                        )

                if not score_data_list:
                    print(
                        f"    No valid score data entries found or extracted in {filename}."
                    )
                    continue

                lowest_items = heapq.nsmallest(
                    num_lowest, score_data_list, key=lambda x: x["score"]
                )

                output_list = []
                for item in lowest_items:
                    output_entry = {
                        "score": item["score"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "reference": item["reference"],
                    }
                    if "query_id" in item and item["query_id"] is not None:
                        output_entry["query_id"] = item["query_id"]
                    output_list.append(output_entry)

                results_for_this_dir[filename] = output_list
                print(
                    f"    Found {len(score_data_list)} score data entries. Identified {len(lowest_items)} lowest."
                )

            except json.JSONDecodeError:
                print(
                    f"    Error: Could not decode JSON from {filename}. Skipping."
                )
            except FileNotFoundError:
                print(
                    f"    Error: File not found {filename} during processing. Skipping."
                )
            except Exception as e:
                print(
                    f"    An unexpected error occurred processing {filename}: {e}. Skipping."
                )

    if not found_json_files_in_dir:
        print(f"  No JSON files found in this directory.")
        return {}

    return results_for_this_dir


if __name__ == "__main__":
    overall_success = True
    any_results_generated = False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(
        os.path.join(script_dir, "..", "Lowest-Scores")
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output files will be saved in: {output_dir}")

    for target_dir in DIRECTORIES_TO_SCAN:
        dir_results = process_directory(target_dir, NUM_LOWEST)

        if dir_results is None:
            overall_success = False
            continue

        output_filename = generate_output_filename(
            target_dir, OUTPUT_FILE_PREFIX
        )
        output_path = os.path.join(output_dir, output_filename)

        if dir_results:
            any_results_generated = True

        try:
            output_json = json.dumps(dir_results, indent=4)
            with open(output_path, "w", encoding="utf-8") as outfile:
                outfile.write(output_json)
            if dir_results:
                print(
                    f"Results for '{target_dir}' successfully saved to {output_path}"
                )
            else:
                print(
                    f"No results generated for directory '{target_dir}', empty report file saved to {output_path}"
                )

        except IOError as e:
            print(f"Error writing output file '{output_path}': {e}")
            overall_success = False
        except TypeError as e:
            print(f"Error serializing results for '{target_dir}' to JSON: {e}")
            overall_success = False

    print("\n--- Script Finished ---")
    if not overall_success:
        print("Completed with some errors.")
        sys.exit(1)
    elif not any_results_generated:
        print(
            "Completed, but no results were generated across any directories (check directories and file contents)."
        )
    else:
        print("Completed successfully.")
