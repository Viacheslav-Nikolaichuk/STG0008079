import json
import argparse
import logging
from pathlib import Path
from evaluate import load
import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_dataset(dataset_path):
    """Load the original dataset with ground truth answers"""
    try:
        with open(dataset_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        return None

def load_responses(responses_path):
    """Load the LLM responses file"""
    try:
        with open(responses_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading responses: {str(e)}")
        return None

def extract_comparisons(dataset, responses):
    """Extract prediction-reference pairs for BERTScore computation"""
    predictions = []
    references = []
    metadata = []
    
    # Maps scenario and question IDs to their indices in the arrays
    index_mapping = {}
    counter = 0
    
    for scenario_idx, scenario in enumerate(responses.get("scenarios", [])):
        scenario_id = scenario.get("id")
        
        # Find matching scenario in original dataset
        orig_scenario = next((s for s in dataset.get("scenarios", []) 
                             if s.get("id") == scenario_id), None)
        
        if not orig_scenario:
            logging.warning(f"Scenario {scenario_id} not found in original dataset")
            continue
            
        for question_idx, question in enumerate(scenario.get("questions", [])):
            question_id = question.get("id")
            
            # Find matching question in original dataset
            orig_question = next((q for q in orig_scenario.get("questions", []) 
                                if q.get("id") == question_id), None)
            
            if not orig_question:
                logging.warning(f"Question {question_id} not found in scenario {scenario_id}")
                continue
                
            ground_truth = orig_question.get("ground_truth", "")
            
            for answer_idx, model_answer in enumerate(question.get("model_answers", [])):
                model_name = model_answer.get("model", "base")
                answer = model_answer.get("answer", "")
                
                if not answer or answer == "Error":
                    logging.warning(f"Skipping empty/error answer for {scenario_id}/{question_id}/{model_name}")
                    continue
                
                predictions.append(answer)
                references.append(ground_truth)
                
                meta = {
                    "scenario_id": scenario_id,
                    "question_id": question_id,
                    "model": model_name,
                    "question_type": question.get("question-type", ""),
                    "difficulty": question.get("difficulty", "")
                }
                metadata.append(meta)
                
                # Store the index mapping
                key = (scenario_id, question_id, model_name)
                index_mapping[key] = counter
                counter += 1
    
    return predictions, references, metadata, index_mapping

def compute_bertscore(predictions, references, batch_size=8):
    """Compute BERTScore for prediction-reference pairs"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    bertscore = load("bertscore")
    
    all_results = {"precision": [], "recall": [], "f1": []}
    
    # Process in batches to avoid OOM issues
    for i in tqdm(range(0, len(predictions), batch_size), desc="Computing BERTScore"):
        batch_preds = predictions[i:i+batch_size]
        batch_refs = references[i:i+batch_size]
        
        results = bertscore.compute(
            predictions=batch_preds,
            references=batch_refs,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",  # Using a stronger model
            device=device,
            use_fast_tokenizer=False
        )
        
        all_results["precision"].extend([float(p) for p in results["precision"]])
        all_results["recall"].extend([float(r) for r in results["recall"]])
        all_results["f1"].extend([float(f) for f in results["f1"]])
    
    return all_results

def combine_results(predictions, references, bertscore_results, metadata):
    """Combine all results into a structured format"""
    combined = []
    
    for i in range(len(predictions)):
        entry = {
            "prediction": predictions[i],
            "reference": references[i],
            "precision": bertscore_results["precision"][i],
            "recall": bertscore_results["recall"][i],
            "f1": bertscore_results["f1"][i],
            **metadata[i]  # Include all metadata
        }
        combined.append(entry)
    
    return combined

def compute_aggregated_metrics(results, index_mapping):
    """Compute aggregated metrics by different criteria"""
    # Overall average
    overall = {
        "precision": sum(result["precision"] for result in results) / len(results),
        "recall": sum(result["recall"] for result in results) / len(results),
        "f1": sum(result["f1"] for result in results) / len(results),
    }
    
    # Group by model
    by_model = {}
    for result in results:
        model = result["model"]
        if model not in by_model:
            by_model[model] = {"results": [], "precision": 0, "recall": 0, "f1": 0}
        by_model[model]["results"].append(result)
    
    # Calculate averages for each model
    for model, data in by_model.items():
        data["precision"] = sum(r["precision"] for r in data["results"]) / len(data["results"])
        data["recall"] = sum(r["recall"] for r in data["results"]) / len(data["results"])
        data["f1"] = sum(r["f1"] for r in data["results"]) / len(data["results"])
    
    # Group by question type
    by_question_type = {}
    for result in results:
        q_type = result["question_type"]
        if q_type not in by_question_type:
            by_question_type[q_type] = {"results": [], "precision": 0, "recall": 0, "f1": 0}
        by_question_type[q_type]["results"].append(result)
    
    # Calculate averages for each question type
    for q_type, data in by_question_type.items():
        data["precision"] = sum(r["precision"] for r in data["results"]) / len(data["results"])
        data["recall"] = sum(r["recall"] for r in data["results"]) / len(data["results"])
        data["f1"] = sum(r["f1"] for r in data["results"]) / len(data["results"])
        
    # Group by difficulty
    by_difficulty = {}
    for result in results:
        difficulty = result["difficulty"]
        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = {"results": [], "precision": 0, "recall": 0, "f1": 0}
        by_difficulty[difficulty]["results"].append(result)
    
    # Calculate averages for each difficulty
    for difficulty, data in by_difficulty.items():
        data["precision"] = sum(r["precision"] for r in data["results"]) / len(data["results"])
        data["recall"] = sum(r["recall"] for r in data["results"]) / len(data["results"])
        data["f1"] = sum(r["f1"] for r in data["results"]) / len(data["results"])
    
    return {
        "overall": overall,
        "by_model": {k: {key: val for key, val in v.items() if key != "results"} for k, v in by_model.items()},
        "by_question_type": {k: {key: val for key, val in v.items() if key != "results"} for k, v in by_question_type.items()},
        "by_difficulty": {k: {key: val for key, val in v.items() if key != "results"} for k, v in by_difficulty.items()}
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM responses using BERTScore')
    parser.add_argument('--dataset', default='data/TempQuestions.json', help='Path to original dataset with ground truth')
    parser.add_argument('--responses', required=True, help='Path to LLM responses JSON file')
    parser.add_argument('--output-dir', default='Results', help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for BERTScore computation')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    dataset = load_dataset(args.dataset)
    responses = load_responses(args.responses)
    
    if not dataset or not responses:
        logging.error("Failed to load required data. Exiting.")
        return
    
    # Extract model name from responses filename
    model_name = Path(args.responses).stem.replace('_responses', '')
    logging.info(f"Evaluating responses from {model_name}")
    
    # Extract predictions and references
    predictions, references, metadata, index_mapping = extract_comparisons(dataset, responses)
    logging.info(f"Extracted {len(predictions)} prediction-reference pairs")
    
    if not predictions:
        logging.error("No valid prediction-reference pairs found. Exiting.")
        return
    
    bertscore_results = compute_bertscore(predictions, references, args.batch_size)
    
    combined_results = combine_results(predictions, references, bertscore_results, metadata)
    
    aggregated_metrics = compute_aggregated_metrics(combined_results, index_mapping)
    
    # Save detailed results
    detailed_output_path = output_dir / f"bertscore_detailed_{model_name}.json"
    with open(detailed_output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    # Save aggregated results
    aggregated_output_path = output_dir / f"bertscore_aggregated_{model_name}.json"
    with open(aggregated_output_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    
    logging.info(f"Successfully saved detailed results to {detailed_output_path}")
    logging.info(f"Successfully saved aggregated results to {aggregated_output_path}")
    
    # Print overall performance
    overall_f1 = aggregated_metrics["overall"]["f1"]
    logging.info(f"Overall BERTScore F1: {overall_f1:.4f}")
    
    # Print model performance
    logging.info("BERTScore F1 by model:")
    for model, metrics in aggregated_metrics["by_model"].items():
        model_name_display = model if model else "base"
        logging.info(f"  {model_name_display}: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()