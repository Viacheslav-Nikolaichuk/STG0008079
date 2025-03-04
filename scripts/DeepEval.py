import json
import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config():
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        for model in config.get("models", {}).values():
            api_type = model.get("api_type")
            if api_type == "openai" and "api_key" in model:
                os.environ["OPENAI_API_KEY"] = model["api_key"]
        return config
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return None


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

def create_test_cases(dataset, responses, reference_type='ground_truth'):
    """Create DeepEval test cases from dataset and responses"""
    test_cases = []
    metadata = []
    
    for scenario in responses.get("scenarios", []):
        scenario_id = scenario.get("id")
        context = scenario.get("context", "")
        
        # Find matching scenario in original dataset
        orig_scenario = next((s for s in dataset.get("scenarios", []) 
                             if s.get("id") == scenario_id), None)
        
        if not orig_scenario:
            logging.warning(f"Scenario {scenario_id} not found in original dataset")
            continue
            
        for question in scenario.get("questions", []):
            question_id = question.get("id")
            question_text = question.get("question", "")
            
            # Find matching question in original dataset
            orig_question = next((q for q in orig_scenario.get("questions", []) 
                                if q.get("id") == question_id), None)
            
            if not orig_question:
                logging.warning(f"Question {question_id} not found in scenario {scenario_id}")
                continue
                
            if reference_type == 'ground_truth':
                base_reference = orig_question.get("ground_truth", "")
            else:
                base_reference = None
            
            for model_answer in question.get("model_answers", []):
                model_name = model_answer.get("model", "base")
                answer = model_answer.get("answer", "")
                
                if not answer or answer == "Error":
                    logging.warning(f"Skipping empty/error answer for {scenario_id}/{question_id}/{model_name}")
                    continue
                
                if reference_type == 'model_answer':
                    orig_model_answer = next((a for a in orig_question.get("model_answers", [])
                                            if a.get("model") == model_name), None)
                    if not orig_model_answer:
                        logging.warning(f"No original model answer found for {model_name} in {scenario_id}/{question_id}")
                        continue
                    reference = orig_model_answer.get("answer", "")
                else:
                    reference = base_reference
                
                if not reference:
                    logging.warning(f"Empty reference for {scenario_id}/{question_id}/{model_name}")
                    continue
                
                test_case = LLMTestCase(
                    input=f"Context: {context}\nQuestion: {question_text}",
                    actual_output=answer,
                    expected_output=reference,
                    retrieval_context=[context] if context else None
                )
                
                test_cases.append(test_case)
                
                meta = {
                    "scenario_id": scenario_id,
                    "question_id": question_id,
                    "model": model_name,
                    "question_type": question.get("question-type", ""),
                    "difficulty": question.get("difficulty", ""),
                    "question": question_text,
                    "reference": reference,
                    "answer": answer,
                }
                metadata.append(meta)
    
    return test_cases, metadata

def evaluate_test_cases(test_cases, eval_model="gpt-4o-mini", threshold=0.7, batch_size=12):
    """Evaluate test cases using DeepEval's AnswerRelevancyMetric"""
    all_results = []
    
    metric = AnswerRelevancyMetric(threshold=threshold, model=eval_model)
    
    # Process in batches
    for i in tqdm(range(0, len(test_cases), batch_size), desc="Evaluating with DeepEval"):
        batch = test_cases[i:i+batch_size]
        batch_results = evaluate(batch, [metric])
        all_results.extend(batch_results)
    
    return all_results

def extract_scores(deepeval_results, metadata):
    """Extract scores from DeepEval results and combine with metadata"""
    combined_results = []
    meta_idx = 0
    
    for result in deepeval_results:
        # Skip results that don't match our expected structure
        if not (isinstance(result, tuple) and len(result) >= 2):
            continue
            
        _, test_results = result
        if not isinstance(test_results, list):
            continue
            
        for test_result in test_results:
            if not hasattr(test_result, 'metrics_data'):
                continue
                
            # Find the Answer Relevancy metric
            relevancy_metrics = [m for m in test_result.metrics_data if m.name == "Answer Relevancy"]
            
            if relevancy_metrics and meta_idx < len(metadata):
                metric = relevancy_metrics[0]
                combined = {
                    "score": metric.score,
                    "passed": metric.success,
                    "reason": metric.reason,
                    **metadata[meta_idx]
                }
                combined_results.append(combined)
                meta_idx += 1
    
    if meta_idx < len(metadata):
        logging.warning(f"Only processed {meta_idx} out of {len(metadata)} metadata entries")
    
    return combined_results

def compute_aggregated_metrics(results):
    """Compute aggregated metrics by different criteria"""
    # Overall average
    overall = {
        "score": sum(result["score"] for result in results) / len(results),
        "pass_rate": sum(1 for result in results if result["passed"]) / len(results)
    }
    
    # Group by model
    by_model = {}
    for result in results:
        model = result["model"]
        if model not in by_model:
            by_model[model] = {"results": [], "score": 0, "pass_rate": 0}
        by_model[model]["results"].append(result)
    
    # Calculate averages for each model
    for model, data in by_model.items():
        data["score"] = sum(r["score"] for r in data["results"]) / len(data["results"])
        data["pass_rate"] = sum(1 for r in data["results"] if r["passed"]) / len(data["results"])
    
    # Group by question type
    by_question_type = {}
    for result in results:
        q_type = result["question_type"]
        if q_type not in by_question_type:
            by_question_type[q_type] = {"results": [], "score": 0, "pass_rate": 0}
        by_question_type[q_type]["results"].append(result)
    
    # Calculate averages for each question type
    for q_type, data in by_question_type.items():
        data["score"] = sum(r["score"] for r in data["results"]) / len(data["results"])
        data["pass_rate"] = sum(1 for r in data["results"] if r["passed"]) / len(data["results"])
        
    # Group by difficulty
    by_difficulty = {}
    for result in results:
        difficulty = result["difficulty"]
        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = {"results": [], "score": 0, "pass_rate": 0}
        by_difficulty[difficulty]["results"].append(result)
    
    # Calculate averages for each difficulty
    for difficulty, data in by_difficulty.items():
        data["score"] = sum(r["score"] for r in data["results"]) / len(data["results"])
        data["pass_rate"] = sum(1 for r in data["results"] if r["passed"]) / len(data["results"])
    
    return {
        "overall": overall,
        "by_model": {k: {key: val for key, val in v.items() if key != "results"} for k, v in by_model.items()},
        "by_question_type": {k: {key: val for key, val in v.items() if key != "results"} for k, v in by_question_type.items()},
        "by_difficulty": {k: {key: val for key, val in v.items() if key != "results"} for k, v in by_difficulty.items()}
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM responses using DeepEval')
    parser.add_argument('--dataset', default='data/TempQuestions.json', help='Path to original dataset')
    parser.add_argument('--responses', required=True, help='Path to LLM responses JSON file')
    parser.add_argument('--output-dir', default='Results', help='Output directory for results')
    parser.add_argument('--eval-model', default='gpt-4o-mini', help='Model to use for evaluation')
    parser.add_argument('--threshold', type=float, default=0.7, help='Threshold for passing the metric')
    parser.add_argument('--batch-size', type=int, default=12, help='Batch size for evaluation')
    parser.add_argument('--reference-type', choices=['ground_truth', 'model_answer'], default='ground_truth',
                        help='Reference type to compare against (ground_truth or model_answer)')
    args = parser.parse_args()
    
    config = load_config()
    if not config:
        logging.error("Failed to load config. Exiting.")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    dataset = load_dataset(args.dataset)
    responses = load_responses(args.responses)
    
    if not dataset or not responses:
        logging.error("Failed to load required data. Exiting.")
        return
    
    # Extract model name from responses filename
    model_name = Path(args.responses).stem.replace('_responses', '')
    logging.info(f"Evaluating responses from {model_name} using {args.eval_model}")
    logging.info(f"Using reference type: {args.reference_type}")
    
    test_cases, metadata = create_test_cases(dataset, responses, args.reference_type)
    logging.info(f"Created {len(test_cases)} test cases")
    
    if not test_cases:
        logging.error("No valid test cases created. Exiting.")
        return
    
    # Evaluate test cases
    deepeval_results = evaluate_test_cases(
        test_cases, 
        eval_model=args.eval_model, 
        threshold=args.threshold,
        batch_size=args.batch_size
    )
    
    # Extract and combine scores with metadata
    combined_results = extract_scores(deepeval_results, metadata)
    
    # Compute aggregated metrics
    aggregated_metrics = compute_aggregated_metrics(combined_results)
    
    detailed_results = {
        "reference_type": args.reference_type,
        "results": combined_results
    }
    
    aggregated_results = {
        "reference_type": args.reference_type,
        "metrics": aggregated_metrics
    }
    
    # Tag output file names with the reference type
    detailed_output_path = output_dir / f"deepeval_detailed_{model_name}_{args.reference_type}.json"
    with open(detailed_output_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    aggregated_output_path = output_dir / f"deepeval_aggregated_{model_name}_{args.reference_type}.json"
    with open(aggregated_output_path, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    logging.info(f"Successfully saved detailed results to {detailed_output_path}")
    logging.info(f"Successfully saved aggregated results to {aggregated_output_path}")
    
    overall_score = aggregated_metrics["overall"]["score"]
    overall_pass_rate = aggregated_metrics["overall"]["pass_rate"]
    logging.info(f"Overall DeepEval Score: {overall_score:.4f}")
    logging.info(f"Overall Pass Rate: {overall_pass_rate:.2%}")
    
    logging.info("DeepEval Scores by model:")
    for model, metrics in aggregated_metrics["by_model"].items():
        model_name_display = model if model else "base"
        logging.info(f"  {model_name_display}: {metrics['score']:.4f} (Pass Rate: {metrics['pass_rate']:.2%})")

if __name__ == "__main__":
    main()