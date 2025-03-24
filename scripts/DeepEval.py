import json
import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric, 
    HallucinationMetric,
    BiasMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    DAGMetric,
    TaskCompletionMetric
)
        
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

AVAILABLE_METRICS = {
    'answer_relevancy': AnswerRelevancyMetric,
    'faithfulness': FaithfulnessMetric,
    'hallucination': HallucinationMetric,
    'bias': BiasMetric,
    'contextual_precision': ContextualPrecisionMetric,
    'contextual_recall': ContextualRecallMetric,
    'contextual_relevancy': ContextualRelevancyMetric,
    'dag': DAGMetric,
    'task_completion': TaskCompletionMetric
}

def load_config():
    """Load configuration from config.json"""
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
    skipped_count = 0
    
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
            question_id = question.get("query_id")
            question_text = question.get("question", "")
            
            # Find matching question in original dataset
            orig_question = next((q for q in orig_scenario.get("questions", []) 
                                if q.get("query_id") == question_id), None)
            
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
                        skipped_count += 1
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
                    "query_id": question_id,
                    "model": model_name,
                    "question_type": question.get("question-type", ""),
                    "difficulty": question.get("difficulty", ""),
                    "question": question_text,
                    "reference": reference,
                    "answer": answer,
                }
                metadata.append(meta)
    
    return test_cases, metadata

def create_metrics(metric_names, threshold=0.7, eval_model="gpt-4o-mini"):
    """
    Create metric instances based on provided metric names
    
    Args:
        metric_names (list): List of metric names to use
        threshold (float): Threshold for passing metrics
        eval_model (str): Evaluation model to use
    
    Returns:
        list: List of instantiated metric objects
    """
    metrics = []
    for name in metric_names:
        metric_class = AVAILABLE_METRICS.get(name.lower())
        if not metric_class:
            logging.warning(f"Metric {name} not found. Skipping.")
            continue
        
        # Dynamic metric creation with threshold
        if name.lower() in ['answer_relevancy', 'faithfulness', 'contextual_precision', 'contextual_recall', 'contextual_relevancy']:
            metric = metric_class(threshold=threshold, model=eval_model)
        else:
            # For metrics that might not support threshold
            metric = metric_class(model=eval_model)
        
        metrics.append(metric)
    
    return metrics

def evaluate_test_cases(test_cases, metrics, batch_size=15):
    """
    Evaluate test cases using specified metrics
    
    Args:
        test_cases (list): List of LLMTestCase objects
        metrics (list): List of metric objects
        batch_size (int): Number of test cases to process in each batch
    
    Returns:
        list: Evaluation results
    """
    all_results = []
    
    # Process in batches
    for i in tqdm(range(0, len(test_cases), batch_size), desc="Evaluating with DeepEval"):
        batch = test_cases[i:i+batch_size]
        batch_results = evaluate(batch, metrics)
        all_results.extend(batch_results)
    
    return all_results

def extract_scores(deepeval_results, metadata, metric_names):
    """
    Extract scores from DeepEval results and combine with metadata
    
    Args:
        deepeval_results (list): Results from DeepEval evaluation
        metadata (list): Metadata for each test case
        metric_names (list): Names of metrics used
    
    Returns:
        list: Combined results with scores and metadata
    """
    combined_results = []
    meta_idx = 0
    
    for result in deepeval_results:
        if not (isinstance(result, tuple) and len(result) >= 2):
            continue
            
        _, test_results = result
        if not isinstance(test_results, list):
            continue
            
        for test_result in test_results:
            if not hasattr(test_result, 'metrics_data'):
                continue
                
            # Collect metrics for this test case
            metrics_data = {}
            for metric_name in metric_names:
                matching_metrics = [
                    m for m in test_result.metrics_data 
                    if m.name.lower().replace(' ', '_') == metric_name.lower()
                ]
                
                if matching_metrics:
                    metric = matching_metrics[0]
                    metrics_data[metric_name] = {
                        "score": metric.score,
                        "passed": metric.success,
                        "reason": metric.reason
                    }
            
            if metrics_data and meta_idx < len(metadata):
                combined = {
                    **metrics_data,
                    **metadata[meta_idx]
                }
                combined_results.append(combined)
                meta_idx += 1
    
    if meta_idx < len(metadata):
        logging.warning(f"Only processed {meta_idx} out of {len(metadata)} metadata entries")
    
    return combined_results

def compute_aggregated_metrics(results, metric_names):
    """
    Compute aggregated metrics by different criteria
    
    Args:
        results (list): Combined results with scores and metadata
        metric_names (list): Names of metrics used
    
    Returns:
        dict: Aggregated metrics
    """
    aggregated_metrics = {}
    
    # Compute metrics for each metric name
    for metric_name in metric_names:
        metric_results = [r[metric_name] for r in results if metric_name in r]
        
        # Overall metric
        aggregated_metrics[metric_name] = {
            "overall": {
                "score": sum(r["score"] for r in metric_results) / len(metric_results),
                "pass_rate": sum(1 for r in metric_results if r["passed"]) / len(metric_results)
            },
            "by_model": _compute_group_metrics(results, metric_name, "model"),
            "by_question_type": _compute_group_metrics(results, metric_name, "question_type"),
            "by_difficulty": _compute_group_metrics(results, metric_name, "difficulty")
        }
    
    return aggregated_metrics

def _compute_group_metrics(results, metric_name, group_key):
    """
    Helper function to compute metrics for a specific grouping
    """
    grouped_results = {}
    for result in results:
        if metric_name not in result:
            continue
        
        group_value = result.get(group_key, "unknown")
        if group_value not in grouped_results:
            grouped_results[group_value] = []
        
        grouped_results[group_value].append(result[metric_name])
    
    # Compute metrics for each group
    group_metrics = {}
    for group, group_metric_results in grouped_results.items():
        group_metrics[group] = {
            "score": sum(r["score"] for r in group_metric_results) / len(group_metric_results),
            "pass_rate": sum(1 for r in group_metric_results if r["passed"]) / len(group_metric_results)
        }
    
    return group_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM responses using DeepEval')
    parser.add_argument('--dataset', default='data/dataset.json', help='Path to original dataset')
    parser.add_argument('--responses', required=True, help='Path to LLM responses JSON file')
    parser.add_argument('--output-dir', default='Results', help='Output directory for results')
    parser.add_argument('--eval-model', default='gpt-4o-mini', help='Model to use for evaluation')
    parser.add_argument('--metrics', nargs='+', default=['answer_relevancy'], 
                        choices=list(AVAILABLE_METRICS.keys()),
                        help='Metrics to use for evaluation')
    parser.add_argument('--threshold', type=float, default=0.7, help='Threshold for passing metrics')
    parser.add_argument('--batch-size', type=int, default=15, help='Batch size for evaluation')
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
    
    model_name = Path(args.responses).stem.replace('_responses', '')
    logging.info(f"Evaluating responses from {model_name} using {args.eval_model}")
    logging.info(f"Using reference type: {args.reference_type}")
    
    metrics = create_metrics(
        args.metrics, 
        threshold=args.threshold, 
        eval_model=args.eval_model
    )
    
    test_cases, metadata = create_test_cases(dataset, responses, args.reference_type)
    logging.info(f"Created {len(test_cases)} test cases")
    
    if not test_cases:
        logging.error("No valid test cases created. Exiting.")
        return
    
    deepeval_results = evaluate_test_cases(
        test_cases, 
        metrics,
        batch_size=args.batch_size
    )
    
    combined_results = extract_scores(deepeval_results, metadata, args.metrics)
    
    aggregated_metrics = compute_aggregated_metrics(combined_results, args.metrics)
    
    # Prepare results for saving
    detailed_results = {
        "reference_type": args.reference_type,
        "metrics_used": args.metrics,
        "results": combined_results
    }
    
    aggregated_results = {
        "reference_type": args.reference_type,
        "metrics_used": args.metrics,
        "metrics": aggregated_metrics
    }
    
    metrics_str = "_".join(args.metrics)
    
    detailed_output_path = output_dir / f"deepeval_detailed_{model_name}_{metrics_str}_{args.reference_type}.json".replace(':', '-')
    with open(detailed_output_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    aggregated_output_path = output_dir / f"deepeval_aggregated_{model_name}_{metrics_str}_{args.reference_type}.json".replace(':', '-')
    with open(aggregated_output_path, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    logging.info(f"Successfully saved detailed results to {detailed_output_path}")
    logging.info(f"Successfully saved aggregated results to {aggregated_output_path}")
    
    # Log overall results for each metric
    for metric_name in args.metrics:
        metric_overall = aggregated_metrics[metric_name]["overall"]
        logging.info(f"{metric_name.upper()} Metric:")
        logging.info(f"  Score: {metric_overall['score']:.4f}")
        logging.info(f"  Pass Rate: {metric_overall['pass_rate']:.2%}")
        
        logging.info("  Breakdown by Model:")
        for model, model_metrics in aggregated_metrics[metric_name]["by_model"].items():
            logging.info(f"    {model or 'base'}: Score = {model_metrics['score']:.4f}, Pass Rate = {model_metrics['pass_rate']:.2%}")
        
        # Log breakdown by question type
        logging.info("  Breakdown by Question Type:")
        for q_type, type_metrics in aggregated_metrics[metric_name]["by_question_type"].items():
            logging.info(f"    {q_type}: Score = {type_metrics['score']:.4f}, Pass Rate = {type_metrics['pass_rate']:.2%}")
        
        # Log breakdown by difficulty
        logging.info("  Breakdown by Difficulty:")
        for difficulty, diff_metrics in aggregated_metrics[metric_name]["by_difficulty"].items():
            logging.info(f"    {difficulty}: Score = {diff_metrics['score']:.4f}, Pass Rate = {diff_metrics['pass_rate']:.2%}")

if __name__ == "__main__":
    main()