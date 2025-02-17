# evaluate_text.py
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import json
import os

with open("config.json", "r") as f:
    config = json.load(f)

os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY")
os.environ["GEMINI_API_KEY"] = config.get("GEMINI_API_KEY")
os.environ["DEEPSEEK_API_KEY"] = config.get("DEEPSEEK_API_KEY")

def main():
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris is the capital of France.",
        retrieval_context=["Paris is the capital and largest city of France."]
    )

    # Initialize an evaluation metric.
    # Here, AnswerRelevancyMetric checks whether the generated answer is relevant and correct.
    # The threshold (e.g., 0.5) indicates the minimum acceptable score.
    metric = AnswerRelevancyMetric(threshold=0.5, model="gpt-4o")

    # Evaluate the test case using DeepEval's evaluate function.
    # The function returns a dictionary containing scores, reasons, and other details.
    results = evaluate([test_case], [metric])

    # Print out the evaluation results.
    print("Evaluation Results:")
    print(results)

if __name__ == "__main__":
    main()
