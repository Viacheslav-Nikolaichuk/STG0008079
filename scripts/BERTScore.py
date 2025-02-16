from evaluate import load
import json
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

bertscore = load("bertscore")

predictions = ["hello there", "general kenobi"]
references = ["hello there", "general kenobi"]

results = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en",
    model_type="microsoft/deberta-base-mnli",
    device=device,
    use_fast_tokenizer=True
)

# Convert results to JSON-serializable format
output_data = [{
    "prediction": pred,
    "reference": ref,
    "f1": float(f1),
    "precision": float(prec),
    "recall": float(rec)
} for pred, ref, f1, prec, rec in zip(
    predictions,
    references,
    results["f1"],
    results["precision"],
    results["recall"]
)]

with open("results/bertscore_results.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("Successfully saved results to bertscore_results.json")