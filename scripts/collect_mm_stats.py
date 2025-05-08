from pathlib import Path
import json
import re
from collections import defaultdict, Counter

# 1. Load mental-model names
with open(Path("data") / "models.json", encoding="utf-8") as fp:
    mm_names = [m["model_name"] for m in json.load(fp) if m["model_name"]]

mm_pattern = re.compile(
    r"\b(" + "|".join(map(re.escape, mm_names)) + r")\b", re.I
)


# 2. Helpers
def classify(found: set[str], requested_mm: str) -> str:
    """
    Return one of 'correct', 'offlabel', or 'none' (see doc-string in OP).
    """
    if not found:
        return "none"

    requested_parts = [
        p.strip().lower() for p in requested_mm.split("+") if p.strip()
    ]
    if requested_parts and any(part in found for part in requested_parts):
        return "correct"
    return "offlabel"


def pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


# 3. Walk through every *responses.json file
root = Path("Results/LLM-Responses-with-description")
totals: dict[str, Counter] = defaultdict(Counter)
mm_usage_global = Counter()

for fpath in root.glob("*.json"):
    with open(fpath, encoding="utf-8") as fp:
        blob = json.load(fp)

    # deduce LLM name from filename
    llm_name = fpath.stem.replace("_temp1_0_responses", "").replace(
        "_responses", ""
    )

    for sc in blob["scenarios"]:
        for q in sc["questions"]:
            for ans in q["model_answers"]:
                text = ans["answer"]
                requested_mm = ans["model"]  # '' for baseline answers

                found = {m.group(0).lower() for m in mm_pattern.finditer(text)}

                mm_usage_global.update(found)

                bucket = classify(found, requested_mm)
                totals[llm_name][bucket] += 1

# 4. Print one summary line per LLM
header = f"{'Model':<25} {'No MM':>7} {'Correct MM':>12} {'Off-label':>12}"
print(header)
print("-" * len(header))

for llm in [
    "deepseek-r1-1.5b",
    "deepseek-r1-8b",
    "deepseek-r1",
    "deepseek-v3",
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
    "gemini-2.0-flash",
    "gemini-2.0-flash-thinking",
    "llama3.1-8b",
]:
    c = totals[llm]
    total = sum(c.values())
    print(
        f"{llm:<25} {pct(c['none'], total):>6} {pct(c['correct'], total):>12} {pct(c['offlabel'], total):>12}"
    )

# 5. Show the most-used mental models overall
print("\nTop of mental models across all answers:")
for name, count in mm_usage_global.most_common(40):
    print(f"  {name}  -  {count} mentions")