from pathlib import Path
import json
import re
from collections import defaultdict, Counter

# load mental-model names (case-insensitive regex with word-boundaries)
with open(Path("data") / "models.json", encoding="utf-8") as fp:
    mm_names = [m["model_name"] for m in json.load(fp) if m["model_name"]]

# compile the set into one regex:  r"\b(Cost\-Benefit Analysis|Halo Effect|…)\b"
mm_pattern = re.compile(r"\b(" + "|".join(map(re.escape, mm_names)) + r")\b", re.I)


# helper to classify a single answer
def classify(text: str, requested_mm: str) -> str:
    """
    Return one of:
        'correct'   - answer explicitly names *the* requested MM
        'offlabel'  - answer names some MM but not the requested one
        'none'      - no MM name at all
    """
    found = set(m.group(0).lower() for m in mm_pattern.finditer(text))
    if not found:
        return "none"

    requested_parts = [p.strip().lower()
        for p in requested_mm.split("+") if p.strip()]
    if requested_parts and any(part in found for part in requested_parts):
        return "correct"
    return "offlabel"

# walk through every *responses.json file
root = Path("LLM-Responses-with-description")
totals: dict[str, Counter] = defaultdict(Counter)

for fpath in root.glob("*.json"):
    with open(fpath, encoding="utf-8") as fp:
        blob = json.load(fp)

    # deduce “LLM name” from filename, e.g.  deepseek-r1-1.5b_temp1_0_responses.json
    llm_name = fpath.stem.replace("_temp1_0_responses", "").replace("_responses", "")

    for sc in blob["scenarios"]:
        for q in sc["questions"]:
            for ans in q["model_answers"]:
                text          = ans["answer"]
                requested_mm  = ans["model"]          # '' for baseline answers
                bucket        = classify(text, requested_mm)
                totals[llm_name][bucket] += 1

# print one summary line per LLM
header = f"{'Model':<25} {'No MM':>7} {'Correct MM':>12} {'Off-label':>12}"
print(header)
print("-" * len(header))

def pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0

for llm in [
    "deepseek-r1-1.5b", "deepseek-r1-8b", "deepseek-r1",
    "deepseek-v3", "gpt-4o", "gpt-4o-mini", "o3-mini",
    "gemini-2.0-flash", "gemini-2.0-flash-thinking", "llama3.1-8b"
]:
    c = totals[llm]
    total = sum(c.values())
    print(f"{llm:<25} {pct(c['none'], total):>6} {pct(c['correct'], total):>12} {pct(c['offlabel'], total):>12}")
