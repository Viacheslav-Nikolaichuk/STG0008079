# Scripts

BERTScore.py calculates and outputs the BERTScore for the selected response file in a two different json files, one detailed file for each response, and one aggregated file.

CollectMMStats.py displays stats about mental model usage as well as how many times each mental model is mentioned in the responses.

DeepEval.py calculates and outputs the DeepEval answer_relevancy and faithfulness scores for the selected response file in a two different json files, one detailed file for each response, and one aggregated file.

GetResponses.py collects all the LLM responses through their APIs and outputs them in the same json format at the custom dataset.

LowestScoreFinder.py finds and saves the overall lowest scores, and the per-file lowest scores from each metric in their own json file in Results/Lowest-Scores.