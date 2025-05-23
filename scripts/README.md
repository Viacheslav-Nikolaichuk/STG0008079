# Scripts

BERTScore.py calculates and outputs the BERTScore for the selected response file in a two different json files, one detailed file for each response, and one aggregated file.

CollectMMStats.py displays stats about mental model usage as well as how many times each mental model is mentioned in the responses.

DeepEval.py calculates and outputs the DeepEval answer_relevancy and faithfulness scores for the selected response file in a two different json files, one detailed file for each response, and one aggregated file.

GetResponses.py collects all the LLM responses through their APIs and outputs them in the same json format at the custom dataset.

LowestScoreFinder.py finds and saves the overall lowest scores, and the per-file lowest scores from each metric in their own json file in Results/Lowest-Scores.

AverageScores.py finds the average scores per automated metric according to 0 mental models, one mental model, two mental models and "think hard prompt" and saves the results to the Results/Average-Scores directory. To only get the scores from the last questions (excluding the different temperature results) change line 61 in AverageScores.py to: if not ("lastquestions" in file_path.name.lower() and "1_0" in file_path.name.lower()): continue