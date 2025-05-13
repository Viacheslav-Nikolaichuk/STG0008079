# Results

/LLM-Responses-with-description - Contains each LLMs responses in each their own json file, also including separate files for the 4 LLMs that were tested on the last 20 questions, and the different temperature results for Gemini-2.0-Flash-Thinking on the last 20 questions. These results are from the GetResponses.py script.

/Lowest-Scores - Contains the 50 overall lowest scores for each automated metric as well in their separate json files, as well as each their "per-file" lowest scores which displays the 15 lowest scores per Response file in the /LLM-Responses-with-description directory. These results are from the LowestScoreFinder.py script.

/Results-Bertscore-Aggregated - Contains the aggregated BERTScore results for all the response files.

/Results-Bertscore-Detailed - Contains the detailed BERTScore results from each response in the response files.

/Results-DeepEval-Aggregated - Contains the aggregated DeepEval results for all the response files.

/Results-DeepEval-Detailed - Contains the detailed DeepEval results from each response in the response files.

/Results-Human-Aggregated - Contains the aggregated results from the manual human evaluation by both evaluators for the four chosen LLMs in the last 20 questions.

/Results-Human-Detailed - Contains the detailed results from the manual human evaluation by both evaluators for the four chosen LLMs in the last 20 questions.