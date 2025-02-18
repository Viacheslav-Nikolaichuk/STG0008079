import json
import requests
import logging
from pathlib import Path
from typing import Dict, Any
from openai import OpenAI
import getpass

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    OPENAI_API_KEY = config["OPENAI_API_KEY"]
    GEMINI_API_KEY = config["GEMINI_API_KEY"]
    DEEPSEEK_API_KEY = config["DEEPSEEK_API_KEY"]

except FileNotFoundError:
    print("Configuration file 'config.json' not found.")
    import sys

    sys.exit(1)
except KeyError as e:
    print(f"Key {e} not found in configuration file.")
    import sys

    sys.exit(1)


class LlamaInteraction:
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the llama model
        """
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            logging.error(f"Error communicating with llama: {e}")
            return f"Error: {str(e)}"


class DeepSeekInteraction:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"

    def generate_response(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logging.error(f"Error communicating with DeepSeek: {e}")
            return f"Error: {str(e)}"
        except KeyError as e:
            logging.error(f"Unexpected response format from DeepSeek: {e}")
            return "Error: Unexpected API response format"


class OpenAIInteraction:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the OpenAI model
        """
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error communicating with OpenAI: {e}")
            return f"Error: {str(e)}"


class GeminiInteraction:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-thinking-exp-01-21:generateContent"
        # self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    def generate_response(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
        }
        params = {"key": self.api_key}
        data = {"contents": [{"parts": [{"text": prompt}]}]}

        try:
            response = requests.post(
                self.api_url, headers=headers, params=params, json=data
            )
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0][
                "text"
            ]
        except requests.exceptions.RequestException as e:
            logging.error(f"Error communicating with Gemini: {e}")
            return f"Error: {str(e)}"


def process_json_file(
    input_file: str,
    llama_output: str,
    openai_output: str,
    gemini_output: str,
    deepseek_output: str,
    llama: LlamaInteraction,
    openai: OpenAIInteraction,
    gemini: GeminiInteraction,
    deepseek: DeepSeekInteraction,
):
    """
    Process questions from a JSON file and write responses from three LLMs to separate text files
    """
    try:
        # Read JSON file
        with open(input_file, "r") as f:
            data = json.load(f)

        # Create or open output files
        with (
            open(llama_output, "w") as llama_f,
            open(openai_output, "w") as openai_f,
            open(gemini_output, "w") as gemini_f,
            open(deepseek_output, "w") as deepseek_f,
        ):

            # Process each question
            for i, item in enumerate(data, 1):
                question = item.get("question", "")
                if not question:
                    logging.warning(f"Skipping item {i}: No question found")
                    continue

                # Get response from DeepSeek
                logging.info(
                    f"Processing question {i} with DeepSeek: {question[:50]}..."
                )
                deepseek_response = deepseek.generate_response(question)
                deepseek_f.write(f"Question {i}: {question}\n")
                deepseek_f.write(f"Answer {i}: {deepseek_response}\n")
                deepseek_f.write("-" * 80 + "\n\n")

                # Get response from llama
                logging.info(
                    f"Processing question {i} with llama: {question[:50]}..."
                )
                llama_response = llama.generate_response(question)
                llama_f.write(f"Question {i}: {question}\n")
                llama_f.write(f"Answer {i}: {llama_response}\n")
                llama_f.write("-" * 80 + "\n\n")

                # Get response from OpenAI
                logging.info(
                    f"Processing question {i} with OpenAI: {question[:50]}..."
                )
                openai_response = openai.generate_response(question)
                openai_f.write(f"Question {i}: {question}\n")
                openai_f.write(f"Answer {i}: {openai_response}\n")
                openai_f.write("-" * 80 + "\n\n")

                # Get response from Gemini
                logging.info(
                    f"Processing question {i} with Gemini: {question[:50]}..."
                )
                gemini_response = gemini.generate_response(question)
                gemini_f.write(f"Question {i}: {question}\n")
                gemini_f.write(f"Answer {i}: {gemini_response}\n")
                gemini_f.write("-" * 80 + "\n\n")

        logging.info(
            f"Processing complete. Responses written to {llama_output}, {openai_output}, and {gemini_output}"
        )

    except json.JSONDecodeError as e:
        logging.error(f"Error reading JSON file: {e}")
    except IOError as e:
        logging.error(f"Error handling files: {e}")


def main():
    # Configuration
    INPUT_FILE = "data/TempQuestions.json"
    LLAMA_OUTPUT_FILE = "LLM-responses/llama_responses.txt"
    OPENAI_OUTPUT_FILE = "LLM-responses/openai_responses.txt"
    GEMINI_OUTPUT_FILE = "LLM-responses/gemini_responses.txt"
    DEEPSEEK_OUTPUT_FILE = "LLM-responses/deepseek_responses.txt"
    LLAMA_MODEL_NAME = "llama3.1:8b"

    # Create interaction instances
    llama = LlamaInteraction(model_name=LLAMA_MODEL_NAME)
    openai = OpenAIInteraction(api_key=OPENAI_API_KEY)
    gemini = GeminiInteraction(api_key=GEMINI_API_KEY)
    deepseek = DeepSeekInteraction(api_key=DEEPSEEK_API_KEY)

    # Process the files
    process_json_file(
        INPUT_FILE,
        LLAMA_OUTPUT_FILE,
        OPENAI_OUTPUT_FILE,
        GEMINI_OUTPUT_FILE,
        DEEPSEEK_OUTPUT_FILE,
        llama,
        openai,
        gemini,
        deepseek,
    )


if __name__ == "__main__":
    main()
