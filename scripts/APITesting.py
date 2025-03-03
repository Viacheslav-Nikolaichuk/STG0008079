import json
import requests
import logging
import argparse
import re
from pathlib import Path
from openai import OpenAI

# Enable basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING) # Disables http debug logs (from OpenAI's API)

def load_config():
    try:
        with open('config.json', 'r') as config_file:
            return json.load(config_file)
    except Exception as e:
        logging.error(f"Config error: {str(e)}")
        return None

def load_mental_models():
    """Load mental models and their descriptions from a JSON file"""
    try:
        with open("data/models.json", 'r') as f:
            models_data = json.load(f)
            models_dict = {item['model_name']: item['model_description'] 
                          for item in models_data if item['model_name']}
            return models_dict
    except Exception as e:
        logging.error(f"Error loading mental models: {str(e)}")
        return {}

class OllamaAPI:
    def __init__(self, config: dict, temperature: float = 0.7):
        self.model_name = config['model_name']
        self.base_url = config.get('base_url', "http://localhost:11434")
        self.api_endpoint = f"{self.base_url}/api/generate"
        self.conversation = []
        self.temperature = temperature

    def start_conversation(self):
        self.conversation = [{
            "role": "system",
            "content": ("You are an analytical assistant. Provide a direct, " 
                        "insightful response in a single clear sentence.")
        }]

    def generate_response(self, prompt: str) -> str:
        full_prompt = f"{prompt}\n\nAnswer in one sentence."
        self.conversation.append({"role": "user", "content": full_prompt})
        
        try:
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": self.model_name,
                    "prompt": "\n".join([msg["content"] for msg in self.conversation]),
                    "stream": False,
                    "temperature": self.temperature
                }
            )
            response.raise_for_status()
            return clean_response(response.json()['response'])
        except Exception as e:
            logging.error(f"Ollama error: {str(e)}")
            return "Error"

class DeepSeekAPI:
    def __init__(self, config: dict, temperature: float = 0.7):
        self.api_key = config['api_key']
        self.model_name = config['model_name']
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.conversation = []
        self.temperature = temperature

    def start_conversation(self):
        self.conversation = [{
            "role": "system",
            "content": ("You are an analytical assistant. Provide a direct, " 
                        "insightful response in a single clear sentence.")
        }]

    def generate_response(self, prompt: str) -> str:
        self.conversation.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model_name,
                    "messages": self.conversation,
                    "temperature": self.temperature
                }
            )
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            return clean_response(content)
        except Exception as e:
            logging.error(f"DeepSeek error: {str(e)}")
            return "Error"

class OpenAIAPI:
    def __init__(self, config: dict, temperature: float = 0.7):
        self.client = OpenAI(api_key=config['api_key'])
        self.model_name = config['model_name']
        self.conversation = []
        self.temperature = temperature

    def start_conversation(self):
        self.conversation = [{
            "role": "system",
            "content": ("You are an analytical assistant. Provide a direct, " 
                        "insightful response in a single clear sentence.")
        }]

    def generate_response(self, prompt):
        self.conversation.append({"role": "user", "content": prompt})
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation,
                temperature=self.temperature
            )
            return clean_response(completion.choices[0].message.content)
        except Exception as e:
            logging.error(f"OpenAI error: {str(e)}")
            return "Error"

class GeminiAPI:
    def __init__(self, config: dict, temperature: float = 0.7):
        self.api_key = config['api_key']
        self.model_name = config['model_name']
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        self.conversation = []
        self.temperature = temperature

    def start_conversation(self):
        self.conversation = [{
            "role": "user",
            "parts": [{"text": 
                ("You are an analytical assistant. Provide a direct, " 
                "insightful response in a single clear sentence.")}]
        }]

    def generate_response(self, prompt: str) -> str:
        self.conversation.append({"role": "user", "parts": [{"text": prompt}]})
        
        try:
            response = requests.post(
                self.api_url,
                params={"key": self.api_key},
                json={
                    "contents": self.conversation,
                    "generationConfig": {
                        "temperature": self.temperature
                    }
                }
            )
            response.raise_for_status()
            content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return clean_response(content)
        except Exception as e:
            logging.error(f"Gemini error: {str(e)}")
            return "Error"

def clean_response(response: str) -> str:
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    cleaned = cleaned.replace('**', '')
    cleaned = re.sub(r'\[.*?\]', '', cleaned)
    # Remove escape characters and clean up whitespace
    cleaned = cleaned.replace('\n', ' ') \
                     .replace('\u2019', "'") \
                     .replace('\u2018', "'") \
                     .replace('\u2014', '-') \
                     .replace('\u2013', '-') \
                     .replace('\u2026', '...') \
                     .replace('\"', '') \
                     .replace('\u2248', '≈') 
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    if cleaned and not cleaned.endswith(('.', '!', '?')):
        cleaned += '.'
    return cleaned if cleaned else "Error: Empty response"

class ModelFactory:
    @staticmethod
    def create_handler(model_name: str, config: dict, temperature: float = 0.7):
        model_config = config['models'].get(model_name)
        if not model_config:
            raise ValueError(f"No config for model: {model_name}")
        
        api_type = model_config['api_type']
        if api_type == "ollama":
            return OllamaAPI(model_config, temperature)
        elif api_type == "deepseek":
            return DeepSeekAPI(model_config, temperature)
        elif api_type == "openai":
            return OpenAIAPI(model_config, temperature)
        elif api_type == "gemini":
            return GeminiAPI(model_config, temperature)
        else:
            raise ValueError(f"Unsupported API type: {api_type}")

def process_scenario(input_scenario, handler, mental_models, use_descriptions=False):
    processed_scenario = {
        "id": input_scenario["id"],
        "context": input_scenario["context"],
        "questions": []
    }

    for input_question in input_scenario["questions"]:
        processed_question = {
            "id": input_question["id"],
            "question": input_question["question"],
            "question-type": input_question["question-type"],
            "difficulty": input_question["difficulty"],
            "ground_truth": input_question["ground_truth"],
            "model_answers": []
        }

        # Base answer
        handler.start_conversation()
        base_prompt = f"Context: {input_scenario['context']}\nQuestion: {input_question['question']}"
        base_response = handler.generate_response(base_prompt)
        processed_question["model_answers"].append({
            "model": "",
            "answer": base_response
        })

        # Mental model answers
        for model_answer in input_question["model_answers"][1:]:  # Skip base answer
            mental_model = model_answer["model"]
            handler.start_conversation()
            
            if '+' in mental_model:
                models = mental_model.split(' + ')
                if use_descriptions and mental_models:
                    model1_desc = mental_models.get(models[0], "")
                    model2_desc = mental_models.get(models[1], "")
                    prompt = (f"Context: {input_scenario['context']}\n"
                             f"Combine {models[0]} ({model1_desc}) and {models[1]} ({model2_desc}): "
                             f"{input_question['question']}")
                else:
                    prompt = f"Context: {input_scenario['context']}\nCombine {models[0]} and {models[1]}: {input_question['question']}"
            else:
                if use_descriptions and mental_models and mental_model in mental_models:
                    model_desc = mental_models.get(mental_model, "")
                    prompt = (f"Context: {input_scenario['context']}\n"
                             f"Use {mental_model} ({model_desc}): "
                             f"{input_question['question']}")
                else:
                    prompt = f"Context: {input_scenario['context']}\nUse {mental_model}: {input_question['question']}"
            
            response = handler.generate_response(prompt)
            processed_question["model_answers"].append({
                "model": mental_model,
                "answer": response
            })

        processed_scenario["questions"].append(processed_question)
    
    return processed_scenario

def process_dataset(input_file, output_dir, selected_models, use_descriptions=False, temperature=0.7):
    config = load_config()
    if not config:
        return

    with open(input_file, 'r') as f:
        input_data = json.load(f)
    
    # Load mental models if requested
    mental_models = None
    if use_descriptions:
        mental_models = load_mental_models()
        logging.info(f"Loaded {len(mental_models)} mental models with descriptions")

    if use_descriptions:
        output_directory = Path("LLM-Responses-with-description")
    else:
        output_directory = Path(output_dir)
    
    # Create the output directory if it doesn't exist
    output_directory.mkdir(exist_ok=True, parents=True)

    for model_name in selected_models:
        try:
            handler = ModelFactory.create_handler(model_name, config, temperature)
            logging.info(f"Created handler for {model_name} with temperature={temperature}")
        except Exception as e:
            logging.error(f"Skipping {model_name}: {str(e)}")
            continue

        output_scenarios = []
        
        # Process each scenario
        for input_scenario in input_data["scenarios"]:
            scenario_id = input_scenario['id']
            logging.info(f"Processing scenario {scenario_id} with {model_name}")
            processed = process_scenario(input_scenario, handler, mental_models, use_descriptions)
            output_scenarios.append(processed)

        # Include temperature in the filename
        temp_suffix = f"_temp{temperature}".replace('.', '_')
        output_file = output_directory / f"{model_name}{temp_suffix}_responses.json"
        
        with open(output_file, 'w') as f:
            json.dump({"scenarios": output_scenarios}, f, indent=4)

    logging.info(f"Processing complete. Output in {output_directory}")

def main():
    parser = argparse.ArgumentParser(description='LLM Scenario Processor')
    parser.add_argument('--input', default='data/TempQuestions.json', help='Input dataset path')
    parser.add_argument('--output', default='LLM-responses', help='Output directory')
    parser.add_argument('--models', nargs='+', 
                        default=['llama3.1:8b'],
                        help='Models from config to use')
    parser.add_argument('--use-descriptions', action='store_true',
                        help='Include mental model descriptions in prompts')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature setting for LLM responses (0.0-2.0)')
    args = parser.parse_args()
    
    if args.temperature < 0.0 or args.temperature > 2.0:
        logging.warning(f"Temperature {args.temperature} is outside the range (0.0-2.0)")
    
    # Create output directory if it doesn't exist
    Path(args.output).mkdir(exist_ok=True, parents=True)
    
    process_dataset(
        args.input, 
        args.output, 
        args.models, 
        args.use_descriptions,
        args.temperature
    )

if __name__ == "__main__":
    main()