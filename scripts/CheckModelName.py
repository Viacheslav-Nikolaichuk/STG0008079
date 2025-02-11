import json

def load_json(file_path):
    """Loads a JSON file and returns the data."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Error: Failed to load {file_path}")
        return None

def check_models_in_dataset(dataset_file_path, models_file_path):
    """
    Checks:
    1. Models in the dataset that do not exist in the models file.
    2. Models in the models file that are not used in the dataset.
    """
    dataset = load_json(dataset_file_path)
    models_data = load_json(models_file_path)

    if not dataset or not models_data:
        return

    # Create a set of valid model names (case-insensitive)
    valid_models = {model['model_name'].strip().lower() for model in models_data if isinstance(model, dict)}

    # Track all models used in dataset
    used_models = set()

    for scenario in dataset.get('scenarios', []):  # Now iterating over a list
        for question in scenario.get('questions', []):  # Questions are also a list now
            for model_answer in question.get('model_answers', []):
                if isinstance(model_answer, dict) and 'model' in model_answer:
                    for model in map(str.strip, model_answer['model'].split(' + ')):  # Split only on ' + ' (not spaces)
                        model_lower = model.lower()
                        used_models.add(model_lower)
                        if model_lower not in valid_models:
                            print(f"✗ The model '{model}' does NOT exist in the file.")

    # Find models in models.json that are not used in the dataset
    unused_models = valid_models - used_models
    if unused_models:
        print("\n✓ Models in models.json but NOT used in the dataset:")
        for model in sorted(unused_models):
            print(f"  - {model}")

def main():
    check_models_in_dataset('data/dataset.json', 'data/models.json')

if __name__ == "__main__":
    main()
