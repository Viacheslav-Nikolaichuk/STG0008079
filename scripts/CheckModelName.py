import json

def check_model_exists(json_file_path, model_name):
    """
    Check if a given model name exists in the JSON file.
    
    Args:
    json_file_path (str): Path to the JSON file
    model_name (str): Name of the model to search for
    
    Returns:
    bool: True if model exists, False otherwise
    """
    try:
        with open(json_file_path, 'r') as file:
            models = json.load(file)
        
        # Check if the model name exists (case-insensitive)
        for model in models:
            if model['model_name'].lower() == model_name.lower():
                return True
        
        return False
    
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found.")
        return False
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {json_file_path}")
        return False

def main():
    JSON_FILE_PATH = 'models.json'
    
    # Get model name from user input
    while True:
        model_name = input("Enter a model name to check (or 'quit' to exit): ").strip()
        
        # Allow user to quit
        if model_name.lower() == 'quit':
            break
        
        # Check if model exists
        if check_model_exists(JSON_FILE_PATH, model_name):
            print(f"✓ The model '{model_name}' ALREADY EXISTS in the file.")
        else:
            print(f"✗ The model '{model_name}' does NOT exist in the file.")

if __name__ == "__main__":
    main()