import json
import logging

def load_text_file(file_path: str) -> str:
    """Loads a text file (returns empty string if not found)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logging.warning(f"File {file_path} not found, returning empty string.")
        return ""

def load_json_file(file_path: str) -> dict:
    """Loads a JSON file (returns empty if not found)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"File {file_path} not found, returning empty dict.")
        return {}

def save_json_file(data: dict, file_path: str):
    """Saves a dict to disk as JSON."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    logging.debug(f"Saved JSON data to {file_path}")