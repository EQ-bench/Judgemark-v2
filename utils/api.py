import os
import time
import logging
import requests
from typing import List, Dict
from dotenv import load_dotenv
import random
import string

# Load environment variables from .env if present
load_dotenv()

BASE_URL = os.getenv("OPENAI_API_URL", "https://openrouter.ai/api/v1/chat/completions")
API_KEY = os.getenv("OPENAI_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))

import random

def send_to_judge_model(
    messages: List[Dict],
    judge_model: str,
    include_seed: bool = True,
    max_retries: int = MAX_RETRIES
) -> str:
    """
    Sends user messages to the judge model with basic retry logic and a timeout of 30 seconds.
    Expects an OpenAI-compatible endpoint.
    
    Args:
        messages (List[Dict]): List of message dictionaries.
        judge_model (str): The model to be used.
        include_seed (bool, optional): Whether to include a random seed block. Default is False.
        max_retries (int, optional): Maximum number of retries. Default is MAX_RETRIES.
        
    Returns:
        str: The content of the judge model's response.
    """
    if include_seed:
        # Generate 5 lines of 80 random alphanumeric characters
        
        seed_lines = [
            ''.join(random.choices(string.ascii_letters + string.digits, k=80))            
            for _ in range(5)
        ]
        
        
        random_seed_block = (
            "<RANDOM SEED PLEASE IGNORE>\n" +
            "\n".join(seed_lines) +
            "\n</RANDOM SEED>"
        )        
        
        # Prepend the random seed block as a system message
        messages = [{"role": "system", "content": random_seed_block}] + messages
        #messages[0]["content"] = random_seed_block + '\n\n' + messages[0]["content"]

    for attempt in range(1, max_retries + 1):
        try:
            data = {
                "model": judge_model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 8192, #4096,
                #"provider": {
                #    "order": [
                #       "DeepInfra", # llama-3.1-8b, mistral-small-3, qwen-72b
                #        "Mistral" # mistral-small-3
                #        "Lambda", # llama-3.1-8b
                #        "NovitaAI",  # qwen-72b, llama-3.1-8b
                #        "Nebius AI Studio", # qwen-72b
                #        "Hyperbolic", # qwen-72b
                #        "inference.net", # llama-3.1-8b
                #    ],
                #    "allow_fallbacks": False
                #}
            }
            
            response = requests.post(BASE_URL, headers=HEADERS, json=data, timeout=REQUEST_TIMEOUT)
            
            response.raise_for_status()
            res_json = response.json()
            return res_json['choices'][0]['message']['content']
        except requests.Timeout:
            logging.error(f"Request timed out on attempt {attempt} for judge model {judge_model}")
        except Exception as e:
            logging.error(f"Error on attempt {attempt} for judge model {judge_model}: {e}")

        if attempt == max_retries:
            logging.critical(f"Max retries reached for judge model {judge_model}")
            raise

        time.sleep(RETRY_DELAY)

    return ""