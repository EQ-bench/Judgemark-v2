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

def generate_random_number_lines(num_lines=5, num_digits=80, frequency=None):
    if frequency is None:
        frequency = {i: 1 for i in range(11)}  # Default equal frequency
    
    # Create weighted choices based on frequency
    numbers = [num for num, freq in frequency.items() for _ in range(freq)]

    # Generate lines
    lines = [
        ' '.join(str(random.choice(numbers)) for _ in range(num_digits))
        for _ in range(num_lines)
    ]
    
    return lines

# Example usage
frequency_config = {
    0: 10,
    1: 9,
    2: 8,
    3: 7,
    4: 6,
    5: 4,
    6: 3,
    7: 3,
    8: 4,
    9: 6,
    10: 10
}

_frequency_config = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1
}


def send_to_judge_model(
    messages: List[Dict],
    judge_model: str,
    include_seed: bool = True,
    max_retries: int = MAX_RETRIES,
    temperature: float = 0
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
            #''.join(random.choices(string.digits, k=80))
            for _ in range(5)
        ]
        
        #seed_lines = generate_random_number_lines(num_lines=100, num_digits=80, frequency=frequency_config)
        random_seed_block = (
            "<RANDOM SEED PLEASE IGNORE>\n" +
            "\n".join(seed_lines) +
            "\n</RANDOM SEED>"
        )        
        
        # Prepend the random seed block as a system message
        messages = [{"role": "system", "content": random_seed_block}] + messages
        #messages[0]["content"] = random_seed_block + '\n\n' + messages[0]["content"]

    sysprompt = '''
You are to anchor your assigned scores to these descriptions of the scoring range:

1 - Uninspiring story, weak writing
2 - Below average 
3 - Mixed quality but reasonably readable
4 - Above average; promising writing
5 - Excellent
'''
    #messages = [{"role": "system", "content": sysprompt}] + messages

    if BASE_URL == "https://openrouter.ai/api/v1/chat/completions":
        if True and 'qwen3' in judge_model.lower():
            print('adding nothink')
            # optionally disable thinking for qwen3 models
            system_msg = [{"role": "system", "content": "/no_think"}]
            messages = system_msg + messages
            
    x = random.randint(10000,99999)
    last_err = None
    response = None
    for attempt in range(1, max_retries + 1):
        print(x, 'attempt', attempt, 'of', max_retries)
        try:
            data = {
                "model": judge_model,
                "messages": messages,
                #"temperature": 0.5,
                #"top_k": 3,
                "temperature": temperature,
                #"seed": random.randint(1,99999),
                #"top_k": 3,
                #"min_p": 0.1,
                "max_tokens": 12000, #4096, #16000, #8192, #4096,
                
            }

            if False:
                data["provider"] = {
                   "order": [
                        #"DeepInfra", # llama-3.1-8b, mistral-small-3, qwen-72b
                #        "Mistral" # mistral-small-3
                        #"Lambda", # llama-3.1-8b
                        "Novita",  # qwen-72b, llama-3.1-8b
                        #"Nebius AI Studio", # qwen-72b
                        #"Hyperbolic", # qwen-72b
                        #"inference.net", # llama-3.1-8b
                        #"Groq", # llama 3.1 8b
                        #"inference.net",
                    ],
                    "allow_fallbacks": False
                }

            if False and judge_model == 'moonshotai/kimi-k2':
                print('!kimi')
                data["provider"] = {
                   "order": [
                        #"DeepInfra", # llama-3.1-8b, mistral-small-3, qwen-72b
                        #"Mistral" # mistral-small-3
                        #"Lambda", # llama-3.1-8b
                        #"Novita",  # qwen-72b, llama-3.1-8b
                        #"Nebius AI Studio", # qwen-72b
                        #"Hyperbolic", # qwen-72b
                        #"inference.net", # llama-3.1-8b
                        #"Groq", # llama 3.1 8b
                        #"inference.net",
                        "Chutes"
                    ],
                    "allow_fallbacks": False
                }

            if judge_model == 'openai/o3':
                print('!! o3 low thinking')
                data["reasoning"] = {                
                    "effort": "low", # Can be "high", "medium", or "low" (OpenAI-style)
                    #"max_tokens": 50, # Specific token limit (Anthropic-style)                
                    "exclude": True #Set to true to exclude reasoning tokens from response
                }

            if judge_model in ['o3', 'o4-mini']:
                del data['max_tokens']
                del data['temperature']
                data['max_completion_tokens'] = 8096

            if judge_model in ['gpt-5-2025-08-07', 'gpt-5-mini-2025-08-07', 'gpt-5-nano-2025-08-07']:
                data['reasoning_effort']="minimal"
                del data['max_tokens']
                data['temperature']=1
                data['max_completion_tokens'] = 8096

            if judge_model in ['gpt-5-chat-latest']:
                del data['max_tokens']
                data['max_completion_tokens'] = 8096

            if judge_model in ['openai/gpt-oss-120b', 'openai/gpt-oss-20b']:
                data['provider'] =  {
                    "order": [
                        "Fireworks"
                    ],
                    "allow_fallbacks": False
                }
                data["reasoning"] = {                
                    "effort": "low", # Can be "high", "medium", or "low" (OpenAI-style)                    
                    "exclude": True #Set to true to exclude reasoning tokens from response
                }

            response = requests.post(BASE_URL, headers=HEADERS, json=data, timeout=REQUEST_TIMEOUT)

            response.raise_for_status()
            res_json = response.json()
            return res_json['choices'][0]['message']['content']
        except requests.Timeout as e:
            last_err = e
            logging.error(f"Request timed out on attempt {attempt} for judge model {judge_model}")
        except Exception as e:
            last_err = e
            if e.response is not None:
                logging.error(f"HTTP error on attempt {attempt} for judge model {judge_model}: "
                            f"{e} — Response body: {e.response.text}")
            else:
                logging.error(f"HTTP error on attempt {attempt} for judge model {judge_model}: {e}")

        
        if attempt == max_retries:
            # Use 'raise ... from ...' to chain the exception
            logging.critical(f"Max retries reached for judge model {judge_model}")
            raise RuntimeError(
                f"Max retries reached ({attempt} of {max_retries}) for judge model {judge_model} on attempt {attempt}"
            ) from last_err

        time.sleep(RETRY_DELAY)
    #print('ending req', x)
    return ""