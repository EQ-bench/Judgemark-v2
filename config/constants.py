"""Global constants and reference scores for the Judgemark-v2 benchmark."""

# Stability test configuration
STABILITY_ITEMS = [
    ("gemma-2b-it", "2", "28"),
    ("Llama-3-70b-chat-hf", "1", "42"),
    ("DeepSeek-R1", "1", "2"),
]
STABILITY_REPS = 100

# Reference scores for correlation

REFERENCE_MODEL_SCORES = {
    
    "kimi-k2": 1387,
    "claude-opus-4": 1417,
    "claude-sonnet-4": 1380,
    "chatgpt-4o-latest": 1425,
    "gpt-4.1": 1399,
    "qwen3-235b-a22b": 1366,
    "gemma-3-27b-it": 1355,
    "mistral-small-3.2-24b": 1334,
    "reka-flash-3": 1250,
    "grok-3-beta": 1401,
    "gpt-4.1-mini": 1349,
    "gemma-3-12b-it": 1333,
    "gemma-3-4b-it": 1282,
    "gpt-4.1-nano": 1309,
}

# Negative criteria markers for score computation
NEGATIVE_MARKERS = [
        "meandering", "weak dialogue", "tell-don't-show", "unsurprising or uncreative", "amateurish", "purple prose", "overwrought", "incongruent ending positivity", "unearned transformations"
    ]

MODEL_NAME_REPLACEMENTS = {
    "mistralai/ministral-3b": "ministral/Ministral-3b-instruct",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "deepseek/deepseek-r1": "deepseek-ai/deepseek-r1",
    "anthropic/claude-3.5-haiku-20241022": "claude-3.5-haiku-20241022",
    "anthropic/claude-3.5-sonnet-20240620": "claude-3.5-sonnet-20240620",
    "openai/gpt-4o-2024-11-20": "gpt-4o-2024-11-20",
    "deepseek/deepseek-r1-distill-llama-70b": "deepseek-ai/deepseek-r1-distill-llama-70b",
    "mistralai/mistral-large-2411": "mistralai/mistral-large-instruct-2411",
    "cohere/command-r-08-2024": "CohereForAI/c4ai-command-r-08-2024",
    "google/gemini-pro-1.5": "gemini-pro-1_5",
    "openai/o3-mini": "o3-mini",
    "qwen/qwen-plus": "qwen-plus",
    "qwen/qwen-max": "qwen-max",
    "qwen/qwen-2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
    "google/gemini-2.0-flash-001": "gemini-2.0-flash-001",
    "google/gemini-2-0-flash-001": "gemini-2.0-flash-001",
    "anthropic/claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
    "openrouter/quasar-alpha": "quasar-alpha",
    "google/gemini-2.5-pro-preview-03-25": "gemini-2.5-pro-preview-03-25",
    "google/gemini-2.5-flash-preview": "gemini-2.5-flash-preview",
    #"qwen/qwen3-235b-a22b": "qwen/qwen3-235b-a22b:thinking",
    "mistralai/mistral-medium-3": "mistral-medium-3",
    "openai/gpt-4.1": "gpt-4.1",
    "google/gemini-2.5-flash-preview-05-20": "gemini-2.5-flash-preview-05-20",
    "openai/o3": "o3:low-reasoning",
    "mistralai/mistral-small-3.2-24b-instruct": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "mistralai/devstral-small": "mistralai/Devstral-Small-2507",
    "mistralai/devstral-medium": "devstral-medium",
    "tencent/hunyuan-a13b-instruct:free": "tencent/Hunyuan-A13B-Instruct",
    "moonshotai/kimi-k2": "moonshotai/Kimi-K2-Instruct",
    "gpt-5-2025-08-07": "gpt-5-2025-08-07:minimal-reasoning",
    "gpt-5-mini-2025-08-07": "gpt-5-mini-2025-08-07:minimal-reasoning",
    "gpt-5-nano-2025-08-07": "gpt-5-nano-2025-08-07:minimal-reasoning",
    "google/gemini-2.5-flash-lite-preview-06-17": "gemini-2.5-flash-lite-preview-06-17",
    "google/gemini-2.5-pro": "gemini-2.5-pro",
    "google/gemini-2.5-flash": "gemini-2.5-flash",
    "openai/gpt-4.1-mini": "gpt-4.1-mini",
    "google/gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gpt-5-nano-2025-08-07,gpt-5-nano-2025-08-07,gpt-5-nano-2025-08-07,gpt-5-nano-2025-08-07,gpt-5-nano-2025-08-07": "gpt-5-nano-2025-08-07__5x-ensemble",
    "anthropic/claude-sonnet-4": "claude-sonnet-4",
}
