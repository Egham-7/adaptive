from typing import TypedDict, Literal


class TaskTypeParametersType(TypedDict):
    Temperature: float
    TopP: float
    PresencePenalty: float
    FrequencyPenalty: float
    MaxCompletionTokens: int
    N: int


class ModelCapability(TypedDict):
    description: str
    provider: Literal["GROQ", "OpenAI", "DEEPSEEK", "Anthropic"]


model_capabilities = {
    # OpenAI
    "gpt-4o": {
        "description": "OpenAI's flagship GPT-4o model with multimodal capabilities and 128K context.",
        "provider": "OpenAI",
    },
    "gpt-4-turbo": {
        "description": "High-performance GPT-4-turbo with a 128K context window.",
        "provider": "OpenAI",
    },
    "gpt-3.5-turbo": {
        "description": "Mid-range GPT-3.5 model optimized for cost-effectiveness.",
        "provider": "OpenAI",
    },
    # Anthropic
    "claude-3-opus": {
        "description": "Anthropic's most advanced Claude-3 Opus model with 200K context.",
        "provider": "Anthropic",
    },
    "claude-3-sonnet": {
        "description": "Balanced Claude-3 Sonnet model suitable for most general-purpose tasks.",
        "provider": "Anthropic",
    },
    "claude-3-haiku": {
        "description": "Lightweight Claude-3 Haiku for fast, cost-efficient tasks (200K context).",
        "provider": "Anthropic",
    },
    # Meta (GroqAPI)
    "llama-3-70b-instruct": {
        "description": "Meta's 70B parameter model capable of advanced instruction following.",
        "provider": "GROQ",
    },
    "llama-3-8b-instruct": {
        "description": "Meta's lightweight 8B model for instruct-style tasks (~8-16K context).",
        "provider": "GROQ",
    },
    # Google (Gemma)
    "gemma-7b": {
        "description": "Google's 7B parameter open model suitable for light tasks.",
        "provider": "GROQ",
    },
    "gemma-2b": {
        "description": "Tiny model from Google optimized for ultra-lightweight use cases.",
        "provider": "GROQ",
    },
}

task_type_model_mapping = {
    "Open QA": {
        "easy": {"model": "gpt-3.5-turbo", "complexity_threshold": 0.25},
        "medium": {"model": "gpt-4-turbo", "complexity_threshold": 0.35},
        "hard": {"model": "gpt-4o", "complexity_threshold": 0.45},
    },
    "Closed QA": {
        "easy": {"model": "claude-3-haiku", "complexity_threshold": 0.2},
        "medium": {"model": "llama-3-70b-instruct", "complexity_threshold": 0.3},
        "hard": {"model": "claude-3-opus", "complexity_threshold": 0.6},
    },
    "Summarization": {
        "easy": {"model": "gpt-3.5-turbo", "complexity_threshold": 0.25},
        "medium": {"model": "claude-3-sonnet", "complexity_threshold": 0.35},
        "hard": {"model": "gpt-4o", "complexity_threshold": 0.65},
    },
    "Text Generation": {
        "easy": {"model": "llama-3-8b-instruct", "complexity_threshold": 0.15},
        "medium": {"model": "claude-3-sonnet", "complexity_threshold": 0.3},
        "hard": {"model": "gpt-4o", "complexity_threshold": 0.7},
    },
    "Code Generation": {
        "easy": {"model": "gpt-3.5-turbo", "complexity_threshold": 0.15},
        "medium": {"model": "gpt-4-turbo", "complexity_threshold": 0.3},
        "hard": {"model": "claude-3-opus", "complexity_threshold": 0.4},
    },
    "Chatbot": {
        "easy": {"model": "llama-3-8b-instruct", "complexity_threshold": 0.2},
        "medium": {"model": "gpt-4-turbo", "complexity_threshold": 0.3},
        "hard": {"model": "claude-3-opus", "complexity_threshold": 0.6},
    },
    "Classification": {
        "easy": {"model": "gemma-7b", "complexity_threshold": 0.15},
        "medium": {"model": "claude-3-haiku", "complexity_threshold": 0.25},
        "hard": {"model": "gpt-4-turbo", "complexity_threshold": 0.5},
    },
    "Rewrite": {
        "easy": {"model": "gemma-7b", "complexity_threshold": 0.1},
        "medium": {"model": "gpt-3.5-turbo", "complexity_threshold": 0.2},
        "hard": {"model": "gpt-4o", "complexity_threshold": 0.6},
    },
    "Brainstorming": {
        "easy": {"model": "llama-3-70b-instruct", "complexity_threshold": 0.1},
        "medium": {"model": "claude-3-opus", "complexity_threshold": 0.15},
        "hard": {"model": "gpt-4o", "complexity_threshold": 0.5},
    },
    "Extraction": {
        "easy": {"model": "llama-3-8b-instruct", "complexity_threshold": 0.1},
        "medium": {"model": "gpt-3.5-turbo", "complexity_threshold": 0.3},
        "hard": {"model": "claude-3-sonnet", "complexity_threshold": 0.6},
    },
    "Other": {
        "easy": {"model": "gemma-7b", "complexity_threshold": 0.2},
        "medium": {"model": "gpt-4-turbo", "complexity_threshold": 0.4},
        "hard": {"model": "claude-3-opus", "complexity_threshold": 0.7},
    },
}


task_type_parameters = {
    "Open QA": {
        "Temperature": 0.3,
        "TopP": 0.7,
        "PresencePenalty": 0.3,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 800,
        "N": 1,
    },
    "Closed QA": {
        "Temperature": 0.2,
        "TopP": 0.6,
        "PresencePenalty": 0.2,
        "FrequencyPenalty": 0.2,
        "MaxCompletionTokens": 600,
        "N": 1,
    },
    "Summarization": {
        "Temperature": 0.4,
        "TopP": 0.8,
        "PresencePenalty": 0.4,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 1000,
        "N": 1,
    },
    "Text Generation": {
        "Temperature": 0.7,
        "TopP": 0.9,
        "PresencePenalty": 0.5,
        "FrequencyPenalty": 0.4,
        "MaxCompletionTokens": 1200,
        "N": 1,
    },
    "Code Generation": {
        "Temperature": 0.2,
        "TopP": 0.6,
        "PresencePenalty": 0.2,
        "FrequencyPenalty": 0.2,
        "MaxCompletionTokens": 1500,
        "N": 1,
    },
    "Chatbot": {
        "Temperature": 0.7,
        "TopP": 0.9,
        "PresencePenalty": 0.6,
        "FrequencyPenalty": 0.5,
        "MaxCompletionTokens": 1000,
        "N": 1,
    },
    "Classification": {
        "Temperature": 0.2,
        "TopP": 0.6,
        "PresencePenalty": 0.2,
        "FrequencyPenalty": 0.2,
        "MaxCompletionTokens": 500,
        "N": 1,
    },
    "Rewrite": {
        "Temperature": 0.5,
        "TopP": 0.8,
        "PresencePenalty": 0.4,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 1000,
        "N": 1,
    },
    "Brainstorming": {
        "Temperature": 0.8,
        "TopP": 0.95,
        "PresencePenalty": 0.8,
        "FrequencyPenalty": 0.7,
        "MaxCompletionTokens": 1500,
        "N": 2,
    },
    "Extraction": {
        "Temperature": 0.3,
        "TopP": 0.7,
        "PresencePenalty": 0.3,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 800,
        "N": 1,
    },
    "Other": {
        "Temperature": 0.5,
        "TopP": 0.8,
        "PresencePenalty": 0.4,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 1000,
        "N": 1,
    },
}
