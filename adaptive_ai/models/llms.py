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
    provider: Literal["GROQ", "OpenAI", "DEEPSEEK", "Anthropic", "Google"]


model_capabilities = {
    # OpenAI
    "o3": {
        "description": "OpenAI's base model optimized for general tasks.",
        "provider": "OpenAI",
    },
    "o4-mini": {
        "description": "Compact version of OpenAI's o4 model for efficient processing.",
        "provider": "OpenAI",
    },
    "gpt-4.1": {
        "description": "OpenAI's advanced GPT-4.1 model with enhanced capabilities.",
        "provider": "OpenAI",
    },
    "gpt-4o": {
        "description": "OpenAI's flagship GPT-4o model with multimodal capabilities.",
        "provider": "OpenAI",
    },
    "gpt-4.1-mini": {
        "description": "Lightweight version of GPT-4.1 for faster processing.",
        "provider": "OpenAI",
    },
    "gpt-4.1-nano": {
        "description": "Ultra-compact version of GPT-4.1 for minimal resource usage.",
        "provider": "OpenAI",
    },
    # Google (Gemini)
    "gemini-2.0-flash": {
        "description": "Google's high-performance Gemini 2.0 model for fast responses.",
        "provider": "Google",
    },
    "gemini-2.0-flash-lite": {
        "description": "Lightweight version of Gemini 2.0 for efficient processing.",
        "provider": "Google",
    },
    # Deepseek
    "deepseek-reasoner": {
        "description": "Deepseek's specialized model for complex reasoning tasks.",
        "provider": "DEEPSEEK",
    },
    "deepseek-chat": {
        "description": "Deepseek's conversational model optimized for chat interactions.",
        "provider": "DEEPSEEK",
    },
    # Anthropic
    "claude-sonnet-4-0": {
        "description": "Anthropic's balanced Claude Sonnet model for general tasks.",
        "provider": "Anthropic",
    },
    "claude-3-5-haiku-latest": {
        "description": "Latest version of Claude's lightweight Haiku model.",
        "provider": "Anthropic",
    },
    "claude-opus-4-0": {
        "description": "Anthropic's most advanced Claude Opus model for complex tasks.",
        "provider": "Anthropic",
    },
}

task_type_model_mapping = {
    "Open QA": {
        "easy": {"model": "o3", "complexity_threshold": 0.25},
        "medium": {"model": "gpt-4.1", "complexity_threshold": 0.35},
        "hard": {"model": "claude-opus-4-0", "complexity_threshold": 0.45},
    },
    "Closed QA": {
        "easy": {"model": "claude-3-5-haiku-latest", "complexity_threshold": 0.2},
        "medium": {"model": "gpt-4.1", "complexity_threshold": 0.3},
        "hard": {"model": "claude-opus-4-0", "complexity_threshold": 0.6},
    },
    "Summarization": {
        "easy": {"model": "o3", "complexity_threshold": 0.25},
        "medium": {"model": "claude-sonnet-4-0", "complexity_threshold": 0.35},
        "hard": {"model": "gpt-4.1", "complexity_threshold": 0.65},
    },
    "Text Generation": {
        "easy": {"model": "gemini-2.0-flash-lite", "complexity_threshold": 0.15},
        "medium": {"model": "claude-sonnet-4-0", "complexity_threshold": 0.3},
        "hard": {"model": "gpt-4.1", "complexity_threshold": 0.7},
    },
    "Code Generation": {
        "easy": {"model": "o3", "complexity_threshold": 0.15},
        "medium": {"model": "gpt-4.1", "complexity_threshold": 0.3},
        "hard": {"model": "claude-opus-4-0", "complexity_threshold": 0.4},
    },
    "Chatbot": {
        "easy": {"model": "gemini-2.0-flash-lite", "complexity_threshold": 0.2},
        "medium": {"model": "gpt-4.1", "complexity_threshold": 0.3},
        "hard": {"model": "claude-opus-4-0", "complexity_threshold": 0.6},
    },
    "Classification": {
        "easy": {"model": "o4-mini", "complexity_threshold": 0.15},
        "medium": {"model": "claude-3-5-haiku-latest", "complexity_threshold": 0.25},
        "hard": {"model": "gpt-4.1", "complexity_threshold": 0.5},
    },
    "Rewrite": {
        "easy": {"model": "o4-mini", "complexity_threshold": 0.1},
        "medium": {"model": "o3", "complexity_threshold": 0.2},
        "hard": {"model": "gpt-4.1", "complexity_threshold": 0.6},
    },
    "Brainstorming": {
        "easy": {"model": "gemini-2.0-flash", "complexity_threshold": 0.1},
        "medium": {"model": "claude-opus-4-0", "complexity_threshold": 0.15},
        "hard": {"model": "gpt-4.1", "complexity_threshold": 0.5},
    },
    "Extraction": {
        "easy": {"model": "gemini-2.0-flash-lite", "complexity_threshold": 0.1},
        "medium": {"model": "o3", "complexity_threshold": 0.3},
        "hard": {"model": "claude-sonnet-4-0", "complexity_threshold": 0.6},
    },
    "Other": {
        "easy": {"model": "o4-mini", "complexity_threshold": 0.2},
        "medium": {"model": "gpt-4.1", "complexity_threshold": 0.4},
        "hard": {"model": "claude-opus-4-0", "complexity_threshold": 0.7},
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
