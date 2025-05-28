from typing import TypedDict, Tuple, Literal, Dict
import numpy as np
import numpy.typing as npt


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
    capability_vector: npt.NDArray[np.float64]  # Specify capability_vector
model_capabilities = {
    # OpenAI
    "gpt-4o": {
        "description": "OpenAI's flagship GPT-4o model with multimodal capabilities and 128K context.",
        "provider": "OpenAI",
        "capability_vector": np.array([0.88, 0.92, 0.90, 0.94]),
    },
    "gpt-4-turbo": {
        "description": "High-performance GPT-4-turbo with a 128K context window for complex tasks.",
        "provider": "OpenAI",
        "capability_vector": np.array([0.84, 0.89, 0.86, 0.90]),
    },
    "gpt-3.5-turbo": {
        "description": "Mid-range GPT-3.5 model optimized for speed and cost-effectiveness (16K context).",
        "provider": "OpenAI",
        "capability_vector": np.array([0.58, 0.66, 0.60, 0.55]),
    },

    # Anthropic
    "claude-3-opus": {
        "description": "Anthropic's most advanced Claude-3 Opus model with 200K context.",
        "provider": "Anthropic",
        "capability_vector": np.array([0.87, 0.90, 0.92, 0.93]),
    },
    "claude-3-sonnet": {
        "description": "Balanced Claude-3 Sonnet model suitable for most general-purpose tasks.",
        "provider": "Anthropic",
        "capability_vector": np.array([0.72, 0.78, 0.76, 0.74]),
    },
    "claude-3-haiku": {
        "description": "Lightweight Claude-3 Haiku for fast, cost-efficient tasks (200K context).",
        "provider": "Anthropic",
        "capability_vector": np.array([0.45, 0.50, 0.52, 0.40]),
    },

    # Meta (GroqAPI)
    "llama-3-8b-instruct": {
        "description": "Meta's lightweight 8B model for instruct-style tasks (~8–16K context).",
        "provider": "GROQ",
        "capability_vector": np.array([0.42, 0.46, 0.48, 0.40]),
    },
    "llama-3-70b-instruct": {
        "description": "Meta's 70B parameter model capable of advanced instruction following.",
        "provider": "GROQ",
        "capability_vector": np.array([0.76, 0.83, 0.80, 0.82]),
    },

    # Google (Gemma)
    "gemma-7b": {
        "description": "Google's 7B parameter open model suitable for light summarization and generation tasks.",
        "provider": "GROQ",
        "capability_vector": np.array([0.38, 0.42, 0.45, 0.32]),
    },
    "gemma-2b": {
        "description": "Tiny model from Google optimized for ultra-lightweight use cases.",
        "provider": "GROQ",
        "capability_vector": np.array([0.22, 0.25, 0.24, 0.15]),
    },
}



task_type_model_mapping = {
    "Open QA":        ["gpt-4o",           "claude-3-opus",           "llama-3-70b-instruct"],
    "Closed QA":      ["llama-3-70b-instruct","claude-3-sonnet",      "gpt-4o"],
    "Summarization":  ["gemma-7b",         "gpt-4-turbo",             "gpt-3.5-turbo"],
    "Text Generation":["gpt-4o",           "claude-3-sonnet",         "llama-3-70b-instruct"],
    "Code Generation":["claude-3-opus",    "gpt-4-turbo",             "gpt-3.5-turbo"],
    "Chatbot":        ["claude-3-opus",    "gpt-4o",                  "llama-3-70b-instruct"],
    "Classification": ["gpt-4-turbo",      "claude-3-haiku",          "claude-3-sonnet"],
    "Rewrite":        ["gpt-4o",           "gpt-3.5-turbo",           "gemma-7b"],
    "Brainstorming":  ["gpt-4o",           "claude-3-opus",           "llama-3-70b-instruct"],
    "Extraction":     ["llama-3-8b-instruct","claude-3-sonnet",       "gpt-3.5-turbo"],
    "Other":          ["gpt-4-turbo",      "claude-3-opus",           "gemma-7b"],
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
