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
    capability_vector: npt.NDArray[np.float64]  # [Creativity, Reasoning, Context, Domain, Constraints]

# Capability Dimension Definitions (0-1 scale):
# 1. Creativity: Originality, ideation, divergent thinking (0-1)
# 2. Reasoning: Logical problem-solving, step-by-step analysis (0-1)
# 3. Contextual Knowledge: Understanding of long context/world knowledge (0-1)
# 4. Domain Knowledge: Specialized expertise (coding, science, etc.) (0-1)
# 5. Constraints: Handles complex instructions/constraints well (0-1)

model_capabilities = {
    # OpenAI
    "gpt-4o": {
        "description": "OpenAI's flagship GPT-4o model with multimodal capabilities and 128K context.",
        "provider": "OpenAI",
        "capability_vector": np.array([0.95, 0.96, 0.97, 0.94, 0.98]),  # Best all-around
    },
    "gpt-4-turbo": {
        "description": "High-performance GPT-4-turbo with a 128K context window.",
        "provider": "OpenAI",
        "capability_vector": np.array([0.65, 0.68, 0.92, 0.80, 0.95]),
        # Still strong, but more balanced for general tasks
    },

    "gpt-3.5-turbo": {
        "description": "Mid-range GPT-3.5 model optimized for cost-effectiveness.",
        "provider": "OpenAI",
        "capability_vector": np.array([0.45, 0.50, 0.70, 0.60, 0.78]),
        # Better for light, simple code tasks
    },
    # Anthropic
    "claude-3-opus": {
        "description": "Anthropic's most advanced Claude-3 Opus model with 200K context.",
        "provider": "Anthropic",
        "capability_vector": np.array([0.70, 0.75, 0.96, 0.85, 0.99]),
        # Strong domain and constraint; lowered creativity/reasoning
    },
    "claude-3-sonnet": {
        "description": "Balanced Claude-3 Sonnet model suitable for most general-purpose tasks.",
        "provider": "Anthropic",
        "capability_vector": np.array([0.80, 0.85, 0.87, 0.82, 0.84]),
    },
    "claude-3-haiku": {
        "description": "Lightweight Claude-3 Haiku for fast, cost-efficient tasks (200K context).",
        "provider": "Anthropic",
        "capability_vector": np.array([0.65, 0.72, 0.75, 0.68, 0.70]),
    },

    # Meta (GroqAPI)
    "llama-3-70b-instruct": {
        "description": "Meta's 70B parameter model capable of advanced instruction following.",
        "provider": "GROQ",
        "capability_vector": np.array([0.85, 0.88, 0.86, 0.83, 0.87]),
    },
    "llama-3-8b-instruct": {
        "description": "Meta's lightweight 8B model for instruct-style tasks (~8-16K context).",
        "provider": "GROQ",
        "capability_vector": np.array([0.60, 0.65, 0.63, 0.58, 0.62]),
    },

    # Google (Gemma)
    "gemma-7b": {
        "description": "Google's 7B parameter open model suitable for light tasks.",
        "provider": "GROQ",
        "capability_vector": np.array([0.55, 0.60, 0.58, 0.52, 0.56]),
    },
    "gemma-2b": {
        "description": "Tiny model from Google optimized for ultra-lightweight use cases.",
        "provider": "GROQ",
        "capability_vector": np.array([0.40, 0.45, 0.42, 0.38, 0.41]),
    },
}

task_type_model_mapping = {
    "Open QA": {
        "easy": {
            "model": "gpt-3.5-turbo",
            "complexity_threshold": 0.3
        },
        "medium": {
            "model": "gpt-4-turbo",
            "complexity_threshold": 0.4
        },
        "hard": {
            "model": "gpt-4o",
            "complexity_threshold": 0.5
        }
    },
    "Closed QA": {
        "easy": {
            "model": "claude-3-haiku",
            "complexity_threshold": 0.2
        },
        "medium": {
            "model": "llama-3-70b-instruct",
            "complexity_threshold": 0.3
        },
        "hard": {
            "model": "claude-3-opus",
            "complexity_threshold": 0.6
        }
    },
    "Summarization": {
        "easy": {
            "model": "gpt-3.5-turbo",
            "complexity_threshold": 0.2
        },
        "medium": {
            "model": "claude-3-sonnet",
            "complexity_threshold": 0.35
        },
        "hard": {
            "model": "gpt-4o",
            "complexity_threshold": 0.65
        }
    },
    "Text Generation": {
        "easy": {
            "model": "llama-3-8b-instruct",
            "complexity_threshold": 0.15
        },
        "medium": {
            "model": "claude-3-sonnet",
            "complexity_threshold": 0.3
        },
        "hard": {
            "model": "gpt-4o",
            "complexity_threshold": 0.7
        }
    },
    "Code Generation": {
        "easy": {
            "model": "gpt-3.5-turbo",
            "complexity_threshold": 0.15
        },
        "medium": {
            "model": "gpt-4-turbo",
            "complexity_threshold": 0.3
        },
        "hard": {
            "model": "claude-3-opus",
            "complexity_threshold": 0.6
        }
    },
    "Chatbot": {
        "easy": {
            "model": "llama-3-8b-instruct",
            "complexity_threshold": 0.2
        },
        "medium": {
            "model": "gpt-4-turbo",
            "complexity_threshold": 0.3
        },
        "hard": {
            "model": "claude-3-opus",
            "complexity_threshold": 0.6
        }
    },
    "Classification": {
        "easy": {
            "model": "gemma-7b",
            "complexity_threshold": 0.15
        },
        "medium": {
            "model": "claude-3-haiku",
            "complexity_threshold": 0.25
        },
        "hard": {
            "model": "gpt-4-turbo",
            "complexity_threshold": 0.5
        }
    },
    "Rewrite": {
        "easy": {
            "model": "gemma-7b",
            "complexity_threshold": 0.1
        },
        "medium": {
            "model": "gpt-3.5-turbo",
            "complexity_threshold": 0.2
        },
        "hard": {
            "model": "gpt-4o",
            "complexity_threshold": 0.6
        }
    },
    "Brainstorming": {
        "easy": {
            "model": "llama-3-70b-instruct",
            "complexity_threshold": 0.1
        },
        "medium": {
            "model": "claude-3-opus",
            "complexity_threshold": 0.15
        },
        "hard": {
            "model": "gpt-4o",
            "complexity_threshold": 0.5
        }
    },
    "Extraction": {
        "easy": {
            "model": "llama-3-8b-instruct",
            "complexity_threshold": 0.1
        },
        "medium": {
            "model": "gpt-3.5-turbo",
            "complexity_threshold": 0.3
        },
        "hard": {
            "model": "claude-3-sonnet",
            "complexity_threshold": 0.6
        }
    },
    "Other": {
        "easy": {
            "model": "gemma-7b",
            "complexity_threshold": 0.2
        },
        "medium": {
            "model": "gpt-4-turbo",
            "complexity_threshold": 0.4
        },
        "hard": {
            "model": "claude-3-opus",
            "complexity_threshold": 0.7
        }
    }
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

# Task weights for different task types (multipliers)
# Format: [Creativity, Reasoning, Context, Domain, Constraints]
task_weights = {
    "Open QA":        [0.15, 0.25, 0.30, 0.20, 0.10],  # Focus on context and reasoning
    "Closed QA":      [0.10, 0.30, 0.25, 0.25, 0.10],  # Strong emphasis on reasoning and domain knowledge
    "Summarization":  [0.15, 0.20, 0.35, 0.20, 0.10],  # Context is crucial for good summaries
    "Text Generation":[0.35, 0.15, 0.20, 0.20, 0.10],  # Creativity is key for text generation
    "Code Generation":[0.10, 0.30, 0.25, 0.1, 0.15],  # Reasoning and domain knowledge are critical
    "Chatbot":        [0.25, 0.20, 0.25, 0.20, 0.10],  # Balanced but with some creativity
    "Classification": [0.10, 0.30, 0.20, 0.30, 0.10],  # Strong on reasoning and domain knowledge
    "Rewrite":        [0.20, 0.20, 0.25, 0.25, 0.10],  # Balanced across most aspects
    "Brainstorming":  [0.40, 0.20, 0.15, 0.15, 0.10],  # Creativity is paramount
    "Extraction":     [0.10, 0.25, 0.30, 0.25, 0.10],  # Context and reasoning are important
    "Other":          [0.20, 0.20, 0.20, 0.20, 0.20],  # Balanced weights for unknown tasks
}