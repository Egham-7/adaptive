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
            #[Creativity ,Reasoning, Contextual, Domain, Constraints]
            "vector": np.array([0.05, 0.2, 0.1,  0.5, 0.1]),
            "model": "gpt-3.5-turbo"
        },
        "medium": {
            "vector": np.array([0.1, 0.3, 0.1, 0.7, 0.2]),
            "model": "claude-4-turbo"
        },
        "hard": {
            "vector": np.array([0.25, 0.5, 0.35, 0.9, 0.4]),
            "model": "gpt-4o"
        }
    },
    "Closed QA": {
        "easy": {
            "vector": np.array([0.05, 0.2, 0.1,  0.5, 0.1]),
            "model": "claude-3-haiku"
        },
        "medium": {
            "vector": np.array([0.1, 0.3, 0.1, 0.7, 0.2]),
            "model": "llama-3-70b-instruct"
        },
        "hard": {
            "vector": np.array([0.25, 0.5, 0.35, 0.9, 0.4]),
            "model": "claude-3-opus"
        }
    },
    "Summarization": {
        "easy": {
            "vector": np.array([0.4, 0.5, 0.6, 0.4, 0.5]),
            "model": "gpt-3.5-turbo"
        },
        "medium": {
            "vector": np.array([0.7, 0.7, 0.8, 0.7, 0.8]),
            "model": "claude-3-sonnet"
        },
        "hard": {
            "vector": np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
            "model": "gpt-4o"
        }
    },
    "Text Generation": {
        "easy": {
            "vector": np.array([0.5, 0.5, 0.5, 0.4, 0.5]),
            "model": "llama-3-8b-instruct"
        },
        "medium": {
            "vector": np.array([0.7, 0.7, 0.8, 0.7, 0.8]),
            "model": "claude-3-sonnet"
        },
        "hard": {
            "vector": np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
            "model": "gpt-4o"
        }
    },
    "Code Generation": {
        "easy": {
            "vector": np.array([0.4, 0.5, 0.6, 0.5, 0.6]),
            "model": "gpt-3.5-turbo"
        },
        "medium": {
            "vector": np.array([0.6, 0.7, 0.8, 0.7, 0.8]),
            "model": "gpt-4-turbo"
        },
        "hard": {
            "vector": np.array([0.7, 0.8, 0.9, 0.8, 0.9]),
            "model": "claude-3-opus"
        }
    },
    "Chatbot": {
        "easy": {
            "vector": np.array([0.5, 0.5, 0.5, 0.4, 0.5]),
            "model": "llama-3-8b-instruct"
        },
        "medium": {
            "vector": np.array([0.6, 0.7, 0.8, 0.7, 0.8]),
            "model": "gpt-4-turbo"
        },
        "hard": {
            "vector": np.array([0.7, 0.8, 0.9, 0.8, 0.9]),
            "model": "claude-3-opus"
        }
    },
    "Classification": {
        "easy": {
            "vector": np.array([0.4, 0.5, 0.5, 0.4, 0.5]),
            "model": "gemma-7b"
        },
        "medium": {
            "vector": np.array([0.6, 0.7, 0.7, 0.6, 0.7]),
            "model": "claude-3-haiku"
        },
        "hard": {
            "vector": np.array([0.6, 0.7, 0.8, 0.7, 0.8]),
            "model": "gpt-4-turbo"
        }
    },
    "Rewrite": {
        "easy": {
            "vector": np.array([0.4, 0.5, 0.5, 0.4, 0.5]),
            "model": "gemma-7b"
        },
        "medium": {
            "vector": np.array([0.4, 0.5, 0.6, 0.5, 0.6]),
            "model": "gpt-3.5-turbo"
        },
        "hard": {
            "vector": np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
            "model": "gpt-4o"
        }
    },
    "Brainstorming": {
        "easy": {
            "vector": np.array([0.7, 0.7, 0.7, 0.6, 0.7]),
            "model": "llama-3-70b-instruct"
        },
        "medium": {
            "vector": np.array([0.7, 0.8, 0.9, 0.8, 0.9]),
            "model": "claude-3-opus"
        },
        "hard": {
            "vector": np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
            "model": "gpt-4o"
        }
    },
    "Extraction": {
        "easy": {
            "vector": np.array([0.5, 0.5, 0.5, 0.4, 0.5]),
            "model": "llama-3-8b-instruct"
        },
        "medium": {
            "vector": np.array([0.4, 0.5, 0.6, 0.5, 0.6]),
            "model": "gpt-3.5-turbo"
        },
        "hard": {
            "vector": np.array([0.7, 0.7, 0.8, 0.7, 0.8]),
            "model": "claude-3-sonnet"
        }
    },
    "Other": {
        "easy": {
            "vector": np.array([0.4, 0.5, 0.5, 0.4, 0.5]),
            "model": "gemma-7b"
        },
        "medium": {
            "vector": np.array([0.6, 0.7, 0.8, 0.7, 0.8]),
            "model": "gpt-4-turbo"
        },
        "hard": {
            "vector": np.array([0.7, 0.8, 0.9, 0.8, 0.9]),
            "model": "claude-3-opus"
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
# All weights set to 1.0 to effectively cancel weighting
task_weights = {
    "Open QA":        [1.0, 1.0, 1.0, 1.0, 1.0],
    "Closed QA":      [1.0, 1.0, 1.0, 1.0, 1.0],
    "Summarization":  [1.0, 1.0, 1.0, 1.0, 1.0],
    "Text Generation":[1.0, 1.0, 1.0, 1.0, 1.0],
    "Code Generation":[1.0, 1.0, 1.0, 1.0, 1.0],
    "Chatbot":        [1.0, 1.0, 1.0, 1.0, 1.0],
    "Classification": [1.0, 1.0, 1.0, 1.0, 1.0],
    "Rewrite":        [1.0, 1.0, 1.0, 1.0, 1.0],
    "Brainstorming":  [1.0, 1.0, 1.0, 1.0, 1.0],
    "Extraction":     [1.0, 1.0, 1.0, 1.0, 1.0],
    "Other":          [1.0, 1.0, 1.0, 1.0, 1.0],
}