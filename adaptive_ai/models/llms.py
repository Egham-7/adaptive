from typing import TypedDict, Tuple, Literal, Dict


class DomainParametersType(TypedDict):
    Temperature: float
    TopP: float
    PresencePenalty: float
    FrequencyPenalty: float
    MaxCompletionTokens: int
    N: int


class ModelCapability(TypedDict):
    description: str
    complexity_range: Tuple[float, float]
    provider: Literal["GROQ", "OpenAI", "DEEPSEEK"]


model_capabilities: Dict[str, ModelCapability] = {
    "gemma2-9b-it": {
        "description": "A compact 9B parameter model tailored for Italian language tasks and creative generation.",
        "complexity_range": (0.0, 0.15),  # Lower complexity range
        "provider": "GROQ",
    },
    "llama-3.3-70b-verstile": {
        "description": "A versatile 70B parameter model capable of handling a wide range of tasks.",
        "complexity_range": (0.20, 0.40),  # Higher complexity range for advanced tasks
        "provider": "GROQ",
    },
    "llama-3.1-8b-instant": {
        "description": "An 8B parameter model optimized for quick, instant responses.",
        "complexity_range": (0.0, 0.15),  # Lower complexity range for quick tasks
        "provider": "GROQ",
    },
    "llama-guard-3-8b": {
        "description": "An 8B parameter model with enhanced safety features and guardrails for sensitive applications.",
        "complexity_range": (0.0, 0.20),  # Guardrails for sensitive domains
        "provider": "GROQ",
    },
    "llama3-70b-8192": {
        "description": "A 70B parameter model with an extended context window (8192 tokens) for detailed analyses.",
        "complexity_range": (0.20, 0.40),  # Higher complexity range
        "provider": "GROQ",
    },
    "llama3-8b-8192": {
        "description": "An efficient 8B parameter model featuring an extended context window for concise tasks.",
        "complexity_range": (0.0, 0.15),  # Optimized for concise tasks
        "provider": "GROQ",
    },
    "mixtral-8x7b-32768": {
        "description": "A multi-model ensemble comprising eight 7B models with a 32768-token context window for complex tasks.",
        "complexity_range": (0.15, 0.30),  # For medium-to-high complexity tasks
        "provider": "GROQ",
    },
    "o1-mini": {
        "description": "A smaller version of the O1 model, designed for quicker and lighter tasks.",
        "complexity_range": (0.20, 1.0),  # For lightweight tasks
        "provider": "OpenAI",
    },
    "o1": {
        "description": "A powerful general-purpose model capable of handling a wide range of tasks efficiently.",
        "complexity_range": (0.20, 1.0),  # General-purpose, higher complexity range
        "provider": "OpenAI",
    },
}

domain_model_mapping = {
    "Adult": ["llama-3.3-70b-verstile", "llama-guard-3-8b", "o1"],
    "Arts_and_Entertainment": ["gemma2-9b-it", "llama-3.3-70b-verstile", "o1-mini"],
    "Autos_and_Vehicles": ["llama3-70b-8192", "llama3-8b-8192", "o1"],
    "Beauty_and_Fitness": ["llama-3.1-8b-instant", "llama-guard-3-8b", "o1-mini"],
    "Books_and_Literature": ["llama-3.3-70b-verstile", "llama3-70b-8192", "o1"],
    "Business_and_Industrial": ["mixtral-8x7b-32768", "llama3-8b-8192", "o1-mini"],
    "Computers_and_Electronics": ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "o1"],
    "Finance": ["mixtral-8x7b-32768", "llama-3.3-70b-verstile", "o1-mini"],
    "Food_and_Drink": ["llama-3.1-8b-instant", "llama-guard-3-8b", "o1"],
    "Games": ["llama3-70b-8192", "llama-guard-3-8b", "o1-mini"],
    "Health": ["llama3-8b-8192", "llama-guard-3-8b", "o1"],
    "Hobbies_and_Leisure": ["gemma2-9b-it", "llama-3.3-70b-verstile", "o1-mini"],
    "Home_and_Garden": ["llama-3.1-8b-instant", "llama-guard-3-8b", "o1"],
    "Internet_and_Telecom": ["mixtral-8x7b-32768", "llama-3.1-8b-instant", "o1-mini"],
    "Jobs_and_Education": ["llama-3.3-70b-verstile", "llama3-8b-8192", "o1"],
    "Law_and_Government": ["llama-guard-3-8b", "mixtral-8x7b-32768", "o1-mini"],
    "News": ["llama-3.3-70b-verstile", "llama3-70b-8192", "o1"],
    "Online_Communities": ["gemma2-9b-it", "llama-3.1-8b-instant", "o1-mini"],
    "People_and_Society": ["llama-3.3-70b-verstile", "llama3-8b-8192", "o1"],
    "Pets_and_Animals": ["llama-guard-3-8b", "llama-3.1-8b-instant", "o1-mini"],
    "Real_Estate": ["mixtral-8x7b-32768", "llama3-70b-8192", "o1"],
    "Science": ["llama3-8b-8192", "llama-3.3-70b-verstile", "o1-mini"],
    "Sensitive_Subjects": ["llama-guard-3-8b", "mixtral-8x7b-32768", "o1"],
    "Shopping": ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "o1-mini"],
    "Sports": ["llama3-70b-8192", "llama-3.3-70b-verstile", "o1"],
    "Travel_and_Transportation": [
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "o1-mini",
    ],
}


domain_parameters = {
    "Adult": {
        "Temperature": 0.7,
        "TopP": 0.9,
        "PresencePenalty": 0.5,
        "FrequencyPenalty": 0.4,
        "MaxCompletionTokens": 1200,
        "N": 1,
    },
    "Arts_and_Entertainment": {
        "Temperature": 0.8,
        "TopP": 0.95,
        "PresencePenalty": 1.0,
        "FrequencyPenalty": 0.8,
        "MaxCompletionTokens": 1000,
        "N": 2,
    },
    "Autos_and_Vehicles": {
        "Temperature": 0.6,
        "TopP": 0.85,
        "PresencePenalty": 0.4,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 1100,
        "N": 1,
    },
    "Beauty_and_Fitness": {
        "Temperature": 0.7,
        "TopP": 0.9,
        "PresencePenalty": 0.6,
        "FrequencyPenalty": 0.5,
        "MaxCompletionTokens": 1000,
        "N": 1,
    },
    "Books_and_Literature": {
        "Temperature": 0.5,
        "TopP": 0.8,
        "PresencePenalty": 0.4,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 1300,
        "N": 1,
    },
    "Business_and_Industrial": {
        "Temperature": 0.4,
        "TopP": 0.75,
        "PresencePenalty": 0.5,
        "FrequencyPenalty": 0.4,
        "MaxCompletionTokens": 900,
        "N": 1,
    },
    "Computers_and_Electronics": {
        "Temperature": 0.4,
        "TopP": 0.7,
        "PresencePenalty": 0.3,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 1500,
        "N": 1,
    },
    "Finance": {
        "Temperature": 0.3,
        "TopP": 0.7,
        "PresencePenalty": 0.5,
        "FrequencyPenalty": 0.4,
        "MaxCompletionTokens": 700,
        "N": 1,
    },
    "Food_and_Drink": {
        "Temperature": 0.6,
        "TopP": 0.85,
        "PresencePenalty": 0.5,
        "FrequencyPenalty": 0.4,
        "MaxCompletionTokens": 900,
        "N": 1,
    },
    "Games": {
        "Temperature": 0.9,
        "TopP": 1.0,
        "PresencePenalty": 1.0,
        "FrequencyPenalty": 0.9,
        "MaxCompletionTokens": 1500,
        "N": 2,
    },
    "Health": {
        "Temperature": 0.5,
        "TopP": 0.8,
        "PresencePenalty": 0.4,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 1200,
        "N": 1,
    },
    "Hobbies_and_Leisure": {
        "Temperature": 0.7,
        "TopP": 0.9,
        "PresencePenalty": 0.6,
        "FrequencyPenalty": 0.5,
        "MaxCompletionTokens": 1000,
        "N": 1,
    },
    "Home_and_Garden": {
        "Temperature": 0.5,
        "TopP": 0.8,
        "PresencePenalty": 0.4,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 1300,
        "N": 1,
    },
    "Internet_and_Telecom": {
        "Temperature": 0.4,
        "TopP": 0.75,
        "PresencePenalty": 0.3,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 1400,
        "N": 1,
    },
    "Jobs_and_Education": {
        "Temperature": 0.4,
        "TopP": 0.75,
        "PresencePenalty": 0.3,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 1100,
        "N": 1,
    },
    "Law_and_Government": {
        "Temperature": 0.4,
        "TopP": 0.75,
        "PresencePenalty": 0.3,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 1200,
        "N": 1,
    },
    "News": {
        "Temperature": 0.5,
        "TopP": 0.8,
        "PresencePenalty": 0.4,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 1200,
        "N": 1,
    },
    "Online_Communities": {
        "Temperature": 0.7,
        "TopP": 0.9,
        "PresencePenalty": 0.6,
        "FrequencyPenalty": 0.5,
        "MaxCompletionTokens": 1000,
        "N": 1,
    },
    "People_and_Society": {
        "Temperature": 0.7,
        "TopP": 0.9,
        "PresencePenalty": 0.6,
        "FrequencyPenalty": 0.5,
        "MaxCompletionTokens": 1000,
        "N": 1,
    },
    "Pets_and_Animals": {
        "Temperature": 0.6,
        "TopP": 0.85,
        "PresencePenalty": 0.5,
        "FrequencyPenalty": 0.4,
        "MaxCompletionTokens": 1000,
        "N": 1,
    },
    "Real_Estate": {
        "Temperature": 0.5,
        "TopP": 0.8,
        "PresencePenalty": 0.4,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 1200,
        "N": 1,
    },
    "Science": {
        "Temperature": 0.3,
        "TopP": 0.75,
        "PresencePenalty": 0.4,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 1200,
        "N": 1,
    },
    "Sensitive_Subjects": {
        "Temperature": 0.6,
        "TopP": 0.85,
        "PresencePenalty": 0.5,
        "FrequencyPenalty": 0.4,
        "MaxCompletionTokens": 1100,
        "N": 1,
    },
    "Shopping": {
        "Temperature": 0.6,
        "TopP": 0.85,
        "PresencePenalty": 0.5,
        "FrequencyPenalty": 0.4,
        "MaxCompletionTokens": 1000,
        "N": 1,
    },
    "Sports": {
        "Temperature": 0.7,
        "TopP": 0.9,
        "PresencePenalty": 0.6,
        "FrequencyPenalty": 0.5,
        "MaxCompletionTokens": 1000,
        "N": 1,
    },
    "Travel_and_Transportation": {
        "Temperature": 0.5,
        "TopP": 0.8,
        "PresencePenalty": 0.4,
        "FrequencyPenalty": 0.3,
        "MaxCompletionTokens": 1200,
        "N": 1,
    },
}
