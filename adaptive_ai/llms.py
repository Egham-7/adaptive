# Define the model capabilities and their suitable complexity ranges
model_capabilities = {
    "qwen-2.5-32b": {
        "description": "A high-capacity, general-purpose model with strong performance for diverse tasks.",
        "complexity_range": (0.3, 0.8)
    },
    "deepseek-r1-distill-qwen-32b": {
        "description": "A distilled version of Qwen, optimized for efficiency and responsiveness.",
        "complexity_range": (0.2, 0.7)
    },
    "deepseek-r1-distill-llama-70b": {
        "description": "A distilled version of Llama 70B that balances performance with efficiency.",
        "complexity_range": (0.4, 0.8)
    },
    "deepseek-r1-distill-llama-70b-specdec": {
        "description": "A specialized distilled Llama 70B model optimized for complex decoding tasks.",
        "complexity_range": (0.5, 0.9)
    },
    "gemma2-9b-it": {
        "description": "A compact model tailored for Italian language tasks and creative generation.",
        "complexity_range": (0.3, 0.6)
    },
    "distil-whisper-large-v3-en": {
        "description": "A distilled version of Whisper for efficient English speech-to-text processing.",
        "complexity_range": (0.1, 0.4)
    },
    "llama-3.1-8b-instant": {
        "description": "An 8B model designed for rapid inference in low-latency applications.",
        "complexity_range": (0.2, 0.5)
    },
    "llama-3.2-11b-vision-preview": {
        "description": "An 11B multimodal model with vision capabilities for complex tasks.",
        "complexity_range": (0.4, 0.8)
    },
    "llama-3.2-1b-preview": {
        "description": "A lightweight 1B model for quick responses and low-resource tasks.",
        "complexity_range": (0.1, 0.3)
    },
    "llama-3.2-3b-preview": {
        "description": "A balanced 3B model suited for moderate complexity tasks.",
        "complexity_range": (0.2, 0.5)
    },
    "llama-3.2-90b-vision-preview": {
        "description": "A large vision-enabled model for handling complex multimodal inputs.",
        "complexity_range": (0.5, 0.9)
    },
    "llama-3.3-70b-specdec": {
        "description": "A specialized decoding-optimized variant of the 70B model.",
        "complexity_range": (0.5, 0.9)
    },
    "llama-3.3-70b-versatile": {
        "description": "A versatile 70B model capable of handling a wide range of tasks.",
        "complexity_range": (0.3, 0.7)
    },
    "llama-guard-3-8b": {
        "description": "A robust 8B model optimized for safety and content filtering.",
        "complexity_range": (0.2, 0.5)
    },
    "llama3-70b-8192": {
        "description": "A 70B model with an extended 8192-token context window for detailed analysis.",
        "complexity_range": (0.4, 0.8)
    },
    "llama3-8b-8192": {
        "description": "An 8B model with an extended 8192-token context window for efficient processing.",
        "complexity_range": (0.2, 0.5)
    }
}


# Define the domain to model mapping
domain_model_mapping = {
    "Adult": ["llama-3.3-70b-versatile"],
    "Arts_and_Entertainment": ["gemma2-9b-it", "llama-3.3-70b-versatile"],
    "Autos_and_Vehicles": ["qwen-2.5-32b", "deepseek-r1-distill-llama-70b"],
    "Beauty_and_Fitness": ["llama-3.3-70b-versatile", "llama-guard-3-8b"],
    "Books_and_Literature": ["llama3-70b-8192", "llama-3.3-70b-versatile"],
    "Business_and_Industrial": ["qwen-2.5-32b", "llama-3.3-70b-versatile"],
    "Computers_and_Electronics": ["deepseek-r1-distill-qwen-32b", "deepseek-r1-distill-llama-70b-specdec"],
    "Finance": ["qwen-2.5-32b", "llama-3.3-70b-versatile"],
    "Food_and_Drink": ["llama-3.3-70b-versatile", "llama-3.2-1b-preview"],
    "Games": ["llama3-70b-8192", "deepseek-r1-distill-llama-70b-specdec"],
    "Health": ["deepseek-r1-distill-llama-70b", "llama-3.3-70b-versatile"],
    "Hobbies_and_Leisure": ["llama-3.3-70b-versatile", "llama-3.2-3b-preview"],
    "Home_and_Garden": ["llama-3.3-70b-versatile", "llama-guard-3-8b"],
    "Internet_and_Telecom": ["deepseek-r1-distill-qwen-32b", "llama-3.2-90b-vision-preview"],
    "Jobs_and_Education": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    "Law_and_Government": ["deepseek-r1-distill-llama-70b-specdec", "llama-3.3-70b-versatile"],
    "News": ["llama-3.3-70b-specdec", "llama-3.2-3b-preview"],
    "Online_Communities": ["qwen-2.5-32b", "llama-3.3-70b-versatile"],
    "People_and_Society": ["llama-3.3-70b-versatile", "llama-3.2-3b-preview"],
    "Pets_and_Animals": ["llama-3.2-1b-preview", "llama-3.3-70b-versatile"],
    "Real_Estate": ["qwen-2.5-32b", "llama-3.3-70b-versatile"],
    "Science": ["deepseek-r1-distill-llama-70b", "llama-3.3-70b-versatile"],
    "Sensitive_Subjects": ["llama-3.3-70b-versatile", "llama-guard-3-8b"],
    "Shopping": ["llama-3.2-11b-vision-preview", "llama-3.3-70b-versatile"],
    "Sports": ["llama3-8b-8192", "llama-3.3-70b-versatile"],
    "Travel_and_Transportation": ["llama-3.2-90b-vision-preview", "llama-3.3-70b-versatile"]
}

