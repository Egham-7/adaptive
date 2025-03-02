# Define the model capabilities and their suitable complexity ranges
model_capabilities = {
    "qwen-2.5-32b": {
        "description": "A high-capacity, general-purpose model with strong performance for diverse tasks.",
        "complexity_range": (0.3, 0.8),
        "provider": "GROQ"
    },
    "deepseek-r1-distill-qwen-32b": {
        "description": "A distilled version of Qwen, optimized for efficiency and responsiveness.",
        "complexity_range": (0.2, 0.7),
        "provider": "GROQ"
    },
    "deepseek-r1-distill-llama-70b": {
        "description": "A distilled version of Llama 70B that balances performance with efficiency.",
        "complexity_range": (0.4, 0.8),
        "provider": "GROQ"
    },
    "gpt-4-turbo": {
        "description": "OpenAI's latest high-performance model, optimized for cost and speed.",
        "complexity_range": (0.4, 0.9),
        "provider": "OPENAI"
    },
    "gpt-4": {
        "description": "A powerful and versatile model for complex reasoning and content generation.",
        "complexity_range": (0.5, 0.9),
        "provider": "OPENAI"
    },
    "gpt-3.5-turbo": {
        "description": "A cost-efficient and responsive model suitable for general tasks.",
        "complexity_range": (0.2, 0.6),
        "provider": "OPENAI"
    },
    "gpt-4o": {
        "description": "A highly optimized version of GPT-4 for enhanced multimodal reasoning.",
        "complexity_range": (0.5, 0.95),
        "provider": "OPENAI"
    },
    "deepseek-r1-distill-llama-70b-specdec": {
        "description": "A specialized distilled Llama 70B model optimized for complex decoding tasks.",
        "complexity_range": (0.5, 0.9),
        "provider": "GROQ"
    },
    "gemma2-9b-it": {
        "description": "A compact model tailored for Italian language tasks and creative generation.",
        "complexity_range": (0.3, 0.6),
        "provider": "GROQ"
    },
    "llama-3.3-70b-versatile": {
        "description": "A versatile 70B model capable of handling a wide range of tasks.",
        "complexity_range": (0.3, 0.7),
        "provider": "GROQ"
    }
}

# Define the domain to model mapping
domain_model_mapping = {
    "Adult": ["llama-3.3-70b-versatile", "gpt-4"],
    "Arts_and_Entertainment": ["gemma2-9b-it", "gpt-3.5-turbo"],
    "Autos_and_Vehicles": ["qwen-2.5-32b", "gpt-4"],
    "Beauty_and_Fitness": ["llama-3.3-70b-versatile", "gpt-4-turbo"],
    "Books_and_Literature": ["gpt-4", "llama-3.3-70b-versatile"],
    "Business_and_Industrial": ["qwen-2.5-32b", "gpt-4-turbo"],
    "Computers_and_Electronics": ["deepseek-r1-distill-qwen-32b", "gpt-4"],
    "Finance": ["qwen-2.5-32b", "gpt-4-turbo"],
    "Food_and_Drink": ["llama-3.3-70b-versatile", "gpt-3.5-turbo"],
    "Games": ["gpt-4-turbo", "deepseek-r1-distill-llama-70b-specdec"],
    "Health": ["deepseek-r1-distill-llama-70b", "gpt-4"],
    "Hobbies_and_Leisure": ["llama-3.3-70b-versatile", "gpt-3.5-turbo"],
    "Home_and_Garden": ["llama-3.3-70b-versatile", "gpt-4"],
    "Internet_and_Telecom": ["deepseek-r1-distill-qwen-32b", "gpt-4-turbo"],
    "Jobs_and_Education": ["gpt-3.5-turbo", "llama-3.3-70b-versatile"],
    "Law_and_Government": ["deepseek-r1-distill-llama-70b-specdec", "gpt-4"],
    "News": ["gpt-4-turbo", "llama-3.3-70b-versatile"],
    "Online_Communities": ["qwen-2.5-32b", "gpt-4-turbo"],
    "People_and_Society": ["llama-3.3-70b-versatile", "gpt-3.5-turbo"],
    "Pets_and_Animals": ["gpt-3.5-turbo", "llama-3.3-70b-versatile"],
    "Real_Estate": ["qwen-2.5-32b", "gpt-4-turbo"],
    "Science": ["deepseek-r1-distill-llama-70b", "gpt-4o"],
    "Sensitive_Subjects": ["llama-3.3-70b-versatile", "gpt-4-turbo"],
    "Shopping": ["gpt-4-turbo", "llama-3.3-70b-versatile"],
    "Sports": ["gpt-3.5-turbo", "llama-3.3-70b-versatile"],
    "Travel_and_Transportation": ["gpt-4", "llama-3.3-70b-versatile"]
}
