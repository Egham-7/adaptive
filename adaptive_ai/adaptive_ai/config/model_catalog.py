# config/model_catalog.py

from adaptive_ai.models import (
    ModelCapability,
    ProviderType,
    TaskModelEntry,
    TaskModelMapping,
    TaskType,
)
from adaptive_ai.models.llm_classification_models import DomainType

# --- ACTIVE PROVIDERS CONFIGURATION ---
# Only these providers will be used for model selection
# Other providers remain in the catalog but are inactive
ACTIVE_PROVIDERS = {
    ProviderType.OPENAI,
    ProviderType.GROQ,  # Fast inference provider
    ProviderType.GROK,  # X.AI's models (grok-3, grok-3-mini)
    ProviderType.DEEPSEEK,
    ProviderType.ADAPTIVE,  # Adaptive minion service
}

# --- In-memory map of model capabilities aggregated by provider ---
# This dictionary holds aggregated model capabilities, crucial for dynamic routing.
# The keys are ProviderType enums, and values are lists of ModelCapability objects.
provider_model_capabilities: dict[ProviderType, list[ModelCapability]] = {
    ProviderType.GOOGLE: [
        ModelCapability(
            description="Compact and very fast, suitable for simple, low-latency tasks.",
            provider=ProviderType.GOOGLE,
            model_name="gemini-2.5-flash-lite-preview-06-17",
            cost_per_1m_input_tokens=0.075,
            cost_per_1m_output_tokens=0.30,
            max_context_tokens=128000,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="Proprietary-Very-Small",
            latency_tier="very low",
        ),
        ModelCapability(
            description="Google's flexible and fast model, good for many tasks including complex reasoning.",
            provider=ProviderType.GOOGLE,
            model_name="gemini-2.5-flash",
            cost_per_1m_input_tokens=0.15,
            cost_per_1m_output_tokens=0.60,
            max_context_tokens=1_000_000,
            max_output_tokens=8192,
            supports_function_calling=True,
            languages_supported=["en", "es", "fr", "de", "ja", "ko", "zh"],
            model_size_params="Proprietary-Medium",
            latency_tier="medium",
        ),
        ModelCapability(
            description="Google's most capable model, designed for highly complex tasks and long contexts.",
            provider=ProviderType.GOOGLE,
            model_name="gemini-2.5-pro",
            cost_per_1m_input_tokens=1.25,
            cost_per_1m_output_tokens=10.00,
            max_context_tokens=1_000_000,
            max_output_tokens=8192,
            supports_function_calling=True,
            languages_supported=["en", "es", "fr", "de", "ja", "ko", "zh"],
            model_size_params="Proprietary-Large",
            latency_tier="high",
        ),
    ],
    ProviderType.MISTRAL: [
        ModelCapability(
            description="Compact yet powerful, known for efficiency and good performance on general tasks.",
            provider=ProviderType.MISTRAL,
            model_name="mistral-small-latest",
            cost_per_1m_input_tokens=0.10,
            cost_per_1m_output_tokens=0.30,
            max_context_tokens=32768,
            max_output_tokens=8192,
            supports_function_calling=True,
            languages_supported=["en", "fr", "de", "es", "it"],
            model_size_params="Proprietary-Small",
            latency_tier="low",
        ),
    ],
    ProviderType.OPENAI: [
        ModelCapability(
            description="Cost-effective, highly efficient version of GPT-4, suitable for simple queries.",
            provider=ProviderType.OPENAI,
            model_name="gpt-4.1-nano",
            cost_per_1m_input_tokens=0.10,
            cost_per_1m_output_tokens=0.40,
            max_context_tokens=128000,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="Proprietary-Mini",
            latency_tier="very low",
        ),
        ModelCapability(
            description="Optimized mini version of GPT-4o, offering good balance of cost and performance.",
            provider=ProviderType.OPENAI,
            model_name="gpt-4o-mini",
            cost_per_1m_input_tokens=0.15,
            cost_per_1m_output_tokens=0.60,
            max_context_tokens=128000,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=["en", "es", "fr", "de", "it", "ja", "ko", "zh"],
            model_size_params="Proprietary-Mini",
            latency_tier="low",
        ),
        ModelCapability(
            description="A more capable mini variant, mapping to OpenAI's efficient GPT-4 series.",
            provider=ProviderType.OPENAI,
            model_name="gpt-4.1-mini",
            cost_per_1m_input_tokens=0.40,
            cost_per_1m_output_tokens=1.60,
            max_context_tokens=128000,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="Proprietary-Mini",
            latency_tier="medium",
        ),
        ModelCapability(
            description="A foundational GPT-4 model for general purpose use.",
            provider=ProviderType.OPENAI,
            model_name="gpt-4.1",
            cost_per_1m_input_tokens=2.00,
            cost_per_1m_output_tokens=8.00,
            max_context_tokens=128000,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=["en", "es", "fr", "de", "it", "ja", "ko", "zh"],
            model_size_params="Proprietary-Large",
            latency_tier="medium",
        ),
        ModelCapability(
            description="OpenAI's latest flagship multimodal model.",
            provider=ProviderType.OPENAI,
            model_name="gpt-4o",
            cost_per_1m_input_tokens=2.50,
            cost_per_1m_output_tokens=10.00,
            max_context_tokens=128000,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=["en", "es", "fr", "de", "it", "ja", "ko", "zh"],
            model_size_params="Proprietary-Large",
            latency_tier="medium",
        ),
        ModelCapability(
            description="A very high-tier 'o' series model, likely equivalent to GPT-4 level.",
            provider=ProviderType.OPENAI,
            model_name="o3-mini",
            cost_per_1m_input_tokens=1.10,
            cost_per_1m_output_tokens=4.40,
            max_context_tokens=16385,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="Proprietary-Small",
            latency_tier="low",
        ),
        ModelCapability(
            description="A likely advanced 'o4' mini variant.",
            provider=ProviderType.OPENAI,
            model_name="o4-mini",
            cost_per_1m_input_tokens=1.10,
            cost_per_1m_output_tokens=4.40,
            max_context_tokens=128000,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="Proprietary-Small",
            latency_tier="low",
        ),
        ModelCapability(
            description="A premium 'o' series model for general tasks.",
            provider=ProviderType.OPENAI,
            model_name="o3",
            cost_per_1m_input_tokens=10.00,
            cost_per_1m_output_tokens=40.00,
            max_context_tokens=128000,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="Proprietary-Large",
            latency_tier="medium",
        ),
        ModelCapability(
            description="A likely next-generation OpenAI model, powerful and high-cost.",
            provider=ProviderType.OPENAI,
            model_name="gpt-4.5",
            cost_per_1m_input_tokens=75.00,
            cost_per_1m_output_tokens=150.00,
            max_context_tokens=256000,
            max_output_tokens=8192,
            supports_function_calling=True,
            languages_supported=["en", "es", "fr", "de", "it", "ja", "ko", "zh"],
            model_size_params="Proprietary-Ultra",
            latency_tier="high",
        ),
        ModelCapability(
            description="A very high-tier 'o' series model, likely flagship.",
            provider=ProviderType.OPENAI,
            model_name="o1",
            cost_per_1m_input_tokens=15.00,
            cost_per_1m_output_tokens=60.00,
            max_context_tokens=128000,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="Proprietary-Very-Large",
            latency_tier="high",
        ),
        ModelCapability(
            description="An extremely high-cost and powerful 'O1 Pro' variant.",
            provider=ProviderType.OPENAI,
            model_name="o1-pro",
            cost_per_1m_input_tokens=150.00,
            cost_per_1m_output_tokens=600.00,
            max_context_tokens=512000,
            max_output_tokens=8192,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="Proprietary-Extreme",
            latency_tier="very high",
        ),
    ],
    ProviderType.DEEPSEEK: [
        ModelCapability(
            description="An earlier version of DeepSeek v3, offering strong base capabilities.",
            provider=ProviderType.DEEPSEEK,
            model_name="deepseek-chat",
            cost_per_1m_input_tokens=0.14,
            cost_per_1m_output_tokens=0.28,
            max_context_tokens=128000,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=["en", "zh"],
            model_size_params="236B (MoE)",
            latency_tier="medium",
        ),
        ModelCapability(
            description="A specialized DeepSeek model for reasoning tasks.",
            provider=ProviderType.DEEPSEEK,
            model_name="deepseek-reasoner",
            cost_per_1m_input_tokens=0.55,
            cost_per_1m_output_tokens=2.19,
            max_context_tokens=128000,
            max_output_tokens=4096,
            supports_function_calling=False,
            languages_supported=["en", "zh"],
            model_size_params="685B (r1)",
            latency_tier="medium",
        ),
    ],
    ProviderType.GROQ: [
        # Add GROQ (fast inference) models here when available
    ],
    ProviderType.GROK: [
        ModelCapability(
            description="A mini version of Grok, optimized for very low latency and cost.",
            provider=ProviderType.GROK,
            model_name="grok-3-mini",
            cost_per_1m_input_tokens=0.30,
            cost_per_1m_output_tokens=0.50,
            max_context_tokens=8192,
            max_output_tokens=4096,
            supports_function_calling=False,
            languages_supported=["en"],
            model_size_params="Proprietary-Mini",
            latency_tier="very low",
        ),
        ModelCapability(
            description="Grok's general-purpose model, known for extreme speed.",
            provider=ProviderType.GROK,
            model_name="grok-3",
            cost_per_1m_input_tokens=3.00,
            cost_per_1m_output_tokens=15.00,
            max_context_tokens=32768,
            max_output_tokens=8192,
            supports_function_calling=False,
            languages_supported=["en"],
            model_size_params="Proprietary-Large",
            latency_tier="very low",
        ),
    ],
    ProviderType.ANTHROPIC: [
        ModelCapability(
            description="A very recent version of Claude Sonnet, focused on balanced performance.",
            provider=ProviderType.ANTHROPIC,
            model_name="claude-sonnet-4-20250514",
            cost_per_1m_input_tokens=3.00,
            cost_per_1m_output_tokens=15.00,
            max_context_tokens=200000,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="Proprietary-Sonnet",
            latency_tier="medium",
        ),
        ModelCapability(
            description="Anthropic's most intelligent and powerful model, with superior reasoning and long context.",
            provider=ProviderType.ANTHROPIC,
            model_name="claude-opus-4-20250514",
            cost_per_1m_input_tokens=15.00,
            cost_per_1m_output_tokens=75.00,
            max_context_tokens=200000,
            max_output_tokens=4096,
            supports_function_calling=True,
            languages_supported=["en"],
            model_size_params="Proprietary-Opus",
            latency_tier="high",
        ),
    ],
    ProviderType.HUGGINGFACE: [
        # General-purpose models around 10B params
        ModelCapability(
            description="Qwen2.5 14B - Alibaba's advanced multilingual model with strong reasoning capabilities.",
            provider=ProviderType.HUGGINGFACE,
            model_name="Qwen/Qwen2.5-14B-Instruct",
            cost_per_1m_input_tokens=0.12,
            cost_per_1m_output_tokens=0.12,
            max_context_tokens=32768,
            max_output_tokens=8192,
            supports_function_calling=True,
            languages_supported=["en", "zh", "es", "fr", "de", "ja", "ko"],
            model_size_params="14B",
            latency_tier="medium",
        ),
        ModelCapability(
            description="Meta's Llama 3.1 8B - Excellent general-purpose model with strong instruction following.",
            provider=ProviderType.HUGGINGFACE,
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            cost_per_1m_input_tokens=0.10,
            cost_per_1m_output_tokens=0.10,
            max_context_tokens=131072,
            max_output_tokens=4096,
            supports_function_calling=False,
            languages_supported=["en", "es", "fr", "de", "it", "pt", "hi", "th"],
            model_size_params="8B",
            latency_tier="low",
        ),
        # Code generation specialist
        ModelCapability(
            description="CodeLlama 13B - Meta's specialized model for code generation and understanding.",
            provider=ProviderType.HUGGINGFACE,
            model_name="codellama/CodeLlama-13b-Instruct-hf",
            cost_per_1m_input_tokens=0.11,
            cost_per_1m_output_tokens=0.11,
            max_context_tokens=16384,
            max_output_tokens=4096,
            supports_function_calling=False,
            languages_supported=["en"],
            model_size_params="13B",
            latency_tier="medium",
        ),
        # Conversational specialist
        ModelCapability(
            description="Mistral 7B Instruct v0.3 - Optimized for conversational AI and chat applications.",
            provider=ProviderType.HUGGINGFACE,
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            cost_per_1m_input_tokens=0.08,
            cost_per_1m_output_tokens=0.08,
            max_context_tokens=32768,
            max_output_tokens=4096,
            supports_function_calling=False,
            languages_supported=["en", "fr", "de", "es", "it"],
            model_size_params="7B",
            latency_tier="low",
        ),
        # Text generation and summarization specialist
        ModelCapability(
            description="FLAN-T5-XL - Google's instruction-tuned model excellent for summarization and text processing.",
            provider=ProviderType.HUGGINGFACE,
            model_name="google/flan-t5-xl",
            cost_per_1m_input_tokens=0.06,
            cost_per_1m_output_tokens=0.06,
            max_context_tokens=2048,
            max_output_tokens=1024,
            supports_function_calling=False,
            languages_supported=["en"],
            model_size_params="3B",
            latency_tier="very low",
        ),
        # Classification and extraction specialist
        ModelCapability(
            description="DeBERTa v3 Large - Microsoft's model optimized for classification and NLU tasks.",
            provider=ProviderType.HUGGINGFACE,
            model_name="microsoft/deberta-v3-large",
            cost_per_1m_input_tokens=0.04,
            cost_per_1m_output_tokens=0.04,
            max_context_tokens=512,
            max_output_tokens=512,
            supports_function_calling=False,
            languages_supported=["en"],
            model_size_params="304M",
            latency_tier="very low",
        ),
    ],
}

# --- In-memory map for TaskModelMapping (Remote/Standard LLM Models) ---
# This maps each TaskType to an ordered list of TaskModelEntry,
# where the first model is considered 'best' for that task type, and so on.
# --- ACTIVE TASK MODEL MAPPINGS (OPENAI, GROK & DEEPSEEK ONLY) ---
# Updated to only include active providers: OpenAI, GROK, and DeepSeek
task_model_mappings_data: dict[TaskType, TaskModelMapping] = {
    TaskType.OPEN_QA: TaskModelMapping(
        model_entries=[
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
        ]
    ),
    TaskType.CODE_GENERATION: TaskModelMapping(
        model_entries=[
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ]
    ),
    TaskType.SUMMARIZATION: TaskModelMapping(
        model_entries=[
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ]
    ),
    TaskType.TEXT_GENERATION: TaskModelMapping(
        model_entries=[
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
        ]
    ),
    TaskType.CHATBOT: TaskModelMapping(
        model_entries=[
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ]
    ),
    TaskType.CLOSED_QA: TaskModelMapping(
        model_entries=[
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ]
    ),
    TaskType.CLASSIFICATION: TaskModelMapping(
        model_entries=[
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1-nano"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ]
    ),
    TaskType.REWRITE: TaskModelMapping(
        model_entries=[
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ]
    ),
    TaskType.BRAINSTORMING: TaskModelMapping(
        model_entries=[
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ]
    ),
    TaskType.EXTRACTION: TaskModelMapping(
        model_entries=[
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1-nano"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ]
    ),
    TaskType.OTHER: TaskModelMapping(
        model_entries=[
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ]
    ),
}

# --- Minion Task Model Mappings (Adaptive Service Models) ---
# This maps each TaskType to a SINGLE designated adaptive service specialist model,
# each optimized for specific task types and available via Adaptive Minion Service

minion_task_model_mappings: dict[TaskType, str] = {
    TaskType.OPEN_QA: "Qwen/Qwen2.5-7B-Instruct",
    TaskType.CODE_GENERATION: "codellama/CodeLlama-7b-Instruct-hf",
    TaskType.SUMMARIZATION: "Qwen/Qwen2.5-7B-Instruct",
    TaskType.TEXT_GENERATION: "Qwen/Qwen2.5-14B-Instruct",
    TaskType.CHATBOT: "meta-llama/Meta-Llama-3-8B-Instruct",
    TaskType.CLASSIFICATION: "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
    TaskType.CLOSED_QA: "Qwen/Qwen2.5-14B-Instruct",
    TaskType.REWRITE: "microsoft/Phi-4-mini-reasoning",
    TaskType.BRAINSTORMING: "meta-llama/Meta-Llama-3-8B-Instruct",
    TaskType.EXTRACTION: "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
    TaskType.OTHER: "Qwen/Qwen2.5-7B-Instruct",
}

# --- Domain-Based Model Mappings for Adaptive Service ---
# This maps domain types to specialized models based on domain expertise

minion_domain_model_mappings: dict[DomainType, str] = {
    DomainType.BUSINESS_AND_INDUSTRIAL: "Qwen/Qwen2.5-14B-Instruct",
    DomainType.HEALTH: "Qwen/Qwen2.5-14B-Instruct",
    DomainType.NEWS: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.OTHERDOMAINS: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.REAL_ESTATE: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.COMPUTERS_AND_ELECTRONICS: "codellama/CodeLlama-7b-Instruct-hf",
    DomainType.INTERNET_AND_TELECOM: "codellama/CodeLlama-7b-Instruct-hf",
    DomainType.FINANCE: "Qwen/Qwen2.5-Math-7B-Instruct",
    DomainType.SCIENCE: "Qwen/Qwen2.5-Math-7B-Instruct",
    DomainType.JOBS_AND_EDUCATION: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    DomainType.LAW_AND_GOVERNMENT: "microsoft/Phi-4-mini-reasoning",
    DomainType.SENSITIVE_SUBJECTS: "meta-llama/Meta-Llama-3-8B-Instruct",
    # Default for remaining domains
    DomainType.ADULT: "meta-llama/Meta-Llama-3-8B-Instruct",
    DomainType.ARTS_AND_ENTERTAINMENT: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.AUTOS_AND_VEHICLES: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.BEAUTY_AND_FITNESS: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.BOOKS_AND_LITERATURE: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.FOOD_AND_DRINK: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.GAMES: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.HOBBIES_AND_LEISURE: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.HOME_AND_GARDEN: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.ONLINE_COMMUNITIES: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.PEOPLE_AND_SOCIETY: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.PETS_AND_ANIMALS: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.REFERENCE: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    DomainType.SHOPPING: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.SPORTS: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.TRAVEL_AND_TRANSPORTATION: "Qwen/Qwen2.5-7B-Instruct",
}

# --- UNIFIED DOMAIN-TASK MODEL MAPPING ---
# Structure: domains[domain][task_type] -> list[TaskModelEntry]
# Used for standard LLM protocol routing
domains: dict[DomainType, dict[TaskType, list[TaskModelEntry]]] = {
    # Business and Industrial
    DomainType.BUSINESS_AND_INDUSTRIAL: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Computers and Electronics
    DomainType.COMPUTERS_AND_ELECTRONICS: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Finance
    DomainType.FINANCE: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Health
    DomainType.HEALTH: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Internet and Telecom
    DomainType.INTERNET_AND_TELECOM: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Jobs and Education
    DomainType.JOBS_AND_EDUCATION: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Law and Government
    DomainType.LAW_AND_GOVERNMENT: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
    },
    # News
    DomainType.NEWS: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
    },
    # Real Estate
    DomainType.REAL_ESTATE: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Science
    DomainType.SCIENCE: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
    },
    # Sensitive Subjects
    DomainType.SENSITIVE_SUBJECTS: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Other Domains (consolidated)
    DomainType.OTHERDOMAINS: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
}


# --- MINION MODELS (HuggingFace Specialists) ---
# ONLY models with confirmed HuggingFace Inference API support
# ✅ API Ready: microsoft/deberta-v3-base, facebook/bart-large-cnn, ProsusAI/finbert,
#               emilyalsentzer/Bio_ClinicalBERT, nlpaueb/legal-bert-base-uncased, microsoft/codebert-base
minion_domains: dict[DomainType, dict[TaskType, str]] = {
    # Business and Industrial
    DomainType.BUSINESS_AND_INDUSTRIAL: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - best available for generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - general purpose
    },
    # Computers and Electronics
    DomainType.COMPUTERS_AND_ELECTRONICS: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - text generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/codebert-base",
    },
    # Finance
    DomainType.FINANCE: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "ProsusAI/finbert",
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - text generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "ProsusAI/finbert",
        TaskType.CLOSED_QA: "ProsusAI/finbert",
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "ProsusAI/finbert",
        TaskType.OTHER: "ProsusAI/finbert",
    },
    # Health
    DomainType.HEALTH: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "emilyalsentzer/Bio_ClinicalBERT",
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",  # ✅ API Ready - LED not available
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - text generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses  # Fixed: Bio_ClinicalBERT can't chat
        TaskType.CLASSIFICATION: "emilyalsentzer/Bio_ClinicalBERT",
        TaskType.CLOSED_QA: "emilyalsentzer/Bio_ClinicalBERT",
        TaskType.REWRITE: "facebook/bart-large-cnn",  # ✅ API Ready - LED not available
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "emilyalsentzer/Bio_ClinicalBERT",
        TaskType.OTHER: "emilyalsentzer/Bio_ClinicalBERT",
    },
    # Internet and Telecom
    DomainType.INTERNET_AND_TELECOM: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - text generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - general purpose
    },
    # Jobs and Education
    DomainType.JOBS_AND_EDUCATION: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - best available for generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - general purpose
    },
    # Law and Government
    DomainType.LAW_AND_GOVERNMENT: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "nlpaueb/legal-bert-base-uncased",
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",  # ✅ API Ready - LED not available
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - text generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses  # Fixed: Legal-BERT can't chat
        TaskType.CLASSIFICATION: "nlpaueb/legal-bert-base-uncased",
        TaskType.CLOSED_QA: "nlpaueb/legal-bert-base-uncased",
        TaskType.REWRITE: "facebook/bart-large-cnn",  # ✅ API Ready - LED not available
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "nlpaueb/legal-bert-base-uncased",
        TaskType.OTHER: "nlpaueb/legal-bert-base-uncased",
    },
    # News
    DomainType.NEWS: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - best available for generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - general purpose
    },
    # Real Estate
    DomainType.REAL_ESTATE: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - best available for generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - general purpose
    },
    # Science
    DomainType.SCIENCE: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - SciBERT not available
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",  # ✅ API Ready - LED not available
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - text generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",  # ✅ API Ready - SciBERT not available
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - SciBERT not available
        TaskType.REWRITE: "facebook/bart-large-cnn",  # ✅ API Ready - LED not available
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",  # ✅ API Ready - SciBERT not available
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - SciBERT not available
    },
    # Sensitive Subjects
    DomainType.SENSITIVE_SUBJECTS: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - best available for generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - general purpose
    },
    # Other Domains (consolidated)
    DomainType.OTHERDOMAINS: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - best available for generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - general purpose
    },
}

# --- UNIFIED DOMAIN-TASK MODEL MAPPING ---
# Structure: domains[domain][task_type] -> list[TaskModelEntry]
# Used for standard LLM protocol routing
domains: dict[DomainType, dict[TaskType, list[TaskModelEntry]]] = {
    # Business and Industrial
    DomainType.BUSINESS_AND_INDUSTRIAL: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Computers and Electronics
    DomainType.COMPUTERS_AND_ELECTRONICS: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Finance
    DomainType.FINANCE: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Health
    DomainType.HEALTH: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Internet and Telecom
    DomainType.INTERNET_AND_TELECOM: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Jobs and Education
    DomainType.JOBS_AND_EDUCATION: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Law and Government
    DomainType.LAW_AND_GOVERNMENT: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
        ],
    },
    # News
    DomainType.NEWS: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
    },
    # Real Estate
    DomainType.REAL_ESTATE: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Science
    DomainType.SCIENCE: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
    },
    # Sensitive Subjects
    DomainType.SENSITIVE_SUBJECTS: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
    # Other Domains (consolidated)
    DomainType.OTHERDOMAINS: {
        TaskType.CODE_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
        TaskType.OPEN_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
        ],
        TaskType.SUMMARIZATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        TaskType.TEXT_GENERATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CHATBOT: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLASSIFICATION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.CLOSED_QA: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.REWRITE: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
        ],
        TaskType.BRAINSTORMING: [
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.EXTRACTION: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        TaskType.OTHER: [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
        ],
    },
}


# --- MINION MODELS (HuggingFace Specialists) ---
# ONLY models with confirmed HuggingFace Inference API support
# ✅ API Ready: microsoft/deberta-v3-base, facebook/bart-large-cnn, ProsusAI/finbert,
#               emilyalsentzer/Bio_ClinicalBERT, nlpaueb/legal-bert-base-uncased, microsoft/codebert-base
minion_domains: dict[DomainType, dict[TaskType, str]] = {
    # Business and Industrial
    DomainType.BUSINESS_AND_INDUSTRIAL: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - best available for generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - general purpose
    },
    # Computers and Electronics
    DomainType.COMPUTERS_AND_ELECTRONICS: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - text generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/codebert-base",
    },
    # Finance
    DomainType.FINANCE: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "ProsusAI/finbert",
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - text generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "ProsusAI/finbert",
        TaskType.CLOSED_QA: "ProsusAI/finbert",
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "ProsusAI/finbert",
        TaskType.OTHER: "ProsusAI/finbert",
    },
    # Health
    DomainType.HEALTH: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "emilyalsentzer/Bio_ClinicalBERT",
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",  # ✅ API Ready - LED not available
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - text generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses  # Fixed: Bio_ClinicalBERT can't chat
        TaskType.CLASSIFICATION: "emilyalsentzer/Bio_ClinicalBERT",
        TaskType.CLOSED_QA: "emilyalsentzer/Bio_ClinicalBERT",
        TaskType.REWRITE: "facebook/bart-large-cnn",  # ✅ API Ready - LED not available
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "emilyalsentzer/Bio_ClinicalBERT",
        TaskType.OTHER: "emilyalsentzer/Bio_ClinicalBERT",
    },
    # Internet and Telecom
    DomainType.INTERNET_AND_TELECOM: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - text generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - general purpose
    },
    # Jobs and Education
    DomainType.JOBS_AND_EDUCATION: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - best available for generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - general purpose
    },
    # Law and Government
    DomainType.LAW_AND_GOVERNMENT: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "nlpaueb/legal-bert-base-uncased",
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",  # ✅ API Ready - LED not available
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - text generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses  # Fixed: Legal-BERT can't chat
        TaskType.CLASSIFICATION: "nlpaueb/legal-bert-base-uncased",
        TaskType.CLOSED_QA: "nlpaueb/legal-bert-base-uncased",
        TaskType.REWRITE: "facebook/bart-large-cnn",  # ✅ API Ready - LED not available
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "nlpaueb/legal-bert-base-uncased",
        TaskType.OTHER: "nlpaueb/legal-bert-base-uncased",
    },
    # News
    DomainType.NEWS: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - best available for generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - general purpose
    },
    # Real Estate
    DomainType.REAL_ESTATE: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - best available for generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - general purpose
    },
    # Science
    DomainType.SCIENCE: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - SciBERT not available
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",  # ✅ API Ready - LED not available
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - text generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",  # ✅ API Ready - SciBERT not available
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - SciBERT not available
        TaskType.REWRITE: "facebook/bart-large-cnn",  # ✅ API Ready - LED not available
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",  # ✅ API Ready - SciBERT not available
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - SciBERT not available
    },
    # Sensitive Subjects
    DomainType.SENSITIVE_SUBJECTS: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - best available for generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - general purpose
    },
    # Other Domains (consolidated)
    DomainType.OTHERDOMAINS: {
        TaskType.CODE_GENERATION: "microsoft/codebert-base",  # ✅ API Ready - code feature extraction
        TaskType.OPEN_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
        TaskType.TEXT_GENERATION: "facebook/bart-large-cnn",  # ✅ API Ready - best available for generation
        TaskType.CHATBOT: "microsoft/deberta-v3-base",  # ✅ API Ready - classification-based responses
        TaskType.CLASSIFICATION: "microsoft/deberta-v3-base",
        TaskType.CLOSED_QA: "microsoft/deberta-v3-base",  # ✅ API Ready - classification for QA
        TaskType.REWRITE: "facebook/bart-large-cnn",
        TaskType.BRAINSTORMING: "facebook/bart-large-cnn",  # ✅ API Ready - creative text generation
        TaskType.EXTRACTION: "microsoft/deberta-v3-base",
        TaskType.OTHER: "microsoft/deberta-v3-base",  # ✅ API Ready - general purpose
    },
}
