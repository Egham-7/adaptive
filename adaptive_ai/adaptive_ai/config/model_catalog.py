# config/model_catalog.py

from adaptive_ai.models import (
    ModelCapability,
    ProviderType,
    TaskModelEntry,
    TaskModelMapping,
    TaskType,
)

# --- ACTIVE PROVIDERS CONFIGURATION ---
# Only these providers will be used for model selection
# Other providers remain in the catalog but are inactive
ACTIVE_PROVIDERS = {
    ProviderType.OPENAI,
    ProviderType.GROQ,  # Fast inference provider
    ProviderType.GROK,  # X.AI's models (grok-3, grok-3-mini)
    ProviderType.DEEPSEEK,
    ProviderType.ADAPTIVE,  # Adaptive minion models
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
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
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
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
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
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
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
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(
                provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"
            ),
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

# --- Minion Task Model Mappings (HuggingFace Models) ---
# This maps each TaskType to a SINGLE designated HuggingFace specialist model,
# each optimized for specific task types and available via HuggingFace Inference API

minion_task_model_mappings: dict[TaskType, str] = {
    TaskType.OPEN_QA: "Qwen/Qwen2.5-14B-Instruct",  # BUSINESS_AND_INDUSTRIAL/HEALTH
    TaskType.CODE_GENERATION: "codellama/CodeLlama-7b-Instruct-hf",  # COMPUTERS_AND_ELECTRONICS/INTERNET_AND_TELECOM
    TaskType.SUMMARIZATION: "Qwen/Qwen2.5-7B-Instruct",  # NEWS/OTHERDOMAINS/REAL_ESTATE
    TaskType.TEXT_GENERATION: "Qwen/Qwen2.5-14B-Instruct",  # BUSINESS_AND_INDUSTRIAL/HEALTH
    TaskType.CHATBOT: "Qwen/Qwen2.5-7B-Instruct",  # NEWS/OTHERDOMAINS/REAL_ESTATE
    TaskType.CLASSIFICATION: "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # JOBS_AND_EDUCATION
    TaskType.CLOSED_QA: "microsoft/Phi-4-mini-reasoning",  # LAW_AND_GOVERNMENT
    TaskType.REWRITE: "Qwen/Qwen2.5-7B-Instruct",  # NEWS/OTHERDOMAINS/REAL_ESTATE
    TaskType.BRAINSTORMING: "Qwen/Qwen2.5-Math-7B-Instruct",  # FINANCE/SCIENCE
    TaskType.EXTRACTION: "meta-llama/Meta-Llama-3-8B-Instruct",  # SENSITIVE_SUBJECTS
    TaskType.OTHER: "Qwen/Qwen2.5-7B-Instruct",  # NEWS/OTHERDOMAINS/REAL_ESTATE
}

# --- Domain-specific Minion Model Mappings ---
# Maps each domain to specific models for each task type
from adaptive_ai.models.llm_classification_models import DomainType

minion_domains: dict[DomainType, dict[TaskType, str]] = {
    # Business and Industrial - Qwen/Qwen2.5-14B-Instruct
    DomainType.BUSINESS_AND_INDUSTRIAL: {
        TaskType.CODE_GENERATION: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.OPEN_QA: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.SUMMARIZATION: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.TEXT_GENERATION: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.CHATBOT: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.CLASSIFICATION: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.CLOSED_QA: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.REWRITE: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.BRAINSTORMING: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.EXTRACTION: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.OTHER: "Qwen/Qwen2.5-14B-Instruct",
    },
    # Computers and Electronics - codellama/CodeLlama-7b-Instruct-hf
    DomainType.COMPUTERS_AND_ELECTRONICS: {
        TaskType.CODE_GENERATION: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.OPEN_QA: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.SUMMARIZATION: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.TEXT_GENERATION: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.CHATBOT: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.CLASSIFICATION: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.CLOSED_QA: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.REWRITE: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.BRAINSTORMING: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.EXTRACTION: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.OTHER: "codellama/CodeLlama-7b-Instruct-hf",
    },
    # Finance - Qwen/Qwen2.5-Math-7B-Instruct
    DomainType.FINANCE: {
        TaskType.CODE_GENERATION: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.OPEN_QA: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.SUMMARIZATION: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.TEXT_GENERATION: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.CHATBOT: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.CLASSIFICATION: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.CLOSED_QA: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.REWRITE: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.BRAINSTORMING: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.EXTRACTION: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.OTHER: "Qwen/Qwen2.5-Math-7B-Instruct",
    },
    # Health - Qwen/Qwen2.5-14B-Instruct
    DomainType.HEALTH: {
        TaskType.CODE_GENERATION: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.OPEN_QA: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.SUMMARIZATION: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.TEXT_GENERATION: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.CHATBOT: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.CLASSIFICATION: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.CLOSED_QA: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.REWRITE: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.BRAINSTORMING: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.EXTRACTION: "Qwen/Qwen2.5-14B-Instruct",
        TaskType.OTHER: "Qwen/Qwen2.5-14B-Instruct",
    },
    # Internet and Telecom - codellama/CodeLlama-7b-Instruct-hf
    DomainType.INTERNET_AND_TELECOM: {
        TaskType.CODE_GENERATION: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.OPEN_QA: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.SUMMARIZATION: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.TEXT_GENERATION: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.CHATBOT: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.CLASSIFICATION: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.CLOSED_QA: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.REWRITE: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.BRAINSTORMING: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.EXTRACTION: "codellama/CodeLlama-7b-Instruct-hf",
        TaskType.OTHER: "codellama/CodeLlama-7b-Instruct-hf",
    },
    # Jobs and Education - HuggingFaceTB/SmolLM2-1.7B-Instruct
    DomainType.JOBS_AND_EDUCATION: {
        TaskType.CODE_GENERATION: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        TaskType.OPEN_QA: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        TaskType.SUMMARIZATION: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        TaskType.TEXT_GENERATION: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        TaskType.CHATBOT: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        TaskType.CLASSIFICATION: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        TaskType.CLOSED_QA: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        TaskType.REWRITE: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        TaskType.BRAINSTORMING: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        TaskType.EXTRACTION: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        TaskType.OTHER: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    },
    # Law and Government - microsoft/Phi-4-mini-reasoning
    DomainType.LAW_AND_GOVERNMENT: {
        TaskType.CODE_GENERATION: "microsoft/Phi-4-mini-reasoning",
        TaskType.OPEN_QA: "microsoft/Phi-4-mini-reasoning",
        TaskType.SUMMARIZATION: "microsoft/Phi-4-mini-reasoning",
        TaskType.TEXT_GENERATION: "microsoft/Phi-4-mini-reasoning",
        TaskType.CHATBOT: "microsoft/Phi-4-mini-reasoning",
        TaskType.CLASSIFICATION: "microsoft/Phi-4-mini-reasoning",
        TaskType.CLOSED_QA: "microsoft/Phi-4-mini-reasoning",
        TaskType.REWRITE: "microsoft/Phi-4-mini-reasoning",
        TaskType.BRAINSTORMING: "microsoft/Phi-4-mini-reasoning",
        TaskType.EXTRACTION: "microsoft/Phi-4-mini-reasoning",
        TaskType.OTHER: "microsoft/Phi-4-mini-reasoning",
    },
    # News - Qwen/Qwen2.5-7B-Instruct
    DomainType.NEWS: {
        TaskType.CODE_GENERATION: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.OPEN_QA: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.SUMMARIZATION: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.TEXT_GENERATION: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.CHATBOT: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.CLASSIFICATION: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.CLOSED_QA: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.REWRITE: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.BRAINSTORMING: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.EXTRACTION: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.OTHER: "Qwen/Qwen2.5-7B-Instruct",
    },
    # Real Estate - Qwen/Qwen2.5-7B-Instruct
    DomainType.REAL_ESTATE: {
        TaskType.CODE_GENERATION: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.OPEN_QA: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.SUMMARIZATION: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.TEXT_GENERATION: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.CHATBOT: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.CLASSIFICATION: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.CLOSED_QA: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.REWRITE: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.BRAINSTORMING: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.EXTRACTION: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.OTHER: "Qwen/Qwen2.5-7B-Instruct",
    },
    # Science - Qwen/Qwen2.5-Math-7B-Instruct
    DomainType.SCIENCE: {
        TaskType.CODE_GENERATION: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.OPEN_QA: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.SUMMARIZATION: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.TEXT_GENERATION: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.CHATBOT: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.CLASSIFICATION: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.CLOSED_QA: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.REWRITE: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.BRAINSTORMING: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.EXTRACTION: "Qwen/Qwen2.5-Math-7B-Instruct",
        TaskType.OTHER: "Qwen/Qwen2.5-Math-7B-Instruct",
    },
    # Sensitive Subjects - meta-llama/Meta-Llama-3-8B-Instruct
    DomainType.SENSITIVE_SUBJECTS: {
        TaskType.CODE_GENERATION: "meta-llama/Meta-Llama-3-8B-Instruct",
        TaskType.OPEN_QA: "meta-llama/Meta-Llama-3-8B-Instruct",
        TaskType.SUMMARIZATION: "meta-llama/Meta-Llama-3-8B-Instruct",
        TaskType.TEXT_GENERATION: "meta-llama/Meta-Llama-3-8B-Instruct",
        TaskType.CHATBOT: "meta-llama/Meta-Llama-3-8B-Instruct",
        TaskType.CLASSIFICATION: "meta-llama/Meta-Llama-3-8B-Instruct",
        TaskType.CLOSED_QA: "meta-llama/Meta-Llama-3-8B-Instruct",
        TaskType.REWRITE: "meta-llama/Meta-Llama-3-8B-Instruct",
        TaskType.BRAINSTORMING: "meta-llama/Meta-Llama-3-8B-Instruct",
        TaskType.EXTRACTION: "meta-llama/Meta-Llama-3-8B-Instruct",
        TaskType.OTHER: "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    # Other Domains - Qwen/Qwen2.5-7B-Instruct
    DomainType.OTHERDOMAINS: {
        TaskType.CODE_GENERATION: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.OPEN_QA: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.SUMMARIZATION: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.TEXT_GENERATION: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.CHATBOT: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.CLASSIFICATION: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.CLOSED_QA: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.REWRITE: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.BRAINSTORMING: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.EXTRACTION: "Qwen/Qwen2.5-7B-Instruct",
        TaskType.OTHER: "Qwen/Qwen2.5-7B-Instruct",
    },
}
