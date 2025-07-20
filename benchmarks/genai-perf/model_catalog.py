"""
Model catalog for benchmark testing - Updated with actual supported models
"""

from enum import Enum


class TaskType(Enum):
    """TaskType enum for benchmark testing"""

    OPEN_QA = "Open QA"
    CODE_GENERATION = "Code Generation"
    SUMMARIZATION = "Summarization"
    TEXT_GENERATION = "Text Generation"
    CHATBOT = "Chatbot"
    CLASSIFICATION = "Classification"
    CLOSED_QA = "Closed QA"
    REWRITE = "Rewrite"
    BRAINSTORMING = "Brainstorming"
    EXTRACTION = "Extraction"
    OTHER = "Other"


class DomainType(Enum):
    """Domain types for minion model selection"""

    BUSINESS_AND_INDUSTRIAL = "Business_and_Industrial"
    HEALTH = "Health"
    NEWS = "News"
    OTHERDOMAINS = "Otherdomains"
    REAL_ESTATE = "Real_Estate"
    COMPUTERS_AND_ELECTRONICS = "Computers_and_Electronics"
    INTERNET_AND_TELECOM = "Internet_and_Telecom"
    FINANCE = "Finance"
    SCIENCE = "Science"
    JOBS_AND_EDUCATION = "Jobs_and_Education"
    LAW_AND_GOVERNMENT = "Law_and_Government"
    SENSITIVE_SUBJECTS = "Sensitive_Subjects"


# --- Standard Models: Task-based selection only (no domains) ---
standard_task_model_mappings: dict[TaskType, str] = {
    TaskType.OPEN_QA: "Qwen/Qwen2.5-14B-Instruct",
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

# --- Minion Models: Domain-based selection ---
# Maps domains to specialized models based on domain expertise
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
}

# All supported models in the adaptive minion service
supported_models = [
    "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "codellama/CodeLlama-7b-Instruct-hf",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-4-mini-reasoning",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]
