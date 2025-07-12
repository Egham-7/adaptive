"""
Model catalog for minion selection - TaskType mappings using tiny Groq HF models
"""

from enum import Enum

class TaskType(Enum):
    OPEN_QA = "open_qa"
    CODE_GENERATION = "code_generation"
    SUMMARIZATION = "summarization"
    TEXT_GENERATION = "text_generation"
    CHATBOT = "chatbot"
    CLASSIFICATION = "classification"
    CLOSED_QA = "closed_qa"
    REWRITE = "rewrite"
    BRAINSTORMING = "brainstorming"
    EXTRACTION = "extraction"
    OTHER = "other"

# --- Minion Task Model Mappings (HuggingFace Models) ---
# This maps each TaskType to a SINGLE designated HuggingFace specialist model,
# each optimized for specific task types and available via HuggingFace Inference API
minion_task_model_mappings: dict[TaskType, str] = {
    TaskType.OPEN_QA: "llama-3.1-8b-instant",
    TaskType.CODE_GENERATION: "meta-llama/llama-4-scout-17b-16e-instruct",
    TaskType.SUMMARIZATION: "gemma2-9b-it",
    TaskType.TEXT_GENERATION: "meta-llama/llama-4-maverick-17b-128e-instruct",
    TaskType.CHATBOT: "gemma2-9b-it",
    TaskType.CLASSIFICATION: "meta-llama/llama-prompt-guard-2-86m",
    TaskType.CLOSED_QA: "llama-3.1-8b-instant",
    TaskType.REWRITE: "gemma2-9b-it",
    TaskType.BRAINSTORMING: "meta-llama/llama-4-maverick-17b-128e-instruct",
    TaskType.EXTRACTION: "meta-llama/llama-prompt-guard-2-86m",
    TaskType.OTHER: "llama-3.1-8b-instant",
}