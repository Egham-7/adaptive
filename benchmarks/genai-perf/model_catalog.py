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
    TaskType.OPEN_QA: "llama-3.2-3b-preview",
    TaskType.CODE_GENERATION: "llama-3-groq-8b-tool-use",
    TaskType.SUMMARIZATION: "llama-3.2-1b-preview",
    TaskType.TEXT_GENERATION: "llama-3.1-8b-instant",
    TaskType.CHATBOT: "gemma2-9b-it",
    TaskType.CLASSIFICATION: "llama-3.2-1b-preview",
    TaskType.CLOSED_QA: "llama-3.2-3b-preview",
    TaskType.REWRITE: "llama-3.1-8b-instant",
    TaskType.BRAINSTORMING: "llama-3.3-70b-versatile",
    TaskType.EXTRACTION: "llama-3.2-1b-preview",
    TaskType.OTHER: "llama-3.2-1b-preview",
}