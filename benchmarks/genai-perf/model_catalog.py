"""
Model catalog for minion selection - TaskType mappings using tiny Groq HF models
"""

from enum import Enum
from typing import Dict

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
minion_task_model_mappings: Dict[TaskType, str] = {
    TaskType.OPEN_QA: "deepset/roberta-base-squad2",
    TaskType.CODE_GENERATION: "microsoft/codebert-base",
    TaskType.SUMMARIZATION: "facebook/bart-large-cnn",
    TaskType.TEXT_GENERATION: "HuggingFaceTB/SmolLM3-3B",
    TaskType.CHATBOT: "HuggingFaceTB/SmolLM3-3B",
    TaskType.CLASSIFICATION: "cardiffnlp/twitter-roberta-base-sentiment",
    TaskType.CLOSED_QA: "google/tapas-base-finetuned-wtq",
    TaskType.REWRITE: "facebook/bart-large",
    TaskType.BRAINSTORMING: "HuggingFaceTB/SmolLM3-3B",
    TaskType.EXTRACTION: "dslim/bert-base-NER",
    TaskType.OTHER: "HuggingFaceTB/SmolLM3-3B",
}