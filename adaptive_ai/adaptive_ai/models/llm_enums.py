from enum import Enum


class TaskType(str, Enum):
    OPEN_QA = "Open QA"
    CLOSED_QA = "Closed QA"
    SUMMARIZATION = "Summarization"
    TEXT_GENERATION = "Text Generation"
    CODE_GENERATION = "Code Generation"
    CHATBOT = "Chatbot"
    CLASSIFICATION = "Classification"
    REWRITE = "Rewrite"
    BRAINSTORMING = "Brainstorming"
    EXTRACTION = "Extraction"
    OTHER = "Other"
