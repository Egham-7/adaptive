from enum import Enum


class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "gemini"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    GROK = "grok"


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


class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ProtocolType(str, Enum):
    STANDARD_LLM = "standard_llm"
    MINION = "minion"
    MINIONS_PROTOCOL = "minions_protocol"
