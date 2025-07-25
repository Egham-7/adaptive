from enum import Enum


class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "gemini"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    GROK = "grok"
    HUGGINGFACE = "huggingface"


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


class DomainType(str, Enum):
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


class ProtocolType(str, Enum):
    STANDARD_LLM = "standard_llm"
    MINION = "minion"
    MINIONS_PROTOCOL = "minions_protocol"
