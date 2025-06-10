from typing import TypedDict, Literal, List

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

# Provider Types
ProviderType = Literal["GROQ", "OpenAI", "DEEPSEEK", "Anthropic", "Google"]

# Task Types
TaskType = Literal[
    "Open QA",
    "Closed QA",
    "Summarization",
    "Text Generation",
    "Code Generation",
    "Chatbot",
    "Classification",
    "Rewrite",
    "Brainstorming",
    "Extraction",
    "Other",
]

# Difficulty Levels
DifficultyLevel = Literal["easy", "medium", "hard"]


class TaskTypeParametersType(TypedDict):
    """Parameters for model configuration by task type"""

    Temperature: float
    TopP: float
    PresencePenalty: float
    FrequencyPenalty: float
    MaxCompletionTokens: int
    N: int


class ModelCapability(TypedDict):
    """Model capability definition"""

    description: str
    provider: ProviderType


class ModelInfo(TypedDict):
    """Model information with scoring"""

    model_name: str
    provider: str
    match_score: float


class PromptScores(TypedDict):
    """Type definition for prompt analysis scores"""

    creativity_scope: List[float]
    reasoning: List[float]
    constraint_ct: List[float]
    contextual_knowledge: List[float]
    domain_knowledge: List[float]


class DifficultyThresholds(TypedDict):
    """Type definition for difficulty thresholds"""

    easy: float
    medium: float
    hard: float


class TaskDifficultyConfig(TypedDict):
    """Configuration for a specific difficulty level"""

    model: str
    complexity_threshold: float


class TaskModelMapping(TypedDict):
    """Model mapping for different difficulty levels of a task"""

    easy: TaskDifficultyConfig
    medium: TaskDifficultyConfig
    hard: TaskDifficultyConfig


class ModelParameters(TypedDict):
    """Type definition for model parameters"""

    task_type: str
    prompt_scores: PromptScores


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ModelSelectionError(Exception):
    """Custom exception for model selection errors"""

    pass
