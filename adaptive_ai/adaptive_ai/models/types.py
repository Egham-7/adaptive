from typing import TypedDict, Literal, List

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

# Provider Types
ProviderType = Literal["GROQ", "OpenAI", "DEEPSEEK", "Anthropic", "Google"]
ModelProvider = ProviderType  # Alias for backward compatibility

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

# Difficulty/Complexity Levels
DifficultyLevel = Literal["easy", "medium", "hard"]
ComplexityLevel = DifficultyLevel  # Alias for backward compatibility


class TaskTypeParametersType(TypedDict):
    """Parameters for model configuration by task type"""

    temperature: float
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    max_completion_tokens: int
    n: int


class ModelCapability(TypedDict):
    """Model capability definition"""

    description: str
    provider: ProviderType
    cost_per_1k_tokens: float
    max_tokens: int
    supports_streaming: bool
    supports_function_calling: bool
    supports_vision: bool


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


class ConfigurationError(Exception):
    """Custom exception for configuration errors"""

    pass


class ProviderError(Exception):
    """Custom exception for provider-related errors"""

    pass


# =============================================================================
# CONSTANTS
# =============================================================================

VALID_TASK_TYPES: List[TaskType] = [
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

VALID_PROVIDERS: List[ProviderType] = [
    "GROQ",
    "OpenAI",
    "DEEPSEEK",
    "Anthropic",
    "Google",
]

VALID_DIFFICULTY_LEVELS: List[DifficultyLevel] = ["easy", "medium", "hard"]
