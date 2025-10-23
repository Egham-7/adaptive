"""Pydantic models for type-safe data structures."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MCQAnswer(BaseModel):
    """Structured output for multiple choice question answers."""

    answer: str = Field(
        ..., description="Selected answer: A, B, C, D, E, F, G, H, I, or J"
    )
    reasoning: str = Field(..., description="Brief explanation for the choice")
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the answer (0.0-1.0)",
    )


class CodeQuestion(BaseModel):
    """A coding question from the dataset."""

    question_id: str
    question: str
    choices: List[str]
    answer: str  # A, B, C, D, E, F, G, H, I, or J
    category: Optional[str] = None
    difficulty: Optional[str] = None

    def format_for_llm(self) -> str:
        """Format question as MCQ prompt for LLM."""
        choices_text = "\n".join(
            [f"{chr(65+i)}) {choice}" for i, choice in enumerate(self.choices)]
        )

        # Generate valid answer options based on number of choices
        max_choice = chr(65 + len(self.choices) - 1)
        answer_range = f"A-{max_choice}" if len(self.choices) > 1 else "A"

        return f"""You are taking a multiple choice coding test. Read the question carefully and select the best answer.

Question: {self.question}

{choices_text}

Instructions:
- Select exactly one answer ({answer_range})
- Provide brief reasoning for your choice
- Be concise and accurate"""


class EvaluationResult(BaseModel):
    """Result of evaluating a model on a single question."""

    model_id: str
    question_id: str
    cluster_id: int
    predicted_answer: str
    correct_answer: str
    is_correct: bool
    response_time_ms: float
    confidence: float = 0.5
    reasoning: Optional[str] = None


class ClusterMetadata(BaseModel):
    """Metadata for clustering results."""

    n_clusters: int
    n_questions: int
    clustering_method: str
    embedding_model: str
    timestamp: str
    silhouette_score: Optional[float] = None


class ModelFeatures(BaseModel):
    """Feature vector for a model (error rates + cost)."""

    model_id: str
    model_name: str
    error_rates: List[float]  # K error rates (one per cluster)
    cost_per_1m_tokens: float
    accuracy: float
    avg_response_time_ms: float
    total_questions_evaluated: int


class RoutingDecision(BaseModel):
    """Result of routing a question to a model."""

    selected_model_id: str
    selected_model_name: str
    routing_score: float
    predicted_accuracy: float
    estimated_cost: float
    cluster_id: int
    cluster_confidence: float
    lambda_param: float
    reasoning: str
    alternatives: List[Dict[str, Any]]  # Other models and their scores
    routing_time_ms: float


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    id: str
    name: str
    provider: str
    cost_per_1m_tokens: float
    description: str


class QuestionRoutingRequest(BaseModel):
    """Request model for question routing endpoint.

    Follows the same pattern as adaptive_router's ModelSelectionRequest.
    """

    question: str = Field(
        ..., min_length=1, description="Question text to route to optimal model"
    )
    cost_preference: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Cost-quality trade-off preference (0.0=cheap, 1.0=quality). Uses default if not specified.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Implement a binary search tree in Python with insert and delete operations",
                "cost_preference": 0.5,
            }
        }


class QuestionRoutingResponse(BaseModel):
    """Response model for question routing endpoint.

    Follows the same pattern as adaptive_router's ModelSelectionResponse.
    """

    selected_model_id: str = Field(..., description="ID of the selected model")
    selected_model_name: str = Field(
        ..., description="Human-readable name of the selected model"
    )
    routing_score: float = Field(..., description="Routing score (lower is better)")
    predicted_accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted accuracy for this question on selected model",
    )
    estimated_cost: float = Field(
        ..., description="Estimated cost in USD for ~2K tokens"
    )
    cluster_id: int = Field(..., description="Cluster ID assigned to the question")
    cluster_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in cluster assignment"
    )
    lambda_param: float = Field(
        ..., description="Lambda parameter used for routing decision"
    )
    reasoning: str = Field(
        ..., description="Human-readable explanation of routing decision"
    )
    alternatives: List[Dict[str, Any]] = Field(
        default_factory=list, description="Alternative models and their scores"
    )
    routing_time_ms: float = Field(
        ..., description="Time taken for routing decision in milliseconds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "selected_model_id": "gpt-5-mini",
                "selected_model_name": "GPT-5 Mini",
                "routing_score": 0.125,
                "predicted_accuracy": 0.95,
                "estimated_cost": 0.00225,
                "cluster_id": 7,
                "cluster_confidence": 0.87,
                "lambda_param": 1.55,
                "reasoning": "Question assigned to cluster 7; Balanced cost-accuracy routing (λ=1.55); Excellent predicted accuracy (95%)",
                "alternatives": [
                    {
                        "model_id": "gpt-5-codex",
                        "model_name": "GPT-5 Codex",
                        "score": 0.15,
                        "accuracy": 0.95,
                        "cost": 2.25,
                    }
                ],
                "routing_time_ms": 12.5,
            }
        }


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(default="healthy", description="Service health status")
    version: str = Field(default="1.0.0", description="Service version")
    models_loaded: int = Field(..., description="Number of models loaded")
    clusters_loaded: int = Field(..., description="Number of clusters loaded")
