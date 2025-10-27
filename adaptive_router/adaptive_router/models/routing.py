"""Internal routing models for router implementation.

This module contains models used internally by the routing engine for scoring
models, tracking routing decisions, and managing model feature vectors.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ModelFeatureVector(BaseModel):
    """Feature vector for a single model used in routing decisions.

    Combines per-cluster error rates with cost information for scoring models.

    Attributes:
        error_rates: List of K error rates (one per cluster)
        cost_per_1m_tokens: Model cost per million tokens
    """

    error_rates: List[float] = Field(..., description="K error rates per cluster")
    cost_per_1m_tokens: float = Field(..., gt=0, description="Cost per 1M tokens")


class ModelFeatures(BaseModel):
    """Feature vector for a model (error rates + cost + metadata).

    Extended version of ModelFeatureVector with additional metadata for evaluation.

    Attributes:
        model_id: Unique model identifier
        model_name: Human-readable model name
        error_rates: K error rates (one per cluster)
        cost_per_1m_tokens: Cost per million tokens
        accuracy: Overall accuracy across all clusters
        avg_response_time_ms: Average response time in milliseconds
        total_questions_evaluated: Total number of questions evaluated
    """

    model_id: str
    model_name: str
    error_rates: List[float]  # K error rates (one per cluster)
    cost_per_1m_tokens: float
    accuracy: float
    avg_response_time_ms: float
    total_questions_evaluated: int


class RoutingDecision(BaseModel):
    """Result of routing a question to a model.

    Contains the selected model, routing scores, predicted performance,
    and reasoning for the decision.

    Attributes:
        selected_model_id: ID of the selected model
        selected_model_name: Human-readable name of selected model
        routing_score: Routing score (lower is better)
        predicted_accuracy: Predicted accuracy for this question
        estimated_cost: Estimated cost in USD for ~2K tokens
        cluster_id: Cluster ID assigned to the question
        cluster_confidence: Confidence in cluster assignment (0.0-1.0)
        lambda_param: Lambda parameter used for routing decision
        reasoning: Human-readable explanation of routing decision
        alternatives: Alternative models and their scores
        routing_time_ms: Time taken for routing decision in milliseconds
    """

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
