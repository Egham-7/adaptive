"""Intelligent routing based on cluster-specific error rates."""

import logging
import time
from typing import Any, Dict, List

from adaptive_router.services.cluster_engine import ClusterEngine
from adaptive_router.services.routing_schemas import ModelConfig, RoutingDecision

logger = logging.getLogger(__name__)

# Cost estimation constants
ESTIMATED_TOKEN_COUNT = 2000  # Assumed average token count for cost estimation
TOKENS_PER_MILLION = 1_000_000  # Tokens per million for pricing calculations
EPSILON = 1e-10  # Small value for floating point comparisons to avoid division issues


class Router:
    """Intelligent router using cluster-based error rates."""

    def __init__(
        self,
        cluster_engine: ClusterEngine,
        model_features: Dict[str, Any],
        models: List[ModelConfig],
        lambda_min: float = 0.0,
        lambda_max: float = 1.0,
        default_cost_preference: float = 0.5,
    ) -> None:
        """Initialize Router.

        Args:
            cluster_engine: Fitted ClusterEngine for question assignment
            model_features: Model feature vectors (error rates + cost)
            models: List of available models
            lambda_min: Minimum lambda parameter
            lambda_max: Maximum lambda parameter
            default_cost_preference: Default cost-quality trade-off (0.0-1.0)
        """
        self.cluster_engine = cluster_engine
        self.model_features = model_features
        self.models = {m.id: m for m in models}

        # Lambda parameter range [0.0, 1.0]
        # 0.0 = no cost penalty (pure quality)
        # 1.0 = equal weight to error rate and cost (both normalized)
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.default_cost_preference = default_cost_preference

        # Pre-compute normalized costs
        all_costs = [f["cost_per_1m_tokens"] for f in model_features.values()]
        self.min_cost = min(all_costs)
        self.max_cost = max(all_costs)

        logger.info(f"Router initialized with {len(models)} models")

    def route(
        self,
        question_text: str,
        cost_preference: float | None = None,
    ) -> RoutingDecision:
        """Route a question to the optimal model.

        Args:
            question_text: Question to route
            cost_preference: 0.0=cheap, 1.0=quality (default from config)

        Returns:
            RoutingDecision with selected model and reasoning
        """
        start_time = time.time()

        # Use default if not specified
        if cost_preference is None:
            cost_preference = self.default_cost_preference

        # 1. Assign to cluster
        cluster_id, distance = self.cluster_engine.assign_question(question_text)

        # 2. Calculate lambda parameter
        lambda_param = self._calculate_lambda(cost_preference)

        # 3. Compute routing scores for each model
        model_scores = {}

        for model_id, features in self.model_features.items():
            error_rate = features["error_rates"][cluster_id]
            cost = features["cost_per_1m_tokens"]

            normalized_cost = self._normalize_cost(cost)
            score = error_rate + lambda_param * normalized_cost

            model_scores[model_id] = {
                "score": score,
                "error_rate": error_rate,
                "accuracy": 1.0 - error_rate,
                "cost": cost,
                "normalized_cost": normalized_cost,
            }

        # 4. Select best model (lowest score)
        best_model_id = min(model_scores, key=lambda k: model_scores[k]["score"])
        best_scores = model_scores[best_model_id]

        # 5. Prepare decision
        model = self.models[best_model_id]

        routing_time = (time.time() - start_time) * 1000

        # Generate reasoning
        reasoning = self._generate_reasoning(
            cluster_id=cluster_id,
            cost_preference=cost_preference,
            lambda_param=lambda_param,
            selected_scores=best_scores,
        )

        # Prepare alternatives
        alternatives = [
            {
                "model_id": mid,
                "model_name": self.models[mid].name,
                "score": scores["score"],
                "accuracy": scores["accuracy"],
                "cost": scores["cost"],
            }
            for mid, scores in sorted(model_scores.items(), key=lambda x: x[1]["score"])
            if mid != best_model_id
        ]

        return RoutingDecision(
            selected_model_id=best_model_id,
            selected_model_name=model.name,
            routing_score=best_scores["score"],
            predicted_accuracy=best_scores["accuracy"],
            estimated_cost=best_scores["cost"]
            * ESTIMATED_TOKEN_COUNT
            / TOKENS_PER_MILLION,
            cluster_id=cluster_id,
            cluster_confidence=1.0 / (1.0 + distance),  # Convert distance to confidence
            lambda_param=lambda_param,
            reasoning=reasoning,
            alternatives=alternatives,
            routing_time_ms=routing_time,
        )

    def _calculate_lambda(self, cost_preference: float) -> float:
        """Calculate lambda parameter.

        Args:
            cost_preference: 0.0=cheap, 1.0=quality

        Returns:
            Lambda parameter (higher = more cost penalty)
        """
        # Invert: high quality preference = low lambda (cost matters less)
        lambda_param = self.lambda_max - cost_preference * (
            self.lambda_max - self.lambda_min
        )

        return lambda_param

    def _normalize_cost(self, cost: float) -> float:
        """Normalize cost to [0, 1] range.

        Args:
            cost: Model cost per 1M tokens

        Returns:
            Normalized cost
        """
        cost_range = self.max_cost - self.min_cost
        if cost_range < EPSILON:
            return 0.0

        return float((cost - self.min_cost) / cost_range)

    def _generate_reasoning(
        self,
        cluster_id: int,
        cost_preference: float,
        lambda_param: float,
        selected_scores: Dict[str, Any],
    ) -> str:
        """Generate human-readable reasoning for routing decision.

        Args:
            cluster_id: Assigned cluster
            cost_preference: User's cost preference
            lambda_param: Calculated lambda
            selected_scores: Scores for selected model

        Returns:
            Reasoning string
        """
        parts = []

        # Cluster info
        parts.append(f"Question assigned to cluster {cluster_id}")

        # Preference info
        if cost_preference < 0.3:
            parts.append(f"Cost-optimized routing (λ={lambda_param:.2f})")
        elif cost_preference < 0.7:
            parts.append(f"Balanced cost-accuracy routing (λ={lambda_param:.2f})")
        else:
            parts.append(f"Quality-optimized routing (λ={lambda_param:.2f})")

        # Performance info
        accuracy = selected_scores["accuracy"]
        if accuracy >= 0.95:
            parts.append(f"Excellent predicted accuracy ({accuracy:.0%})")
        elif accuracy >= 0.75:
            parts.append(f"Strong predicted accuracy ({accuracy:.0%})")
        else:
            parts.append(f"Best available option ({accuracy:.0%} predicted)")

        return "; ".join(parts)
