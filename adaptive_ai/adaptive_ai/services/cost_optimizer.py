"""Cost optimization utilities for dynamic cost-based model routing."""

from typing import Any

import cachetools
import numpy as np

from adaptive_ai.models.llm_core_models import ModelCapability, ModelEntry
from adaptive_ai.models.llm_enums import ProviderType


class CostOptimizer:
    """Cost optimization engine for model selection with pluggable strategies."""

    def __init__(self, strategy: str = "sigmoid"):
        """Initialize cost optimizer with specified strategy.

        Args:
            strategy: Optimization strategy ("sigmoid" or future "ml_model")
        """
        self.strategy = strategy
        self._cost_cache = cachetools.LRUCache(maxsize=10000)
        self._tier_cache = cachetools.LRUCache(maxsize=1000)

        # Strategy configuration
        self._tier_scores = {
            "ultra_budget": 0.4,
            "budget": 0.6,
            "mid": 0.8,
            "premium": 1.0,
        }

        self._tier_thresholds = [0.80, 2.00, 5.00]
        self._tier_names = ["ultra_budget", "budget", "mid", "premium"]

    def sigmoid_cost_bias(self, cost_bias: float) -> float:
        """Convert cost_bias (0-1) to sigmoid for smooth cost-performance interpolation."""
        return float(np.clip(1 / (1 + np.exp(-8 * (cost_bias - 0.5))), 0, 1))

    def calculate_model_cost(
        self,
        model_capability: ModelCapability,
        input_tokens: int,
        output_ratio: float = 0.3,
    ) -> float:
        """Calculate estimated cost for a model based on token usage."""
        # Check cache first
        cache_key = str((model_capability.model_name, input_tokens, f"{output_ratio:.3f}"))
        cached_result = self._cost_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        tokens = np.array([input_tokens, input_tokens * output_ratio])
        costs = np.array(
            [
                model_capability.cost_per_1m_input_tokens,
                model_capability.cost_per_1m_output_tokens,
            ]
        )

        cost = float(np.sum(tokens * costs) / 1_000_000)
        self._cost_cache[cache_key] = cost
        return cost

    def get_cost_tier(self, model_capability: ModelCapability) -> str:
        """Classify model tier based on output token cost."""
        # Check cache first
        cached_tier = self._tier_cache.get(model_capability.model_name)
        if cached_tier is not None:
            return cached_tier

        cost = model_capability.cost_per_1m_output_tokens
        index = int(np.searchsorted(self._tier_thresholds, cost or 0.5))
        tier = self._tier_names[index]

        self._tier_cache[model_capability.model_name] = tier
        return tier

    def get_performance_score_by_tier(self, tier: str) -> float:
        """Get performance score based on cost tier."""
        return self._tier_scores.get(tier, 0.5)

    def rank_models_by_cost_performance(
        self,
        model_entries: list[ModelEntry],
        cost_bias: float,
        model_capabilities: dict[tuple[ProviderType | str, str], ModelCapability],
        estimated_tokens: int,
    ) -> list[ModelEntry]:
        """Rank models using current optimization strategy."""
        if self.strategy == "sigmoid":
            return self._rank_models_sigmoid(
                model_entries, cost_bias, model_capabilities, estimated_tokens
            )
        # Future: elif self.strategy == "ml_model": return self._rank_models_ml(...)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _rank_models_sigmoid(
        self,
        model_entries: list[ModelEntry],
        cost_bias: float,
        model_capabilities: dict[tuple[ProviderType | str, str], ModelCapability],
        estimated_tokens: int,
    ) -> list[ModelEntry]:
        """Rank models using sigmoid cost-performance weighting."""
        if not model_entries:
            return []

        # Extract model data
        model_data = []
        entries_without_cost_data = []

        for entry in model_entries:
            if not entry.providers:
                continue
            key = (entry.providers[0], entry.model_name)
            if model_cap := model_capabilities.get(key):
                cost = self.calculate_model_cost(model_cap, estimated_tokens)
                tier = self.get_cost_tier(model_cap)
                model_data.append((entry, cost, tier))
            else:
                # Custom provider without cost data - preserve original order
                entries_without_cost_data.append(entry)

        if not model_data:
            # No models have cost data, return original order
            return model_entries

        # Vectorized calculations
        costs = np.array([data[1] for data in model_data])
        tiers = [data[2] for data in model_data]

        # Normalize costs and calculate scores
        normalized_costs = (
            costs / np.max(costs) if np.max(costs) > 0 else np.zeros_like(costs)
        )
        cost_scores = 1 - normalized_costs
        performance_scores = np.array(
            [self.get_performance_score_by_tier(tier) for tier in tiers]
        )

        # Sigmoid weighting and final scores
        sigmoid_weight = self.sigmoid_cost_bias(cost_bias)
        final_scores = (
            cost_scores * (1 - sigmoid_weight) + performance_scores * sigmoid_weight
        )

        # Sort by scores (descending)
        sorted_indices = np.argsort(final_scores)[::-1]
        sorted_entries = [model_data[i][0] for i in sorted_indices]

        # Append entries without cost data at the end
        return sorted_entries + entries_without_cost_data

    def get_cost_analysis(
        self,
        model_entries: list[ModelEntry],
        model_capabilities: dict[tuple[ProviderType | str, str], ModelCapability],
        estimated_tokens: int,
    ) -> dict[str, Any]:
        """Get comprehensive cost analysis for a list of models."""
        if not model_entries:
            return {
                "models": [],
                "cost_range": {"min": 0, "max": 0},
                "tier_distribution": {},
            }

        model_costs: list[dict[str, Any]] = []

        for entry in model_entries:
            if not entry.providers:
                continue
            key = (entry.providers[0], entry.model_name)
            if model_cap := model_capabilities.get(key):
                cost = self.calculate_model_cost(model_cap, estimated_tokens)
                tier = self.get_cost_tier(model_cap)

                model_costs.append(
                    {
                        "model_name": entry.model_name,
                        "provider": entry.providers[0].value,
                        "cost": cost,
                        "tier": tier,
                        "cost_per_1m_input": model_cap.cost_per_1m_input_tokens,
                        "cost_per_1m_output": model_cap.cost_per_1m_output_tokens,
                    }
                )

        if not model_costs:
            return {
                "models": [],
                "cost_range": {"min": 0, "max": 0},
                "tier_distribution": {},
            }

        # Vectorized cost calculations
        costs = np.array([m["cost"] for m in model_costs])
        tiers = [m["tier"] for m in model_costs]

        return {
            "models": model_costs,
            "cost_range": {"min": float(np.min(costs)), "max": float(np.max(costs))},
            "tier_distribution": {
                tier: sum(1 for t in tiers if t == tier) for tier in self._tier_names
            },
            "strategy": self.strategy,
            "cache_stats": {
                "cost_cache_size": len(self._cost_cache),
                "tier_cache_size": len(self._tier_cache),
            },
        }

    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._cost_cache.clear()
        self._tier_cache.clear()

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cost_cache": {
                "size": self._cost_cache.currsize,
                "max_size": self._cost_cache.maxsize,
            },
            "tier_cache": {
                "size": self._tier_cache.currsize,
                "max_size": self._tier_cache.maxsize,
            },
        }

    def set_strategy(self, strategy: str) -> None:
        """Change optimization strategy."""
        if strategy not in ["sigmoid"]:  # Future: add "ml_model"
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy

    # Future: Add ML model integration methods
    # def load_ml_model(self, model_path: str) -> None:
    #     """Load ML model for cost optimization."""
    #     pass
    #
    # def _rank_models_ml(self, ...) -> list[ModelEntry]:
    #     """Rank models using ML model predictions."""
    #     pass


# Global instance for backward compatibility and shared caching
_default_optimizer = CostOptimizer()


# Backward compatibility functions that use the global optimizer
def sigmoid_cost_bias(cost_bias: float) -> float:
    """Convert cost_bias (0-1) to sigmoid for smooth cost-performance interpolation."""
    return _default_optimizer.sigmoid_cost_bias(cost_bias)


def calculate_model_cost(
    model_capability: ModelCapability, input_tokens: int, output_ratio: float = 0.3
) -> float:
    """Calculate estimated cost for a model based on token usage."""
    return _default_optimizer.calculate_model_cost(
        model_capability, input_tokens, output_ratio
    )


def get_cost_tier(model_capability: ModelCapability) -> str:
    """Classify model tier based on output token cost."""
    return _default_optimizer.get_cost_tier(model_capability)


def get_performance_score_by_tier(tier: str) -> float:
    """Get performance score based on cost tier."""
    return _default_optimizer.get_performance_score_by_tier(tier)


def rank_models_by_cost_performance(
    model_entries: list[ModelEntry],
    cost_bias: float,
    model_capabilities: dict[tuple[ProviderType | str, str], ModelCapability],
    estimated_tokens: int,
) -> list[ModelEntry]:
    """Rank models using sigmoid cost-performance weighting."""
    return _default_optimizer.rank_models_by_cost_performance(
        model_entries, cost_bias, model_capabilities, estimated_tokens
    )


def get_cost_analysis(
    model_entries: list[ModelEntry],
    model_capabilities: dict[tuple[ProviderType | str, str], ModelCapability],
    estimated_tokens: int,
) -> dict[str, Any]:
    """Get cost analysis for a list of models."""
    return _default_optimizer.get_cost_analysis(
        model_entries, model_capabilities, estimated_tokens
    )
