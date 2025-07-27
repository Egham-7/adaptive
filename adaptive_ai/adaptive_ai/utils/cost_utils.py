"""Cost calculation utilities for dynamic cost-based model routing."""

import numpy as np
from typing import Dict, List, Tuple

from adaptive_ai.models.llm_core_models import ModelEntry, ModelCapability
from adaptive_ai.models.llm_enums import ProviderType


def sigmoid_cost_bias(cost_bias: float) -> float:
    """Convert cost_bias (0-1) to sigmoid for smooth cost-performance interpolation."""
    return float(np.clip(1 / (1 + np.exp(-8 * (cost_bias - 0.5))), 0, 1))


def calculate_model_cost(model_capability: ModelCapability, input_tokens: int, output_ratio: float = 0.3) -> float:
    """Calculate estimated cost for a model based on token usage."""
    tokens = np.array([input_tokens, input_tokens * output_ratio])
    costs = np.array([model_capability.cost_per_1m_input_tokens, model_capability.cost_per_1m_output_tokens])
    return float(np.sum(tokens * costs) / 1_000_000)


def get_cost_tier(model_capability: ModelCapability) -> str:
    """Classify model tier based on output token cost."""
    cost = model_capability.cost_per_1m_output_tokens
    tiers = ["ultra_budget", "budget", "mid", "premium"]
    thresholds = [0.80, 2.00, 5.00]
    return tiers[np.searchsorted(thresholds, cost)]


def get_performance_score_by_tier(tier: str) -> float:
    """Get performance score based on cost tier."""
    scores = {"ultra_budget": 0.4, "budget": 0.6, "mid": 0.8, "premium": 1.0}
    return scores.get(tier, 0.5)


def rank_models_by_cost_performance(
    model_entries: List[ModelEntry],
    cost_bias: float,
    model_capabilities: Dict[Tuple[ProviderType, str], ModelCapability],
    estimated_tokens: int
) -> List[ModelEntry]:
    """Rank models using sigmoid cost-performance weighting."""
    if not model_entries:
        return []
    
    # Extract model data
    model_data = []
    for entry in model_entries:
        key = (entry.providers[0], entry.model_name)
        if model_cap := model_capabilities.get(key):
            cost = calculate_model_cost(model_cap, estimated_tokens)
            tier = get_cost_tier(model_cap)
            model_data.append((entry, cost, tier))
    
    if not model_data:
        return model_entries
    
    # Vectorized calculations
    costs = np.array([data[1] for data in model_data])
    tiers = [data[2] for data in model_data]
    
    # Normalize costs and calculate scores
    normalized_costs = costs / np.max(costs) if np.max(costs) > 0 else np.zeros_like(costs)
    cost_scores = 1 - normalized_costs
    performance_scores = np.array([get_performance_score_by_tier(tier) for tier in tiers])
    
    # Sigmoid weighting and final scores
    sigmoid_weight = sigmoid_cost_bias(cost_bias)
    final_scores = cost_scores * (1 - sigmoid_weight) + performance_scores * sigmoid_weight
    
    # Sort by scores (descending)
    sorted_indices = np.argsort(final_scores)[::-1]
    return [model_data[i][0] for i in sorted_indices]


def get_cost_analysis(
    model_entries: List[ModelEntry],
    model_capabilities: Dict[Tuple[ProviderType, str], ModelCapability],
    estimated_tokens: int
) -> Dict[str, any]:
    """Get cost analysis for a list of models."""
    if not model_entries:
        return {"models": [], "cost_range": {"min": 0, "max": 0}, "tier_distribution": {}}
    
    model_costs = []
    for entry in model_entries:
        key = (entry.providers[0], entry.model_name)
        if model_cap := model_capabilities.get(key):
            cost = calculate_model_cost(model_cap, estimated_tokens)
            tier = get_cost_tier(model_cap)
            
            model_costs.append({
                "model_name": entry.model_name,
                "provider": entry.providers[0].value,
                "cost": cost,
                "tier": tier,
                "cost_per_1m_input": model_cap.cost_per_1m_input_tokens,
                "cost_per_1m_output": model_cap.cost_per_1m_output_tokens
            })
    
    if not model_costs:
        return {"models": [], "cost_range": {"min": 0, "max": 0}, "tier_distribution": {}}
    
    # Vectorized cost calculations
    costs = np.array([m["cost"] for m in model_costs])
    tiers = [m["tier"] for m in model_costs]
    
    return {
        "models": model_costs,
        "cost_range": {"min": float(np.min(costs)), "max": float(np.max(costs))},
        "tier_distribution": {
            tier: sum(1 for t in tiers if t == tier)
            for tier in ["ultra_budget", "budget", "mid", "premium"]
        }
    }