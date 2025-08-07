#!/usr/bin/env python3
"""
Cost tracking functionality for LangGraph model extraction.
Tracks OpenAI API usage and enforces budget limits.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class CostTracker:
    """Tracks API costs and enforces budget limits"""

    def __init__(self, max_budget: float):
        self.max_budget = max_budget
        self.total_cost = 0.0
        self.request_count = 0
        self.start_time = time.time()

        # Token usage tracking
        self.input_tokens = 0
        self.output_tokens = 0

        # Model pricing (per 1M tokens)
        self.pricing = {
            "gpt-4o-mini": {
                "input": 0.150,  # $0.150 per 1M input tokens
                "output": 0.600,  # $0.600 per 1M output tokens
            }
        }

    def track_request(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Track a single API request and return cost"""
        if model not in self.pricing:
            logger.warning(f"Unknown model pricing: {model}")
            return 0.0

        pricing = self.pricing[model]

        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        request_cost = input_cost + output_cost

        # Update tracking
        self.total_cost += request_cost
        self.request_count += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

        logger.debug(
            f"Request cost: ${request_cost:.4f} (Total: ${self.total_cost:.4f})"
        )

        return request_cost

    def can_afford(self, estimated_cost: float) -> bool:
        """Check if we can afford an estimated cost"""
        return (self.total_cost + estimated_cost) <= self.max_budget

    def check_budget_before_batch(self, batch_size: int) -> bool:
        """Check if we can afford to process a batch"""
        estimated_batch_cost = batch_size * 0.003  # ~$0.003 per model
        return self.can_afford(estimated_batch_cost)

    def get_usage_summary(self) -> dict[str, Any]:
        """Get current usage summary"""
        elapsed_time = time.time() - self.start_time
        budget_used_percent = (self.total_cost / self.max_budget) * 100

        return {
            "total_cost": self.total_cost,
            "budget_limit": self.max_budget,
            "remaining_budget": self.max_budget - self.total_cost,
            "budget_used_percent": budget_used_percent,
            "request_count": self.request_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "elapsed_time": elapsed_time,
        }

    def print_summary(self) -> None:
        """Print cost summary"""
        summary = self.get_usage_summary()

        print("\n" + "=" * 50)
        print("💰 COST SUMMARY")
        print("=" * 50)
        print(f"💸 Total Cost: ${summary['total_cost']:.4f}")
        print(f"💰 Budget Limit: ${summary['budget_limit']:.2f}")
        print(f"💵 Remaining: ${summary['remaining_budget']:.4f}")
        print(f"📊 Budget Used: {summary['budget_used_percent']:.1f}%")
        print(f"📞 API Requests: {summary['request_count']}")
        print(f"📥 Input Tokens: {summary['input_tokens']:,}")
        print(f"📤 Output Tokens: {summary['output_tokens']:,}")
        print(f"⏱️  Runtime: {summary['elapsed_time']/60:.1f} minutes")
        print("=" * 50)

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation for cost calculation"""
        # Simple approximation: ~4 characters per token
        return len(text) // 4
