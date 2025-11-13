"""
Adaptive routing model implementation for SWE-bench benchmarking.

This module provides a wrapper around the Adaptive routing API with detailed
cost and token tracking. Tracks which model was actually selected for each request.
"""

import logging
import os
import time
from typing import Any

from openai import OpenAI

from ..utils.response_parser import parse_adaptive_response
from .base import BaseSWEBenchModel, ResponseMetrics

logger = logging.getLogger(__name__)


class AdaptiveModel(BaseSWEBenchModel):
    """
    Adaptive routing model wrapper with response-based cost tracking.

    This implementation uses the Adaptive API and extracts token usage, cost,
    and selected model information directly from API responses.
    """

    DEFAULT_MODELS = [
        "gpt-4o-mini",
        "claude-3-5-sonnet-20241022",
        "gemini-2.0-flash-exp",
    ]

    def __init__(
        self,
        model_name: str = "adaptive-router",
        api_key: str | None = None,
        api_base: str | None = None,
        models: list[str] | None = None,
        cost_bias: float = 0.5,
    ):
        """
        Initialize Adaptive model.

        Args:
            model_name: Adaptive model identifier (default: adaptive-router)
            api_key: Adaptive API key (defaults to ADAPTIVE_API_KEY env var)
            api_base: Adaptive API base URL (defaults to ADAPTIVE_BASE_URL env var)
            models: List of models to route between (defaults to DEFAULT_MODELS)
            cost_bias: Cost vs performance trade-off (0=best performance, 1=cheapest)
        """
        super().__init__(model_name=model_name)

        # Initialize API credentials
        self.api_key = api_key or os.getenv("ADAPTIVE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Adaptive API key not found. Set ADAPTIVE_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.api_base = api_base or os.getenv(
            "ADAPTIVE_BASE_URL", "https://api.llmadaptive.uk/v1"
        )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        self.models = models or self.DEFAULT_MODELS
        self.cost_bias = cost_bias

        # Track model selection statistics
        self.model_selection_counts: dict[str, int] = {}

        self.logger.info(
            f"Initialized Adaptive router with {len(self.models)} models, "
            f"cost_bias={cost_bias}"
        )

    def generate_patch(
        self, problem_statement: str, repo_context: str, temperature: float, max_tokens: int
    ) -> tuple[str, ResponseMetrics]:
        """
        Generate a patch for the given problem using Adaptive routing.

        Args:
            problem_statement: The issue description
            repo_context: Context about the repository
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (patch_content, response_metrics with selected model)
        """
        start_time = time.time()

        try:
            # Construct the prompt for patch generation
            prompt = self._build_patch_prompt(problem_statement, repo_context)

            # Call Adaptive API using OpenAI client with model_router
            completion = self.client.chat.completions.create(
                model="",  # Empty model, routing handled by model_router
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert software engineer. Generate precise code patches to fix issues.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={
                    "model_router": {"models": self.models, "cost_bias": self.cost_bias}
                },
            )

            # Extract generated patch from response
            patch_content = completion.choices[0].message.content or ""

            # Parse metrics from OpenAI-style response
            input_tokens, output_tokens, cost, selected_model = parse_adaptive_response(
                response_json=completion.model_dump(), requested_models=self.models
            )

            # Calculate latency
            latency = time.time() - start_time

            # Create metrics object
            metrics = ResponseMetrics(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_seconds=latency,
                model_used=selected_model,
            )

            # Track which model was selected
            self.model_selection_counts[selected_model] = (
                self.model_selection_counts.get(selected_model, 0) + 1
            )

            self.logger.debug(
                f"Adaptive selected {selected_model}: "
                f"{output_tokens} tokens, cost: ${cost:.6f}"
            )

            return patch_content, metrics

        except Exception as e:
            self.logger.error(f"Error calling Adaptive API: {str(e)}")
            latency = time.time() - start_time
            error_metrics = ResponseMetrics(
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_seconds=latency,
                model_used="error",
                error=str(e),
            )
            return "", error_metrics

    def _build_patch_prompt(self, problem_statement: str, repo_context: str) -> str:
        """
        Build a prompt for patch generation.

        Args:
            problem_statement: The issue description
            repo_context: Context about the repository

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are tasked with fixing a bug in a software repository.

## Problem Statement
{problem_statement}
"""

        if repo_context:
            prompt += f"""
## Repository Context
{repo_context}
"""

        prompt += """
## Instructions
1. Analyze the problem statement carefully
2. Identify the root cause of the issue
3. Generate a precise patch (diff format) that fixes the issue
4. Ensure the patch is minimal and doesn't break existing functionality
5. Output ONLY the patch content in unified diff format

Generate the patch now:
"""

        return prompt

    def get_model_selection_stats(self) -> dict[str, Any]:
        """
        Get statistics about which models were selected.

        Returns:
            Dictionary with model selection counts and percentages
        """
        total = sum(self.model_selection_counts.values())
        if total == 0:
            return {}

        stats: dict[str, Any] = {"total_requests": total, "models": {}}

        for model, count in self.model_selection_counts.items():
            stats["models"][model] = {
                "count": count,
                "percentage": round(count / total * 100, 2),
            }

        return stats

    def clear_stats(self) -> None:
        """Clear model selection statistics."""
        self.model_selection_counts = {}
