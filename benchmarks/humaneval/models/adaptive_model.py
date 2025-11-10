"""
Adaptive routing model implementation for HumanEval benchmarking.

This module provides a wrapper around the Adaptive routing API with detailed
cost and token tracking. Tracks which model was actually selected for each request.
"""

import logging
import os
from typing import Any

from openai import OpenAI

from ..utils.response_parser import parse_adaptive_response
from .base import BaseHumanEvalModel, ResponseMetrics

logger = logging.getLogger(__name__)


class AdaptiveModel(BaseHumanEvalModel):
    """
    Adaptive routing model wrapper with response-based cost tracking.

    This implementation uses the Adaptive API and extracts token usage, cost,
    and selected model information directly from API responses.
    """

    DEFAULT_MODELS = [
        "anthropic:claude-sonnet-4-5-20250929",
        "zai:glm-4.6",
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
            api_base: Adaptive API base URL (defaults to ADAPTIVE_API_BASE env var)
            models: List of models to route between (defaults to DEFAULT_MODELS)
            cost_bias: Cost vs performance trade-off (0=cheapest, 1=best performance)
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
            "ADAPTIVE_API_BASE", "https://api.llmadaptive.uk/v1"
        )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        self.models = models or self.DEFAULT_MODELS
        self.cost_bias = cost_bias

        # Track model selection statistics
        self.model_selection_counts = {}

        self.logger.info(
            f"Initialized Adaptive router with {len(self.models)} models, "
            f"cost_bias={cost_bias}"
        )

    def generate_with_metrics(
        self, prompt: str, temperature: float = 0.8, max_tokens: int = 1024
    ) -> tuple[str, ResponseMetrics]:
        """
        Generate a single code completion with metrics using Adaptive routing.

        Args:
            prompt: The code generation prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_code, response_metrics with selected model)
        """
        try:
            # Call Adaptive API using OpenAI client with model_router
            completion = self.client.chat.completions.create(
                model="",  # Empty model, routing handled by model_router
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={
                    "model_router": {
                        "models": self.models,
                        "cost_bias": self.cost_bias,
                    }
                },
            )

            # Extract generated code from response
            code = completion.choices[0].message.content

            # Parse metrics from OpenAI-style response
            metrics = parse_adaptive_response(
                response_json=completion.model_dump(), requested_models=self.models
            )

            # Track which model was selected
            selected_model = metrics.model_used
            self.model_selection_counts[selected_model] = (
                self.model_selection_counts.get(selected_model, 0) + 1
            )

            self.logger.debug(
                f"Adaptive selected {selected_model}: "
                f"{metrics.output_tokens} tokens, cost: ${metrics.cost_usd:.6f}"
            )

            return code, metrics

        except Exception as e:
            self.logger.error(f"Error calling Adaptive API: {str(e)}")
            error_metrics = ResponseMetrics(
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                model_used="error",
                error=str(e),
            )
            return "", error_metrics

    def get_model_name(self) -> str:
        """Return the model identifier."""
        return self.model_name

    def get_model_selection_stats(self) -> dict[str, Any]:
        """
        Get statistics about which models were selected.

        Returns:
            Dictionary with model selection counts and percentages
        """
        total = sum(self.model_selection_counts.values())
        if total == 0:
            return {}

        stats = {"total_requests": total, "models": {}}

        for model, count in self.model_selection_counts.items():
            stats["models"][model] = {
                "count": count,
                "percentage": round(count / total * 100, 2),
            }

        return stats


class AdaptiveForDeepEval:
    """
    DeepEval-compatible wrapper for Adaptive routing model.

    This class implements the interface expected by DeepEval's benchmarking
    framework while using our AdaptiveModel with full metrics tracking.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        models: list[str] | None = None,
        cost_bias: float = 0.5,
    ):
        self.model = AdaptiveModel(
            api_key=api_key, api_base=api_base, models=models, cost_bias=cost_bias
        )
        self.task_metrics_history = []

    def load_model(self):
        """Load the model (no-op since we use REST API)."""
        return None

    def generate(self, prompt: str) -> str:
        """
        Standard generation method for DeepEval.

        Args:
            prompt: Code generation prompt

        Returns:
            Generated code string
        """
        code, _ = self.model.generate_with_metrics(prompt)
        return code

    def generate_samples(
        self, prompt: str, n: int, temperature: float = 0.8
    ) -> list[str]:
        """
        Generate multiple code samples (required by HumanEval benchmark).

        Args:
            prompt: Code generation prompt
            n: Number of samples to generate
            temperature: Sampling temperature

        Returns:
            List of generated code samples
        """
        # Extract task ID from prompt if possible
        task_id = self._extract_task_id_from_prompt(prompt)

        # Use our base class implementation which tracks metrics
        codes, task_metrics = self.model.generate_samples(
            prompt=prompt,
            n=n,
            temperature=temperature,
            max_tokens=1024,
            task_id=task_id,
        )

        # Store metrics for later retrieval
        self.task_metrics_history.append(task_metrics)

        return codes

    async def a_generate(self, prompt: str) -> str:
        """
        Async generation method for DeepEval.

        Note: Currently uses synchronous implementation.
        TODO: Implement true async version using httpx or aiohttp.
        """
        return self.generate(prompt)

    def get_model_name(self) -> str:
        """Return the model identifier for DeepEval."""
        return self.model.get_model_name()

    def _extract_task_id_from_prompt(self, prompt: str) -> str:
        """
        Try to extract HumanEval task ID from prompt.

        HumanEval prompts typically contain task identifiers.
        """
        lines = prompt.split("\n")
        for line in lines:
            if "HumanEval" in line or "def " in line:
                if "def " in line:
                    func_name = line.split("def ")[1].split("(")[0].strip()
                    return func_name
        return "unknown"

    def get_all_task_metrics(self):
        """
        Get all collected task metrics from benchmark runs.

        Returns:
            List of TaskMetrics objects
        """
        return self.task_metrics_history

    def get_total_cost(self) -> float:
        """
        Calculate total cost across all tasks.

        Returns:
            Total cost in USD
        """
        return sum(tm.total_cost_usd for tm in self.task_metrics_history)

    def get_total_tokens(self) -> int:
        """
        Calculate total tokens across all tasks.

        Returns:
            Total token count
        """
        return sum(
            tm.total_input_tokens + tm.total_output_tokens
            for tm in self.task_metrics_history
        )

    def get_model_selection_stats(self) -> dict[str, Any]:
        """
        Get statistics about which models were selected by Adaptive routing.

        Returns:
            Dictionary with model selection statistics
        """
        return self.model.get_model_selection_stats()

    def clear_metrics(self):
        """Clear collected metrics history."""
        self.task_metrics_history = []
        self.model.model_selection_counts = {}
