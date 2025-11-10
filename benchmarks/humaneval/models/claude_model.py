"""
Claude 4.5 Sonnet model implementation for HumanEval benchmarking.

This module provides a wrapper around Anthropic's Claude API with detailed
cost and token tracking extracted from API responses.
"""

import logging
import os

from anthropic import Anthropic

from ..utils.response_parser import parse_claude_response
from .base import BaseHumanEvalModel, ResponseMetrics

logger = logging.getLogger(__name__)


class ClaudeModel(BaseHumanEvalModel):
    """
    Claude 4.5 Sonnet model wrapper with response-based cost tracking.

    This implementation uses the official Anthropic Python SDK and extracts
    token usage and cost information directly from API responses.
    """

    def __init__(
        self, model_name: str = "claude-sonnet-4-5", api_key: str | None = None
    ):
        """
        Initialize Claude model.

        Args:
            model_name: Claude model identifier (default: claude-sonnet-4-5)
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        super().__init__(model_name=model_name)

        # Initialize Anthropic client
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.client = Anthropic(api_key=self.api_key)
        self.logger.info(f"Initialized Claude model: {model_name}")

    def generate_with_metrics(
        self, prompt: str, temperature: float = 0.8, max_tokens: int = 1024
    ) -> tuple[str, ResponseMetrics]:
        """
        Generate a single code completion with metrics from Claude.

        Args:
            prompt: The code generation prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_code, response_metrics)
        """
        try:
            # Call Claude API
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract generated code
            code = response.content[0].text

            # Parse metrics from response
            metrics = parse_claude_response(response, self.model_name)

            self.logger.debug(
                f"Generated {metrics.output_tokens} tokens, "
                f"cost: ${metrics.cost_usd:.6f}"
            )

            return code, metrics

        except Exception as e:
            self.logger.error(f"Error calling Claude API: {str(e)}")
            error_metrics = ResponseMetrics(
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                model_used=self.model_name,
                error=str(e),
            )
            return "", error_metrics

    def get_model_name(self) -> str:
        """Return the model identifier."""
        return self.model_name


class ClaudeForDeepEval:
    """
    DeepEval-compatible wrapper for Claude model.

    This class implements the interface expected by DeepEval's benchmarking
    framework while using our ClaudeModel with full metrics tracking.
    """

    def __init__(
        self, model_name: str = "claude-sonnet-4-5", api_key: str | None = None
    ):
        self.model = ClaudeModel(model_name=model_name, api_key=api_key)
        self.task_metrics_history = []

    def load_model(self):
        """Load the model (no-op since model is initialized in __init__)."""
        return self.model.client

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
        TODO: Implement true async version using Anthropic's async client.
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
        # Look for patterns like "HumanEval/0" or task names
        lines = prompt.split("\n")
        for line in lines:
            if "HumanEval" in line or "def " in line:
                # Extract function name as task identifier
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

    def clear_metrics(self):
        """Clear collected metrics history."""
        self.task_metrics_history = []
