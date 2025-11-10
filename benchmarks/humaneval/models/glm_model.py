"""
GLM-4.6 model implementation for HumanEval benchmarking.

This module provides a wrapper around the GLM API with detailed
cost and token tracking extracted from API responses.
"""

import logging
import os

import requests

from ..utils.response_parser import parse_glm_response
from .base import BaseHumanEvalModel, ResponseMetrics

logger = logging.getLogger(__name__)


class GLMModel(BaseHumanEvalModel):
    """
    GLM-4.6 model wrapper with response-based cost tracking.

    This implementation uses the GLM REST API and extracts
    token usage and cost information directly from API responses.
    """

    def __init__(
        self,
        model_name: str = "glm-4.6",
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        """
        Initialize GLM model.

        Args:
            model_name: GLM model identifier (default: glm-4.6)
            api_key: GLM API key (defaults to GLM_API_KEY env var)
            api_base: GLM API base URL (defaults to GLM_API_BASE env var)
        """
        super().__init__(model_name=model_name)

        # Initialize API credentials
        self.api_key = api_key or os.getenv("GLM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GLM API key not found. Set GLM_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.api_base = api_base or os.getenv(
            "GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4/"
        )

        # Ensure API base ends with /
        if not self.api_base.endswith("/"):
            self.api_base += "/"

        self.endpoint = f"{self.api_base}chat/completions"
        self.logger.info(f"Initialized GLM model: {model_name}")

    def generate_with_metrics(
        self, prompt: str, temperature: float = 0.8, max_tokens: int = 1024
    ) -> tuple[str, ResponseMetrics]:
        """
        Generate a single code completion with metrics from GLM.

        Args:
            prompt: The code generation prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_code, response_metrics)
        """
        try:
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Call GLM API
            response = requests.post(
                self.endpoint, headers=headers, json=payload, timeout=60
            )

            response.raise_for_status()
            response_json = response.json()

            # Extract generated code
            code = response_json["choices"][0]["message"]["content"]

            # Parse metrics from response
            metrics = parse_glm_response(
                response_json=response_json,
                model_name=self.model_name,
                prompt=prompt,
                completion=code,
            )

            self.logger.debug(
                f"Generated {metrics.output_tokens} tokens, "
                f"cost: ${metrics.cost_usd:.6f}"
            )

            return code, metrics

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error calling GLM API: {str(e)}")
            error_metrics = ResponseMetrics(
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                model_used=self.model_name,
                error=str(e),
            )
            return "", error_metrics

        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
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


class GLMForDeepEval:
    """
    DeepEval-compatible wrapper for GLM model.

    This class implements the interface expected by DeepEval's benchmarking
    framework while using our GLMModel with full metrics tracking.
    """

    def __init__(
        self,
        model_name: str = "glm-4.6",
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        self.model = GLMModel(model_name=model_name, api_key=api_key, api_base=api_base)
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

    def clear_metrics(self):
        """Clear collected metrics history."""
        self.task_metrics_history = []
