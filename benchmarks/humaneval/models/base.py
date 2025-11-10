"""
Base model class for HumanEval benchmarking with response-based cost tracking.

All model implementations should inherit from BaseHumanEvalModel and implement
the required methods for generating code samples with detailed metrics tracking.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ResponseMetrics:
    """
    Metrics extracted directly from API responses.

    This captures the actual billed costs and token usage from each API call,
    ensuring accurate tracking without manual calculation.
    """

    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None  # Actual cost from API response
    latency_seconds: float = 0.0
    model_used: str = ""  # Important for Adaptive routing
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @property
    def total_tokens(self) -> int:
        """Total tokens used in this request."""
        return self.input_tokens + self.output_tokens


@dataclass
class SampleResult:
    """
    Result from a single code generation sample.

    Tracks both the generated code and all associated metrics.
    """

    code: str
    metrics: ResponseMetrics
    task_id: str
    sample_index: int
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "code": self.code,
            "metrics": self.metrics.to_dict(),
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "success": self.success,
        }


@dataclass
class TaskMetrics:
    """
    Aggregated metrics for all samples of a single task.

    This tracks cumulative data across all n samples generated for one HumanEval task.
    """

    task_id: str
    task_name: str
    samples_generated: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency: float = 0.0
    successful_samples: int = 0
    failed_samples: int = 0
    per_sample_costs: list[float] = field(default_factory=list)
    per_sample_metrics: list[ResponseMetrics] = field(default_factory=list)

    def add_sample(self, metrics: ResponseMetrics):
        """Add metrics from a single sample."""
        self.total_input_tokens += metrics.input_tokens
        self.total_output_tokens += metrics.output_tokens
        if metrics.cost_usd is not None:
            self.total_cost_usd += metrics.cost_usd
            self.per_sample_costs.append(metrics.cost_usd)
        self.per_sample_metrics.append(metrics)

        if metrics.error is None:
            self.successful_samples += 1
        else:
            self.failed_samples += 1

    def finalize(self):
        """Calculate final averages after all samples added."""
        if self.successful_samples > 0:
            total_latency = sum(
                m.latency_seconds for m in self.per_sample_metrics if m.error is None
            )
            self.avg_latency = total_latency / self.successful_samples

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "samples_generated": self.samples_generated,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_cost_per_sample": round(
                (
                    self.total_cost_usd / self.samples_generated
                    if self.samples_generated > 0
                    else 0
                ),
                6,
            ),
            "avg_latency": round(self.avg_latency, 3),
            "successful_samples": self.successful_samples,
            "failed_samples": self.failed_samples,
            "success_rate": round(
                (
                    self.successful_samples / self.samples_generated
                    if self.samples_generated > 0
                    else 0
                ),
                3,
            ),
            "per_sample_costs": [round(c, 6) for c in self.per_sample_costs],
        }


class BaseHumanEvalModel(ABC):
    """
    Abstract base class for HumanEval model implementations.

    All model wrappers (Claude, GLM, Adaptive) must inherit from this class
    and implement the required methods.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = logging.getLogger(f"{__name__}.{model_name}")

    @abstractmethod
    def generate_with_metrics(
        self, prompt: str, temperature: float = 0.8, max_tokens: int = 1024
    ) -> tuple[str, ResponseMetrics]:
        """
        Generate a single code completion with metrics.

        Args:
            prompt: The code generation prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (generated_code, response_metrics)

        This method must extract cost and token information from the API response.
        """
        pass

    def generate_samples(
        self,
        prompt: str,
        n: int,
        temperature: float = 0.8,
        max_tokens: int = 1024,
        task_id: str = "",
    ) -> tuple[list[str], TaskMetrics]:
        """
        Generate multiple code samples for a single task.

        This is the main method called by DeepEval's HumanEval benchmark.

        Args:
            prompt: The code generation prompt
            n: Number of samples to generate
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            task_id: HumanEval task identifier

        Returns:
            Tuple of (list of generated codes, aggregated task metrics)
        """
        self.logger.info(f"Generating {n} samples for task: {task_id}")

        codes = []
        task_metrics = TaskMetrics(
            task_id=task_id,
            task_name=self._extract_task_name(task_id),
            samples_generated=n,
        )

        for i in range(n):
            try:
                start_time = time.time()
                code, metrics = self.generate_with_metrics(
                    prompt=prompt, temperature=temperature, max_tokens=max_tokens
                )
                metrics.latency_seconds = time.time() - start_time

                codes.append(code)
                task_metrics.add_sample(metrics)

                if (i + 1) % 10 == 0:
                    self.logger.info(
                        f"Progress: {i + 1}/{n} samples generated "
                        f"(avg latency: {task_metrics.avg_latency:.2f}s)"
                    )

            except Exception as e:
                self.logger.error(f"Error generating sample {i}: {str(e)}")
                # Add failed sample with error metrics
                error_metrics = ResponseMetrics(
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    latency_seconds=0.0,
                    model_used=self.model_name,
                    error=str(e),
                )
                codes.append("")  # Empty code for failed sample
                task_metrics.add_sample(error_metrics)

        task_metrics.finalize()

        self.logger.info(
            f"Completed task {task_id}: "
            f"{task_metrics.successful_samples}/{n} successful, "
            f"cost: ${task_metrics.total_cost_usd:.4f}, "
            f"tokens: {task_metrics.total_input_tokens + task_metrics.total_output_tokens}"
        )

        return codes, task_metrics

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier."""
        pass

    def _extract_task_name(self, task_id: str) -> str:
        """Extract human-readable task name from task_id."""
        # HumanEval task IDs are like "HumanEval/0" or "HumanEval/163"
        if "/" in task_id:
            return task_id.split("/")[-1]
        return task_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
