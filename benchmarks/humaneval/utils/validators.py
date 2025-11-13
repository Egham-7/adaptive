"""
Result validation utilities for HumanEval benchmarks.

This module provides validation functions to ensure data integrity
and catch issues before saving results.
"""

import logging
from typing import TYPE_CHECKING, Any

from ..models.base import ResponseMetrics, TaskMetrics

if TYPE_CHECKING:
    from .result_tracker import BenchmarkRun

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


def validate_response_metrics(metrics: ResponseMetrics) -> bool:
    """
    Validate ResponseMetrics object.

    Args:
        metrics: ResponseMetrics to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    # Check tokens are non-negative
    if metrics.input_tokens < 0:
        raise ValidationError(f"Invalid input_tokens: {metrics.input_tokens} (must be >= 0)")

    if metrics.output_tokens < 0:
        raise ValidationError(
            f"Invalid output_tokens: {metrics.output_tokens} (must be >= 0)"
        )

    # Check cost is non-negative (can be 0 for free tiers)
    if metrics.cost_usd < 0:
        raise ValidationError(f"Invalid cost_usd: {metrics.cost_usd} (must be >= 0)")

    # Check latency is positive
    if metrics.latency_seconds < 0:
        raise ValidationError(
            f"Invalid latency_seconds: {metrics.latency_seconds} (must be >= 0)"
        )

    # Warn if latency is suspiciously high (> 60 seconds)
    if metrics.latency_seconds > 60:
        logger.warning(
            f"Unusually high latency: {metrics.latency_seconds:.2f}s "
            f"for model {metrics.model_used}"
        )

    # Check model_used is not empty
    if not metrics.model_used or not metrics.model_used.strip():
        raise ValidationError("model_used cannot be empty")

    # Warn if cost is 0 (might indicate missing cost tracking)
    if metrics.cost_usd == 0 and metrics.output_tokens > 0:
        logger.warning(
            f"Cost is $0.00 but generated {metrics.output_tokens} tokens - "
            "cost tracking may not be working"
        )

    return True


def validate_task_metrics(metrics: TaskMetrics) -> bool:
    """
    Validate TaskMetrics object.

    Args:
        metrics: TaskMetrics to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    # Check task_id format (should be like "HumanEval/0" or "HumanEval/123")
    if not metrics.task_id or not metrics.task_id.startswith("HumanEval/"):
        raise ValidationError(f"Invalid task_id format: {metrics.task_id}")

    # Check sample counts
    if metrics.samples_generated <= 0:
        raise ValidationError(
            f"Invalid samples_generated: {metrics.samples_generated} (must be > 0)"
        )

    total_samples = metrics.successful_samples + metrics.failed_samples
    if total_samples != metrics.samples_generated:
        raise ValidationError(
            f"Sample count mismatch: generated={metrics.samples_generated}, "
            f"successful={metrics.successful_samples}, failed={metrics.failed_samples}"
        )

    # Check tokens are non-negative
    if metrics.total_input_tokens < 0:
        raise ValidationError(
            f"Invalid total_input_tokens: {metrics.total_input_tokens} (must be >= 0)"
        )

    if metrics.total_output_tokens < 0:
        raise ValidationError(
            f"Invalid total_output_tokens: {metrics.total_output_tokens} (must be >= 0)"
        )

    # Check cost is non-negative
    if metrics.total_cost_usd < 0:
        raise ValidationError(
            f"Invalid total_cost_usd: {metrics.total_cost_usd} (must be >= 0)"
        )

    # Check latency is non-negative
    if metrics.avg_latency < 0:
        raise ValidationError(f"Invalid avg_latency: {metrics.avg_latency} (must be >= 0)")

    # Warn if all samples failed
    if metrics.failed_samples == metrics.samples_generated:
        logger.warning(
            f"All samples failed for task {metrics.task_id} - "
            "this may indicate an API or configuration issue"
        )

    # Warn if no successful samples
    if metrics.successful_samples == 0:
        logger.warning(f"No successful samples for task {metrics.task_id}")

    return True


def validate_benchmark_run(benchmark_run: "BenchmarkRun") -> bool:
    """
    Validate BenchmarkRun object.

    Args:
        benchmark_run: BenchmarkRun to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    # Check model name is not empty
    if not benchmark_run.model_name or not benchmark_run.model_name.strip():
        raise ValidationError("model_name cannot be empty")

    # Check overall_score is in valid range [0, 1]
    if not 0 <= benchmark_run.overall_score <= 1:
        raise ValidationError(
            f"Invalid overall_score: {benchmark_run.overall_score} "
            f"(must be between 0 and 1)"
        )

    # Check pass_at_k structure
    if "k" not in benchmark_run.pass_at_k:
        raise ValidationError("pass_at_k must contain 'k' key")

    if "score" not in benchmark_run.pass_at_k:
        raise ValidationError("pass_at_k must contain 'score' key")

    k = benchmark_run.pass_at_k["k"]
    score = benchmark_run.pass_at_k["score"]

    if not isinstance(k, int) or k <= 0:
        raise ValidationError(f"Invalid k value: {k} (must be positive integer)")

    if not 0 <= score <= 1:
        raise ValidationError(
            f"Invalid pass@k score: {score} (must be between 0 and 1)"
        )

    # Check pass@k score matches overall_score
    if abs(score - benchmark_run.overall_score) > 0.001:
        logger.warning(
            f"pass@k score ({score}) doesn't match overall_score "
            f"({benchmark_run.overall_score})"
        )

    # Check task counts
    if benchmark_run.total_tasks <= 0:
        raise ValidationError(
            f"Invalid total_tasks: {benchmark_run.total_tasks} (must be > 0)"
        )

    expected_samples = benchmark_run.total_tasks * (
        benchmark_run.total_samples // benchmark_run.total_tasks
        if benchmark_run.total_tasks > 0
        else 0
    )

    # Check total samples is reasonable (should be total_tasks * n_samples)
    if benchmark_run.total_samples <= 0:
        raise ValidationError(
            f"Invalid total_samples: {benchmark_run.total_samples} (must be > 0)"
        )

    # Check tokens
    if benchmark_run.total_input_tokens < 0:
        raise ValidationError(
            f"Invalid total_input_tokens: {benchmark_run.total_input_tokens}"
        )

    if benchmark_run.total_output_tokens < 0:
        raise ValidationError(
            f"Invalid total_output_tokens: {benchmark_run.total_output_tokens}"
        )

    # Check cost
    if benchmark_run.total_cost_usd < 0:
        raise ValidationError(f"Invalid total_cost_usd: {benchmark_run.total_cost_usd}")

    # Warn if cost is 0
    if benchmark_run.total_cost_usd == 0 and benchmark_run.total_output_tokens > 0:
        logger.warning(
            "Total cost is $0.00 but generated tokens - "
            "cost tracking may not be working"
        )

    # Check latency
    if benchmark_run.avg_latency_seconds < 0:
        raise ValidationError(
            f"Invalid avg_latency_seconds: {benchmark_run.avg_latency_seconds}"
        )

    # Check success rate is in valid range [0, 1]
    if not 0 <= benchmark_run.success_rate <= 1:
        raise ValidationError(
            f"Invalid success_rate: {benchmark_run.success_rate} "
            f"(must be between 0 and 1)"
        )

    # Check per_task_results length matches total_tasks
    if len(benchmark_run.per_task_results) != benchmark_run.total_tasks:
        raise ValidationError(
            f"per_task_results length ({len(benchmark_run.per_task_results)}) "
            f"doesn't match total_tasks ({benchmark_run.total_tasks})"
        )

    # Validate each task result
    for i, task_result in enumerate(benchmark_run.per_task_results):
        try:
            validate_task_result_dict(task_result)
        except ValidationError as e:
            raise ValidationError(f"Invalid task result at index {i}: {e}")

    # Check timestamp format (ISO format)
    if not benchmark_run.timestamp:
        raise ValidationError("timestamp cannot be empty")

    logger.info(
        f"Validation passed for {benchmark_run.model_name}: "
        f"{benchmark_run.total_tasks} tasks, "
        f"{benchmark_run.total_samples} samples, "
        f"pass@{k}={score:.4f}"
    )

    return True


def validate_task_result_dict(task_result: dict[str, Any]) -> bool:
    """
    Validate a task result dictionary (from per_task_results).

    Args:
        task_result: Task result dictionary

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    required_fields = [
        "task_id",
        "samples_generated",
        "successful_samples",
        "failed_samples",
        "total_input_tokens",
        "total_output_tokens",
        "total_cost_usd",
        "avg_latency",
    ]

    # Check all required fields are present
    for field in required_fields:
        if field not in task_result:
            raise ValidationError(f"Missing required field: {field}")

    # Validate values
    if task_result["samples_generated"] <= 0:
        raise ValidationError(f"Invalid samples_generated in task {task_result['task_id']}")

    total_samples = task_result["successful_samples"] + task_result["failed_samples"]
    if total_samples != task_result["samples_generated"]:
        raise ValidationError(
            f"Sample count mismatch in task {task_result['task_id']}: "
            f"generated={task_result['samples_generated']}, "
            f"successful={task_result['successful_samples']}, "
            f"failed={task_result['failed_samples']}"
        )

    if task_result["total_cost_usd"] < 0:
        raise ValidationError(f"Negative cost in task {task_result['task_id']}")

    return True


def validate_and_warn(obj: Any, obj_type: str = "result") -> bool:
    """
    Validate an object and log warnings instead of raising exceptions.

    This is useful for non-critical validation where you want to proceed
    even if there are issues.

    Args:
        obj: Object to validate
        obj_type: Type of object (for logging)

    Returns:
        True if valid, False if validation failed
    """
    # Import here to avoid circular import
    from .result_tracker import BenchmarkRun

    try:
        if isinstance(obj, ResponseMetrics):
            validate_response_metrics(obj)
        elif isinstance(obj, TaskMetrics):
            validate_task_metrics(obj)
        elif isinstance(obj, BenchmarkRun):
            validate_benchmark_run(obj)
        elif isinstance(obj, dict):
            validate_task_result_dict(obj)
        else:
            logger.warning(f"Unknown object type for validation: {type(obj)}")
            return False

        return True

    except ValidationError as e:
        logger.warning(f"Validation failed for {obj_type}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during validation of {obj_type}: {e}")
        return False
