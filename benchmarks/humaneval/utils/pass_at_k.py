"""
Manual pass@k calculation utilities.

This module provides functions to calculate pass@k metrics manually when DeepEval
fails due to pandas DataFrame bugs.

Note: This is a workaround for a known DeepEval issue where the internal DataFrame
creation fails with column count mismatches.
"""

import logging
from math import comb
from typing import Any

logger = logging.getLogger(__name__)


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric using the unbiased estimator.

    This implements the formula from the HumanEval paper:
    pass@k = E[1 - C(n-c, k) / C(n, k)]

    Where:
        n = total samples generated per task
        c = number of correct samples
        k = number of samples to consider

    Args:
        n: Total number of samples generated
        c: Number of correct samples (passed all tests)
        k: k value for pass@k metric

    Returns:
        pass@k score (probability that at least 1 of k samples passes)

    Examples:
        >>> calculate_pass_at_k(200, 150, 10)  # 150/200 correct, pass@10
        0.9999...

        >>> calculate_pass_at_k(200, 10, 10)   # 10/200 correct, pass@10
        0.401...

        >>> calculate_pass_at_k(200, 0, 10)    # 0/200 correct
        0.0
    """
    if n < k:
        logger.warning(f"n={n} < k={k}, returning 0.0")
        return 0.0

    if c == 0:
        return 0.0

    if c >= k:
        # If we have k or more correct samples, we're guaranteed to get at least one
        return 1.0

    if n - c < k:
        # If we don't have enough incorrect samples to fill k slots, guaranteed success
        return 1.0

    # Calculate using the unbiased estimator
    # pass@k = 1 - (ways to choose k from n-c) / (ways to choose k from n)
    try:
        numerator = comb(n - c, k)
        denominator = comb(n, k)
        return 1.0 - (numerator / denominator)
    except (ValueError, OverflowError) as e:
        logger.error(f"Error calculating pass@k(n={n}, c={c}, k={k}): {e}")
        return 0.0


def extract_task_results_from_benchmark(benchmark: Any) -> list[dict[str, Any]]:
    """
    Extract task-level pass/fail results from DeepEval benchmark object.

    This function attempts to access DeepEval's internal state to get test results
    even when the benchmark.evaluate() method fails due to pandas bug.

    Args:
        benchmark: DeepEval HumanEval benchmark object

    Returns:
        List of task results with pass/fail information

    Note:
        This relies on DeepEval's internal structure and may break with version changes.
    """
    results = []

    try:
        # Log ALL available attributes for debugging
        attrs = [a for a in dir(benchmark) if not a.startswith('_')]
        logger.info(f"=== DeepEval Benchmark Object Debug ===")
        logger.info(f"Total attributes: {len(attrs)}")
        logger.info(f"All attributes: {', '.join(attrs)}")

        # Log key attribute values
        for attr in ['overall_score', 'task_scores', 'scores', 'n', 'k', 'tasks', 'test_cases', 'predictions']:
            if hasattr(benchmark, attr):
                value = getattr(benchmark, attr)
                logger.info(f"  {attr} = {value} (type: {type(value).__name__})")
            else:
                logger.info(f"  {attr} = NOT FOUND")
        logger.info(f"=====================================")

        # Method 0: Try task_scores attribute (DeepEval internal - most reliable after error)
        if hasattr(benchmark, "task_scores") and benchmark.task_scores is not None:
            logger.info(f"Method 0: Found 'task_scores' attribute with type: {type(benchmark.task_scores)}")
            if isinstance(benchmark.task_scores, dict):
                logger.info(f"Method 0: task_scores dict has {len(benchmark.task_scores)} items")
                n = getattr(benchmark, "n", 50)
                for task_id, score in benchmark.task_scores.items():
                    # task_scores maps task -> score (0.0 to 1.0)
                    correct = int(score * n) if score > 0 else 0
                    logger.info(f"Method 0: Task {task_id}: score={score}, n={n}, correct={correct}")
                    results.append({
                        "task_id": f"HumanEval/{task_id}" if not task_id.startswith("HumanEval") else task_id,
                        "total_samples": n,
                        "correct_samples": correct,
                    })
            else:
                logger.info(f"Method 0: task_scores is not a dict, it's {type(benchmark.task_scores)}")

        # Method 1: Try scores attribute (DeepEval v1.2+)
        if not results and hasattr(benchmark, "scores") and benchmark.scores is not None:
            logger.info(f"Method 1: Found 'scores' attribute with type: {type(benchmark.scores)}")
            # Scores is usually a dict mapping task -> score
            if isinstance(benchmark.scores, dict):
                logger.info(f"Method 1: scores dict has {len(benchmark.scores)} items")
                for task_id, score in benchmark.scores.items():
                    # Score of 1.0 means all samples passed
                    # We need to get n from the benchmark config
                    n = getattr(benchmark, "n", 50)  # Default 50
                    correct = int(score * n) if score > 0 else 0
                    logger.info(f"Method 1: Task {task_id}: score={score}, n={n}, correct={correct}")
                    results.append({
                        "task_id": f"HumanEval/{task_id}" if not task_id.startswith("HumanEval") else task_id,
                        "total_samples": n,
                        "correct_samples": correct,
                    })
            else:
                logger.info(f"Method 1: scores is not a dict, it's {type(benchmark.scores)}")

        # Method 2: Try overall_score with tasks
        if not results and hasattr(benchmark, "overall_score") and hasattr(benchmark, "tasks"):
            logger.info("Method 2: Using overall_score with tasks")
            # If there's an overall score, assume all tasks have same performance
            n = getattr(benchmark, "n", 50)
            score = benchmark.overall_score
            logger.info(f"Method 2: overall_score={score}, n={n}, tasks count={len(benchmark.tasks)}")
            for task in benchmark.tasks:
                task_name = task.value if hasattr(task, "value") else str(task)
                correct = int(score * n)
                logger.info(f"Method 2: Task {task_name}: score={score}, correct={correct}")
                results.append({
                    "task_id": f"HumanEval/{task_name}",
                    "total_samples": n,
                    "correct_samples": correct,
                })

        # Method 3: Try test_cases attribute
        if not results and hasattr(benchmark, "test_cases"):
            logger.debug("Found 'test_cases' attribute")
            for test_case in benchmark.test_cases:
                task_id = getattr(test_case, "task_id", None) or getattr(test_case, "name", None)
                if hasattr(test_case, "predictions"):
                    predictions = test_case.predictions
                    correct_count = sum(1 for p in predictions if getattr(p, "passed", False))
                    results.append({
                        "task_id": task_id,
                        "total_samples": len(predictions),
                        "correct_samples": correct_count,
                    })

        # Method 4: Try predictions attribute directly
        if not results and hasattr(benchmark, "predictions") and benchmark.predictions:
            logger.debug("Found 'predictions' attribute")
            for task_id, predictions in benchmark.predictions.items():
                if predictions:
                    correct_count = sum(1 for p in predictions if p.get("passed", False))
                    results.append({
                        "task_id": task_id if task_id.startswith("HumanEval") else f"HumanEval/{task_id}",
                        "total_samples": len(predictions),
                        "correct_samples": correct_count,
                    })

        if results:
            logger.info(f"Extracted {len(results)} task results from benchmark")
        else:
            logger.warning("No results extracted - DeepEval structure may have changed")
            logger.warning(f"Available benchmark attributes: {', '.join(attrs)}")

    except Exception as e:
        logger.error(f"Failed to extract results from benchmark: {e}", exc_info=True)

    return results


def calculate_overall_pass_at_k(task_results: list[dict[str, Any]], k: int) -> float:
    """
    Calculate overall pass@k score across all tasks.

    Args:
        task_results: List of task results with 'total_samples' and 'correct_samples'
        k: k value for pass@k metric

    Returns:
        Overall pass@k score (average across all tasks)
    """
    if not task_results:
        logger.warning("No task results provided, returning 0.0")
        return 0.0

    task_scores = []
    for task in task_results:
        n = task.get("total_samples", 0)
        c = task.get("correct_samples", 0)

        if n == 0:
            logger.warning(f"Task {task.get('task_id')} has 0 samples, skipping")
            continue

        score = calculate_pass_at_k(n, c, k)
        task_scores.append(score)

    if not task_scores:
        logger.warning("No valid task scores calculated, returning 0.0")
        return 0.0

    overall_score = sum(task_scores) / len(task_scores)
    logger.info(
        f"Calculated pass@{k} = {overall_score:.4f} across {len(task_scores)} tasks"
    )

    return overall_score
