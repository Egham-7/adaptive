"""
Result tracking and storage utilities for HumanEval benchmarks.

This module provides classes for collecting, aggregating, and persisting
benchmark results with full per-row cost and token tracking.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models.base import TaskMetrics
from .validators import validate_benchmark_run, validate_task_metrics, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRun:
    """
    Complete benchmark run results for a single model.

    This aggregates all task results and provides summary statistics.
    """

    model_name: str
    timestamp: str
    overall_score: float  # Pass@k score from HumanEval
    pass_at_k: dict[str, Any]  # {"k": 10, "score": 0.847}
    total_tasks: int
    total_samples: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    avg_latency_seconds: float
    success_rate: float
    per_task_results: list[dict[str, Any]]

    # Optional: Adaptive-specific stats
    model_selection_stats: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all tasks."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_cost_per_task(self) -> float:
        """Average cost per task."""
        if self.total_tasks == 0:
            return 0.0
        return self.total_cost_usd / self.total_tasks

    @property
    def avg_cost_per_sample(self) -> float:
        """Average cost per sample."""
        if self.total_samples == 0:
            return 0.0
        return self.total_cost_usd / self.total_samples


class ResultTracker:
    """
    Tracks and stores benchmark results with detailed metrics.

    This class collects task-level metrics during benchmark execution
    and provides methods to save results in various formats.
    """

    def __init__(self, model_name: str, output_dir: str = "results"):
        """
        Initialize result tracker.

        Args:
            model_name: Name of the model being benchmarked
            output_dir: Directory to save results
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().isoformat()
        self.task_metrics: list[TaskMetrics] = []

        # Summary statistics (will be calculated at finalization)
        self.overall_score: float | None = None
        self.pass_at_k: dict[str, Any] = {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized ResultTracker for {model_name}")

    def add_task_result(self, task_metrics: TaskMetrics):
        """
        Add metrics from a completed task.

        Args:
            task_metrics: TaskMetrics object from a HumanEval task

        Raises:
            ValidationError: If task_metrics validation fails
        """
        # Validate task metrics before adding
        try:
            validate_task_metrics(task_metrics)
        except ValidationError as e:
            logger.error(f"Task metrics validation failed for {task_metrics.task_id}: {e}")
            # Still add it but log the error
            # In production, you might want to raise the error instead

        self.task_metrics.append(task_metrics)
        logger.debug(f"Added result for task {task_metrics.task_id}")

    def set_benchmark_score(self, overall_score: float, k: int = 10):
        """
        Set the overall HumanEval benchmark score.

        Args:
            overall_score: Pass@k score from HumanEval
            k: The k value used (default: 10)
        """
        self.overall_score = overall_score
        self.pass_at_k = {"k": k, "score": overall_score}
        logger.info(f"Benchmark score set: pass@{k} = {overall_score:.4f}")

    def finalize(
        self, model_selection_stats: dict[str, Any] | None = None
    ) -> BenchmarkRun:
        """
        Finalize tracking and create complete benchmark run object.

        Args:
            model_selection_stats: Optional statistics for Adaptive routing

        Returns:
            BenchmarkRun object with all aggregated results
        """
        # Calculate aggregate statistics
        total_tasks = len(self.task_metrics)
        total_samples = sum(tm.samples_generated for tm in self.task_metrics)
        total_input_tokens = sum(tm.total_input_tokens for tm in self.task_metrics)
        total_output_tokens = sum(tm.total_output_tokens for tm in self.task_metrics)
        total_cost = sum(tm.total_cost_usd for tm in self.task_metrics)

        # Calculate average latency (weighted by successful samples)
        total_latency = sum(
            tm.avg_latency * tm.successful_samples for tm in self.task_metrics
        )
        total_successful = sum(tm.successful_samples for tm in self.task_metrics)
        avg_latency = total_latency / total_successful if total_successful > 0 else 0

        # Calculate success rate
        total_attempted = sum(
            tm.successful_samples + tm.failed_samples for tm in self.task_metrics
        )
        success_rate = total_successful / total_attempted if total_attempted > 0 else 0

        # Convert task metrics to dictionaries
        per_task_results = [tm.to_dict() for tm in self.task_metrics]

        # Create benchmark run object
        benchmark_run = BenchmarkRun(
            model_name=self.model_name,
            timestamp=self.timestamp,
            overall_score=self.overall_score or 0.0,
            pass_at_k=self.pass_at_k or {"k": 10, "score": 0.0},
            total_tasks=total_tasks,
            total_samples=total_samples,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost_usd=round(total_cost, 6),
            avg_latency_seconds=round(avg_latency, 3),
            success_rate=round(success_rate, 3),
            per_task_results=per_task_results,
            model_selection_stats=model_selection_stats,
        )

        logger.info(
            f"Finalized benchmark run: "
            f"{total_tasks} tasks, "
            f"{total_samples} samples, "
            f"${total_cost:.4f}, "
            f"pass@{self.pass_at_k.get('k', 10)}={self.overall_score:.4f}"
        )

        return benchmark_run

    def save_json(
        self, benchmark_run: BenchmarkRun, filename: str | None = None
    ) -> Path:
        """
        Save benchmark results to JSON file.

        Args:
            benchmark_run: BenchmarkRun object to save
            filename: Optional custom filename (default: auto-generated)

        Returns:
            Path to saved file

        Raises:
            ValidationError: If benchmark_run validation fails
        """
        # Validate before saving
        logger.info("Validating benchmark results...")
        validate_benchmark_run(benchmark_run)

        if filename is None:
            # Auto-generate filename with timestamp
            safe_model_name = self.model_name.replace(":", "_").replace("/", "_")
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_model_name}_{timestamp_str}.json"

        filepath = self.output_dir / filename

        # Save to JSON
        with open(filepath, "w") as f:
            json.dump(benchmark_run.to_dict(), f, indent=2)

        logger.info(f"Saved benchmark results to: {filepath}")
        return filepath

    def save_summary_json(
        self, benchmark_run: BenchmarkRun, filename: str | None = None
    ) -> Path:
        """
        Save a summary version (without per-sample costs) to JSON.

        Args:
            benchmark_run: BenchmarkRun object to save
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            safe_model_name = self.model_name.replace(":", "_").replace("/", "_")
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_model_name}_{timestamp_str}_summary.json"

        filepath = self.output_dir / filename

        # Create summary (exclude per-sample costs for smaller file)
        summary = benchmark_run.to_dict()
        for task in summary.get("per_task_results", []):
            task.pop("per_sample_costs", None)  # Remove detailed cost arrays

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved summary results to: {filepath}")
        return filepath

    def save_csv_summary(
        self, benchmark_run: BenchmarkRun, filename: str | None = None
    ) -> Path:
        """
        Save task-level summary to CSV file.

        Args:
            benchmark_run: BenchmarkRun object to save
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas not installed, skipping CSV export")
            return None

        if filename is None:
            safe_model_name = self.model_name.replace(":", "_").replace("/", "_")
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_model_name}_{timestamp_str}.csv"

        filepath = self.output_dir / filename

        # Convert per-task results to DataFrame
        df = pd.DataFrame(benchmark_run.per_task_results)

        # Drop per_sample_costs column if it exists (too verbose for CSV)
        if "per_sample_costs" in df.columns:
            df = df.drop(columns=["per_sample_costs"])

        # Save to CSV
        df.to_csv(filepath, index=False)

        logger.info(f"Saved CSV summary to: {filepath}")
        return filepath

    @classmethod
    def load_from_json(cls, filepath: str) -> BenchmarkRun:
        """
        Load benchmark results from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            BenchmarkRun object
        """
        with open(filepath) as f:
            data = json.load(f)

        return BenchmarkRun(**data)

    def __repr__(self) -> str:
        return (
            f"ResultTracker(model={self.model_name}, "
            f"tasks={len(self.task_metrics)})"
        )
