"""
Cross-model comparison tests for HumanEval benchmarks.

This module loads previously saved benchmark results and generates
comprehensive comparison reports across Claude, GLM, and Adaptive models.

Run with: pytest benchmarks/humaneval/tests/test_comparison.py -v
"""

import logging
from pathlib import Path

import pytest

from ..config import get_humaneval_config
from ..utils import (
    BenchmarkRun,
    ResultTracker,
    compare_benchmarks,
    generate_cost_breakdown,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestModelComparison:
    """Tests for comparing multiple model benchmark results."""

    @pytest.fixture(scope="class")
    def config(self):
        """Get HumanEval configuration."""
        return get_humaneval_config()

    @pytest.fixture(scope="class")
    def results_dir(self, config):
        """Get base results directory."""
        return Path(config.results_folder)

    def load_latest_results(self, results_dir: Path, model_name: str) -> BenchmarkRun:
        """
        Load the latest benchmark results for a model.

        Args:
            results_dir: Base results directory
            model_name: Name of model subdirectory

        Returns:
            BenchmarkRun object
        """
        model_dir = results_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"No results found for {model_name} in {model_dir}")

        # Find latest JSON file (not summary)
        json_files = sorted(
            [f for f in model_dir.glob("*.json") if "_summary" not in f.name],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if not json_files:
            raise FileNotFoundError(f"No benchmark results found in {model_dir}")

        latest_file = json_files[0]
        logger.info(f"Loading {model_name} results from: {latest_file.name}")

        return ResultTracker.load_from_json(str(latest_file))

    def test_load_all_results(self, results_dir):
        """
        Test loading results from all three models.

        This test verifies that benchmark results exist for Claude, GLM,
        and Adaptive models.
        """
        logger.info("Attempting to load results for all models...")

        results = {}
        models = ["claude", "glm", "adaptive"]

        for model_name in models:
            try:
                result = self.load_latest_results(results_dir, model_name)
                results[model_name] = result
                logger.info(
                    f"✓ Loaded {model_name}: "
                    f"pass@{result.pass_at_k['k']}={result.overall_score:.4f}, "
                    f"cost=${result.total_cost_usd:.4f}"
                )
            except FileNotFoundError as e:
                logger.warning(f"✗ {model_name}: {e}")
                pytest.skip(f"Skipping comparison - {model_name} results not found")

        assert len(results) > 0, "At least one model should have results"
        logger.info(f"Successfully loaded results for {len(results)} models")

    def test_generate_comparison_report(self, results_dir):
        """
        Generate comprehensive comparison report for all models.

        This test loads all available results and generates a comparison
        report showing performance, cost, and efficiency metrics.
        """
        logger.info("=" * 70)
        logger.info("Generating Model Comparison Report")
        logger.info("=" * 70)

        # Load all available results
        benchmark_runs = []
        models = ["claude", "glm", "adaptive"]

        for model_name in models:
            try:
                result = self.load_latest_results(results_dir, model_name)
                benchmark_runs.append(result)
                logger.info(f"✓ Loaded {model_name} results")
            except FileNotFoundError:
                logger.warning(f"✗ Skipping {model_name} - no results found")

        if len(benchmark_runs) < 2:
            pytest.skip("Need at least 2 models with results to generate comparison")

        # Generate comparison report
        comparison_dir = results_dir / "comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = comparison_dir / f"comparison_{timestamp}.md"

        logger.info("\nGenerating comparison report...")
        report = compare_benchmarks(
            benchmark_runs=benchmark_runs, output_path=report_path
        )

        logger.info(f"Comparison report saved to: {report_path}")

        # Print report to console
        print("\n" + report)

        # Assertions
        assert report_path.exists(), "Comparison report should be created"
        assert len(report) > 0, "Report should have content"

        logger.info("=" * 70)
        logger.info("Comparison report generated successfully!")
        logger.info("=" * 70)

    def test_cost_analysis(self, results_dir):
        """
        Analyze cost breakdown for each model.

        This test generates detailed cost analysis showing which tasks
        are most expensive for each model.
        """
        logger.info("=" * 70)
        logger.info("Cost Analysis")
        logger.info("=" * 70)

        models = ["claude", "glm", "adaptive"]

        for model_name in models:
            try:
                result = self.load_latest_results(results_dir, model_name)

                logger.info(f"\n{model_name.upper()} Cost Breakdown:")
                logger.info("-" * 50)

                # Generate cost breakdown
                breakdown = generate_cost_breakdown(result, top_n=10)

                logger.info(f"Total Cost: ${breakdown['total_cost']:.4f}")
                logger.info(f"Avg per Task: ${breakdown['avg_cost_per_task']:.6f}")
                logger.info(
                    f"Top {breakdown['top_n_tasks']} tasks: "
                    f"${breakdown['top_n_cost']:.4f} "
                    f"({breakdown['top_n_percentage']:.2f}%)"
                )

                logger.info("\nMost expensive tasks:")
                for i, task in enumerate(breakdown["top_tasks"][:5], 1):
                    logger.info(
                        f"  {i}. {task['task_id']}: "
                        f"${task['cost']:.6f} "
                        f"({task['tokens']:,} tokens)"
                    )

            except FileNotFoundError:
                logger.warning(f"✗ Skipping {model_name} - no results found")

        logger.info("\n" + "=" * 70)

    def test_performance_summary(self, results_dir):
        """
        Generate performance summary across all models.

        Shows pass@k scores, success rates, and latency metrics.
        """
        logger.info("=" * 70)
        logger.info("Performance Summary")
        logger.info("=" * 70)

        models = ["claude", "glm", "adaptive"]
        summary_data = []

        for model_name in models:
            try:
                result = self.load_latest_results(results_dir, model_name)
                summary_data.append(
                    {
                        "model": model_name,
                        "score": result.overall_score,
                        "k": result.pass_at_k["k"],
                        "success_rate": result.success_rate,
                        "avg_latency": result.avg_latency_seconds,
                        "total_cost": result.total_cost_usd,
                    }
                )
            except FileNotFoundError:
                continue

        if not summary_data:
            pytest.skip("No benchmark results found")

        # Sort by score
        summary_data.sort(key=lambda x: x["score"], reverse=True)

        print(
            "\n| Rank | Model | Pass@k | Score | Success Rate | Avg Latency | Total Cost |"
        )
        print(
            "|------|-------|--------|-------|--------------|-------------|------------|"
        )

        for i, data in enumerate(summary_data, 1):
            print(
                f"| {i} | "
                f"{data['model']:10s} | "
                f"pass@{data['k']} | "
                f"{data['score']:.4f} | "
                f"{data['success_rate']:.2%} | "
                f"{data['avg_latency']:.3f}s | "
                f"${data['total_cost']:.4f} |"
            )

        logger.info("\n" + "=" * 70)

        # Identify best model by different criteria
        best_performance = max(summary_data, key=lambda x: x["score"])
        best_cost = min(summary_data, key=lambda x: x["total_cost"])
        best_efficiency = max(
            summary_data,
            key=lambda x: x["score"] / x["total_cost"] if x["total_cost"] > 0 else 0,
        )

        logger.info("\nBest Models by Criteria:")
        logger.info(
            f"  Performance: {best_performance['model']} (score: {best_performance['score']:.4f})"
        )
        logger.info(
            f"  Cost: {best_cost['model']} (cost: ${best_cost['total_cost']:.4f})"
        )
        logger.info(
            f"  Efficiency: {best_efficiency['model']} (score/cost: {best_efficiency['score']/best_efficiency['total_cost']:.2f})"
        )
        logger.info("=" * 70)


if __name__ == "__main__":
    # Allow running directly with pytest
    pytest.main([__file__, "-v", "-s"])
