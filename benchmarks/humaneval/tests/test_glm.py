"""
HumanEval benchmark tests for GLM-4.6.

Run with: deepeval test run benchmarks/humaneval/tests/test_glm.py
Or pytest: pytest benchmarks/humaneval/tests/test_glm.py -v
"""

import logging

import pytest
from deepeval.benchmarks import HumanEval

from ..config import get_glm_config, get_humaneval_config
from ..models import GLMForDeepEval
from ..utils import ResultTracker, generate_markdown_report, print_console_summary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestGLMHumanEval:
    """HumanEval benchmark tests for GLM model."""

    @pytest.fixture(scope="class")
    def config(self):
        """Get HumanEval configuration."""
        return get_humaneval_config()

    @pytest.fixture(scope="class")
    def glm_config(self):
        """Get and validate GLM configuration."""
        config = get_glm_config()
        config.validate()
        return config

    @pytest.fixture(scope="class")
    def model(self, glm_config):
        """Initialize GLM model."""
        logger.info(f"Initializing GLM model: {glm_config.model_name}")
        return GLMForDeepEval(
            model_name=glm_config.model_name,
            api_key=glm_config.api_key,
            api_base=glm_config.api_base,
        )

    @pytest.fixture(scope="class")
    def result_tracker(self, config, glm_config):
        """Initialize result tracker."""
        results_dir = config.get_results_path("glm")
        return ResultTracker(
            model_name=glm_config.model_name, output_dir=str(results_dir)
        )

    def test_glm_humaneval_full(self, model, config, result_tracker):
        """
        Run full HumanEval benchmark on GLM (all 164 tasks).

        This test generates n=200 samples per task and evaluates with pass@10.
        Expected runtime: 2-4 hours depending on API speed.
        Expected cost: Varies based on GLM pricing.
        """
        logger.info("=" * 70)
        logger.info("Starting full HumanEval benchmark for GLM")
        logger.info(
            f"Tasks: 164, Samples per task: {config.n_samples}, Pass@k: {config.k_value}"
        )
        logger.info("=" * 70)

        # Create HumanEval benchmark (all 164 tasks)
        benchmark = HumanEval(n=config.n_samples)

        # Run evaluation
        logger.info("Running benchmark evaluation...")
        benchmark.evaluate(model=model, k=config.k_value)

        # Get overall score
        overall_score = benchmark.overall_score
        logger.info(f"Benchmark completed! Pass@{config.k_value}: {overall_score:.4f}")

        # Collect all task metrics
        all_task_metrics = model.get_all_task_metrics()
        logger.info(f"Collected metrics for {len(all_task_metrics)} tasks")

        # Add metrics to result tracker
        for task_metrics in all_task_metrics:
            result_tracker.add_task_result(task_metrics)

        # Set benchmark score
        result_tracker.set_benchmark_score(overall_score, k=config.k_value)

        # Finalize results
        benchmark_run = result_tracker.finalize()

        # Save results
        json_path = result_tracker.save_json(benchmark_run)
        summary_path = result_tracker.save_summary_json(benchmark_run)
        csv_path = result_tracker.save_csv_summary(benchmark_run)

        # Generate and save markdown report
        report_path = json_path.parent / f"{json_path.stem}_report.md"
        generate_markdown_report(benchmark_run, output_path=report_path)

        # Print console summary
        print_console_summary(benchmark_run)

        # Log file locations
        logger.info("Results saved to:")
        logger.info(f"  - JSON (detailed): {json_path}")
        logger.info(f"  - JSON (summary):  {summary_path}")
        logger.info(f"  - CSV:             {csv_path}")
        logger.info(f"  - Report (MD):     {report_path}")

        # Assertions
        assert overall_score > 0.0, "Benchmark score should be greater than 0"
        assert len(all_task_metrics) == 164, "Should have metrics for all 164 tasks"
        assert benchmark_run.total_cost_usd > 0, "Total cost should be tracked"

        logger.info("=" * 70)
        logger.info("GLM HumanEval benchmark completed successfully!")
        logger.info("=" * 70)

    def test_glm_humaneval_quick(self, model, config):
        """
        Quick test with a subset of tasks for development/testing.

        This test runs only 5 tasks with 50 samples each for rapid testing.
        Expected runtime: ~5-10 minutes.
        """
        from deepeval.benchmarks.tasks import HumanEvalTask

        logger.info("Starting quick HumanEval test for GLM (5 tasks)")

        # Create benchmark with subset of tasks
        benchmark = HumanEval(
            tasks=[
                HumanEvalTask.SORT_NUMBERS,
                HumanEvalTask.STRLEN,
                HumanEvalTask.SUM_PRODUCT,
                HumanEvalTask.EVEN_ODD_COUNT,
                HumanEvalTask.FIB,
            ],
            n=50,  # Fewer samples for faster testing
        )

        # Run evaluation
        benchmark.evaluate(model=model, k=5)

        # Get results
        overall_score = benchmark.overall_score
        total_cost = model.get_total_cost()
        total_tokens = model.get_total_tokens()

        logger.info("Quick test completed!")
        logger.info(f"  Pass@5: {overall_score:.4f}")
        logger.info(f"  Total cost: ${total_cost:.4f}")
        logger.info(f"  Total tokens: {total_tokens:,}")

        # Assertions
        assert overall_score > 0.0, "Should have some passing samples"
        assert total_cost > 0, "Should track costs"

        # Clear metrics for next test
        model.clear_metrics()


if __name__ == "__main__":
    # Allow running directly with pytest
    pytest.main([__file__, "-v", "-s"])
