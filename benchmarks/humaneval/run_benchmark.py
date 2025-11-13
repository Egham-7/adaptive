#!/usr/bin/env python3
"""
CLI entry point for running HumanEval benchmarks.

This script provides a convenient command-line interface for running
benchmarks on Claude, GLM, and Adaptive models with full cost tracking.

Usage:
    python run_benchmark.py claude --quick          # Quick test with Claude
    python run_benchmark.py glm --full              # Full benchmark with GLM
    python run_benchmark.py adaptive --full         # Full benchmark with Adaptive
    python run_benchmark.py all --quick             # Quick test on all models
    python run_benchmark.py compare                 # Compare existing results
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepeval.benchmarks import HumanEval
from deepeval.benchmarks.tasks import HumanEvalTask

from humaneval.config import BenchmarkSettings
from humaneval.models import AdaptiveForDeepEval, ClaudeForDeepEval, GLMForDeepEval
from humaneval.utils import (
    ResultTracker,
    calculate_overall_pass_at_k,
    compare_benchmarks,
    extract_task_results_from_benchmark,
    generate_markdown_report,
    print_console_summary,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_claude_benchmark(quick: bool = False):
    """Run HumanEval benchmark on Claude."""
    settings = BenchmarkSettings()
    settings.print_summary()

    # Validate configuration
    settings.validate_model("claude")

    # Initialize model
    logger.info("Initializing Claude model...")
    model = ClaudeForDeepEval(
        model_name=settings.claude.model_name, api_key=settings.claude.api_key
    )

    # Create result tracker
    results_dir = settings.humaneval.get_results_path("claude")
    result_tracker = ResultTracker(
        model_name=settings.claude.model_name, output_dir=str(results_dir)
    )

    if quick:
        logger.info("Running quick test (1 task, 50 samples)...")
        benchmark = HumanEval(
            tasks=[
                HumanEvalTask.SORT_NUMBERS,
            ],
            n=50,
        )
        k = 1
    else:
        logger.info(
            f"Running full benchmark (164 tasks, {settings.humaneval.n_samples} samples each)..."
        )
        benchmark = HumanEval(n=settings.humaneval.n_samples)
        k = settings.humaneval.k_value

    # Run evaluation
    logger.info("Running benchmark evaluation...")
    overall_score = 0.0
    try:
        benchmark.evaluate(model=model, k=k)
        # Try to get overall_score - it may be available even if evaluate() throws error later
        if hasattr(benchmark, 'overall_score') and benchmark.overall_score is not None:
            overall_score = benchmark.overall_score
            logger.info(f"Benchmark complete: pass@{k} = {overall_score:.4f}")
    except (ValueError, AssertionError) as e:
        # DeepEval has a pandas DataFrame bug in some versions
        # The evaluation completes but crashes during DataFrame creation
        # But the overall_score is already calculated before the crash
        if "columns passed" in str(e) or "DataFrame" in str(e):
            logger.warning(f"DeepEval DataFrame error (known bug): {e}")

            # First try: check if overall_score was set before the error
            if hasattr(benchmark, 'overall_score') and benchmark.overall_score is not None:
                overall_score = benchmark.overall_score
                logger.info(f"✓ Successfully retrieved pass@{k} = {overall_score:.4f} from benchmark object")
            else:
                # Fallback: extract results manually
                logger.info("Falling back to manual pass@k calculation...")
                try:
                    task_results = extract_task_results_from_benchmark(benchmark)
                    if task_results:
                        overall_score = calculate_overall_pass_at_k(task_results, k)
                        logger.info(f"✓ Manually calculated pass@{k} = {overall_score:.4f}")
                    else:
                        logger.error("Could not extract test results from benchmark")
                        overall_score = 0.0
                except Exception as calc_error:
                    logger.error(f"Manual pass@k calculation failed: {calc_error}")
                    overall_score = 0.0
        else:
            # Some other error - re-raise it
            raise

    # Process results
    all_task_metrics = model.get_all_task_metrics()

    for task_metrics in all_task_metrics:
        result_tracker.add_task_result(task_metrics)

    result_tracker.set_benchmark_score(overall_score, k=k)
    benchmark_run = result_tracker.finalize()

    # Save results
    json_path = result_tracker.save_json(benchmark_run)
    result_tracker.save_summary_json(benchmark_run)
    result_tracker.save_csv_summary(benchmark_run)

    report_path = json_path.parent / f"{json_path.stem}_report.md"
    generate_markdown_report(benchmark_run, output_path=report_path)

    print_console_summary(benchmark_run)

    logger.info(f"Results saved to: {json_path.parent}")


def run_glm_benchmark(quick: bool = False):
    """Run HumanEval benchmark on GLM."""
    settings = BenchmarkSettings()
    settings.print_summary()

    # Validate configuration
    settings.validate_model("glm")

    # Initialize model
    logger.info("Initializing GLM model...")
    model = GLMForDeepEval(
        model_name=settings.glm.model_name,
        api_key=settings.glm.api_key,
        api_base=settings.glm.api_base,
    )

    # Create result tracker
    results_dir = settings.humaneval.get_results_path("glm")
    result_tracker = ResultTracker(
        model_name=settings.glm.model_name, output_dir=str(results_dir)
    )

    if quick:
        logger.info("Running quick test (1 task, 50 samples)...")
        benchmark = HumanEval(
            tasks=[
                HumanEvalTask.SORT_NUMBERS,
            ],
            n=50,
        )
        k = 1
    else:
        logger.info(
            f"Running full benchmark (164 tasks, {settings.humaneval.n_samples} samples each)..."
        )
        benchmark = HumanEval(n=settings.humaneval.n_samples)
        k = settings.humaneval.k_value

    # Run evaluation
    logger.info("Running benchmark evaluation...")
    overall_score = 0.0
    try:
        benchmark.evaluate(model=model, k=k)
        # Try to get overall_score - it may be available even if evaluate() throws error later
        if hasattr(benchmark, 'overall_score') and benchmark.overall_score is not None:
            overall_score = benchmark.overall_score
            logger.info(f"Benchmark complete: pass@{k} = {overall_score:.4f}")
    except (ValueError, AssertionError) as e:
        # DeepEval has a pandas DataFrame bug in some versions
        # The evaluation completes but crashes during DataFrame creation
        # But the overall_score is already calculated before the crash
        if "columns passed" in str(e) or "DataFrame" in str(e):
            logger.warning(f"DeepEval DataFrame error (known bug): {e}")

            # First try: check if overall_score was set before the error
            if hasattr(benchmark, 'overall_score') and benchmark.overall_score is not None:
                overall_score = benchmark.overall_score
                logger.info(f"✓ Successfully retrieved pass@{k} = {overall_score:.4f} from benchmark object")
            else:
                # Fallback: extract results manually
                logger.info("Falling back to manual pass@k calculation...")
                try:
                    task_results = extract_task_results_from_benchmark(benchmark)
                    if task_results:
                        overall_score = calculate_overall_pass_at_k(task_results, k)
                        logger.info(f"✓ Manually calculated pass@{k} = {overall_score:.4f}")
                    else:
                        logger.error("Could not extract test results from benchmark")
                        overall_score = 0.0
                except Exception as calc_error:
                    logger.error(f"Manual pass@k calculation failed: {calc_error}")
                    overall_score = 0.0
        else:
            # Some other error - re-raise it
            raise

    # Process results
    all_task_metrics = model.get_all_task_metrics()

    for task_metrics in all_task_metrics:
        result_tracker.add_task_result(task_metrics)

    result_tracker.set_benchmark_score(overall_score, k=k)
    benchmark_run = result_tracker.finalize()

    # Save results
    json_path = result_tracker.save_json(benchmark_run)
    result_tracker.save_summary_json(benchmark_run)
    result_tracker.save_csv_summary(benchmark_run)

    report_path = json_path.parent / f"{json_path.stem}_report.md"
    generate_markdown_report(benchmark_run, output_path=report_path)

    print_console_summary(benchmark_run)

    logger.info(f"Results saved to: {json_path.parent}")


def run_adaptive_benchmark(quick: bool = False):
    """Run HumanEval benchmark on Adaptive routing."""
    settings = BenchmarkSettings()
    settings.print_summary()

    # Validate configuration
    settings.validate_model("adaptive")

    # Initialize model
    logger.info("Initializing Adaptive routing...")
    model = AdaptiveForDeepEval(
        api_key=settings.adaptive.api_key,
        api_base=settings.adaptive.api_base,
        models=settings.adaptive.models,
        cost_bias=settings.adaptive.cost_bias,
    )

    # Create result tracker
    results_dir = settings.humaneval.get_results_path("adaptive")
    result_tracker = ResultTracker(
        model_name="adaptive-router", output_dir=str(results_dir)
    )

    if quick:
        logger.info("Running quick test (1 task, 50 samples)...")
        benchmark = HumanEval(
            tasks=[
                HumanEvalTask.SORT_NUMBERS,
            ],
            n=50,
        )
        k = 1
    else:
        logger.info(
            f"Running full benchmark (164 tasks, {settings.humaneval.n_samples} samples each)..."
        )
        benchmark = HumanEval(n=settings.humaneval.n_samples)
        k = settings.humaneval.k_value

    # Run evaluation
    logger.info("Running benchmark evaluation...")
    overall_score = 0.0
    try:
        benchmark.evaluate(model=model, k=k)
        # Try to get overall_score - it may be available even if evaluate() throws error later
        if hasattr(benchmark, 'overall_score') and benchmark.overall_score is not None:
            overall_score = benchmark.overall_score
            logger.info(f"Benchmark complete: pass@{k} = {overall_score:.4f}")
    except (ValueError, AssertionError) as e:
        # DeepEval has a pandas DataFrame bug in some versions
        # The evaluation completes but crashes during DataFrame creation
        # But the overall_score is already calculated before the crash
        if "columns passed" in str(e) or "DataFrame" in str(e):
            logger.warning(f"DeepEval DataFrame error (known bug): {e}")

            # First try: check if overall_score was set before the error
            if hasattr(benchmark, 'overall_score') and benchmark.overall_score is not None:
                overall_score = benchmark.overall_score
                logger.info(f"✓ Successfully retrieved pass@{k} = {overall_score:.4f} from benchmark object")
            else:
                # Fallback: extract results manually
                logger.info("Falling back to manual pass@k calculation...")
                try:
                    task_results = extract_task_results_from_benchmark(benchmark)
                    if task_results:
                        overall_score = calculate_overall_pass_at_k(task_results, k)
                        logger.info(f"✓ Manually calculated pass@{k} = {overall_score:.4f}")
                    else:
                        logger.error("Could not extract test results from benchmark")
                        overall_score = 0.0
                except Exception as calc_error:
                    logger.error(f"Manual pass@k calculation failed: {calc_error}")
                    overall_score = 0.0
        else:
            # Some other error - re-raise it
            raise

    # Process results
    all_task_metrics = model.get_all_task_metrics()
    model_selection_stats = model.get_model_selection_stats()

    for task_metrics in all_task_metrics:
        result_tracker.add_task_result(task_metrics)

    result_tracker.set_benchmark_score(overall_score, k=k)
    benchmark_run = result_tracker.finalize(model_selection_stats=model_selection_stats)

    # Save results
    json_path = result_tracker.save_json(benchmark_run)
    result_tracker.save_summary_json(benchmark_run)
    result_tracker.save_csv_summary(benchmark_run)

    report_path = json_path.parent / f"{json_path.stem}_report.md"
    generate_markdown_report(benchmark_run, output_path=report_path)

    print_console_summary(benchmark_run)

    logger.info(f"Results saved to: {json_path.parent}")


def run_comparison():
    """Compare existing benchmark results."""
    settings = BenchmarkSettings()
    results_dir = Path(settings.humaneval.results_folder)

    logger.info("Loading benchmark results...")

    benchmark_runs = []
    for model_name in ["claude", "glm", "adaptive"]:
        model_dir = results_dir / model_name
        if not model_dir.exists():
            logger.warning(f"No results found for {model_name}")
            continue

        # Find latest JSON file
        json_files = sorted(
            [f for f in model_dir.glob("*.json") if "_summary" not in f.name],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if json_files:
            logger.info(f"Loading {model_name} results from: {json_files[0].name}")
            benchmark_runs.append(ResultTracker.load_from_json(str(json_files[0])))
        else:
            logger.warning(f"No JSON results found for {model_name}")

    if len(benchmark_runs) < 2:
        logger.error("Need at least 2 models with results to generate comparison")
        sys.exit(1)

    # Generate comparison
    comparison_dir = results_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = comparison_dir / f"comparison_{timestamp}.md"

    logger.info("Generating comparison report...")
    report = compare_benchmarks(benchmark_runs, output_path=report_path)

    print("\n" + report)
    logger.info(f"Comparison report saved to: {report_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run HumanEval benchmarks with full cost tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s claude --quick        Run quick test on Claude
  %(prog)s glm --full            Run full benchmark on GLM
  %(prog)s adaptive --full       Run full benchmark on Adaptive
  %(prog)s all --quick           Run quick test on all models
  %(prog)s compare               Compare existing results
        """,
    )

    parser.add_argument(
        "model",
        choices=["claude", "glm", "adaptive", "all", "compare"],
        help="Model to benchmark or 'compare' to compare existing results",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test (1 task, 50 samples) instead of full benchmark",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark (164 tasks, 200 samples) - this is the default",
    )

    args = parser.parse_args()

    try:
        if args.model == "compare":
            run_comparison()
        elif args.model == "all":
            for model in ["claude", "glm", "adaptive"]:
                logger.info(f"\n{'=' * 70}")
                logger.info(f"Running benchmark for {model.upper()}")
                logger.info(f"{'=' * 70}\n")
                if model == "claude":
                    run_claude_benchmark(quick=args.quick)
                elif model == "glm":
                    run_glm_benchmark(quick=args.quick)
                elif model == "adaptive":
                    run_adaptive_benchmark(quick=args.quick)
        elif args.model == "claude":
            run_claude_benchmark(quick=args.quick)
        elif args.model == "glm":
            run_glm_benchmark(quick=args.quick)
        elif args.model == "adaptive":
            run_adaptive_benchmark(quick=args.quick)

    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running benchmark: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
