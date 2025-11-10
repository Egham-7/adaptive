"""
Reporting utilities for HumanEval benchmark results.

This module provides functions for generating reports, comparisons, and
visualizations of benchmark results across multiple models.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .result_tracker import BenchmarkRun

logger = logging.getLogger(__name__)


def generate_markdown_report(
    benchmark_run: BenchmarkRun, output_path: Path | None = None
) -> str:
    """
    Generate a markdown report for a single benchmark run.

    Args:
        benchmark_run: BenchmarkRun object
        output_path: Optional path to save report

    Returns:
        Markdown report as string
    """
    md = []

    # Header
    md.append(f"# HumanEval Benchmark Report: {benchmark_run.model_name}")
    md.append(f"\n**Timestamp:** {benchmark_run.timestamp}\n")

    # Overall Score
    md.append("## Overall Results\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    k = benchmark_run.pass_at_k.get("k", 10)
    score = benchmark_run.pass_at_k.get("score", 0.0)
    md.append(f"| **Pass@{k}** | **{score:.4f}** ({score*100:.2f}%) |")
    md.append(f"| Total Tasks | {benchmark_run.total_tasks} |")
    md.append(f"| Total Samples | {benchmark_run.total_samples:,} |")
    md.append(f"| Success Rate | {benchmark_run.success_rate:.2%} |")
    md.append("")

    # Cost and Performance
    md.append("## Cost & Performance\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| **Total Cost** | **${benchmark_run.total_cost_usd:.4f}** |")
    md.append(f"| Avg Cost per Task | ${benchmark_run.avg_cost_per_task:.6f} |")
    md.append(f"| Avg Cost per Sample | ${benchmark_run.avg_cost_per_sample:.6f} |")
    md.append(f"| Total Tokens | {benchmark_run.total_tokens:,} |")
    md.append(f"| Input Tokens | {benchmark_run.total_input_tokens:,} |")
    md.append(f"| Output Tokens | {benchmark_run.total_output_tokens:,} |")
    md.append(f"| Avg Latency | {benchmark_run.avg_latency_seconds:.3f}s |")
    md.append("")

    # Model Selection Stats (for Adaptive)
    if benchmark_run.model_selection_stats:
        md.append("## Model Selection (Adaptive Routing)\n")
        stats = benchmark_run.model_selection_stats
        md.append(f"**Total Requests:** {stats.get('total_requests', 0)}\n")
        md.append("| Model | Count | Percentage |")
        md.append("|-------|-------|------------|")
        for model, data in stats.get("models", {}).items():
            count = data.get("count", 0)
            pct = data.get("percentage", 0)
            md.append(f"| {model} | {count} | {pct:.2f}% |")
        md.append("")

    # Top 10 Most Expensive Tasks
    md.append("## Top 10 Most Expensive Tasks\n")
    tasks_by_cost = sorted(
        benchmark_run.per_task_results,
        key=lambda x: x.get("total_cost_usd", 0),
        reverse=True,
    )[:10]

    md.append("| Task ID | Cost (USD) | Input Tokens | Output Tokens | Avg Latency |")
    md.append("|---------|------------|--------------|---------------|-------------|")
    for task in tasks_by_cost:
        md.append(
            f"| {task['task_id']} | "
            f"${task['total_cost_usd']:.6f} | "
            f"{task['total_input_tokens']:,} | "
            f"{task['total_output_tokens']:,} | "
            f"{task['avg_latency']:.3f}s |"
        )
    md.append("")

    # Failed Tasks (if any)
    failed_tasks = [
        task
        for task in benchmark_run.per_task_results
        if task.get("failed_samples", 0) > 0
    ]
    if failed_tasks:
        md.append("## Failed Tasks\n")
        md.append("| Task ID | Failed Samples | Success Rate |")
        md.append("|---------|----------------|--------------|")
        for task in failed_tasks:
            success_rate = task.get("success_rate", 0)
            failed = task.get("failed_samples", 0)
            md.append(f"| {task['task_id']} | " f"{failed} | " f"{success_rate:.2%} |")
        md.append("")

    report = "\n".join(md)

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Saved markdown report to: {output_path}")

    return report


def compare_benchmarks(
    benchmark_runs: list[BenchmarkRun], output_path: Path | None = None
) -> str:
    """
    Generate a comparison report for multiple benchmark runs.

    Args:
        benchmark_runs: List of BenchmarkRun objects to compare
        output_path: Optional path to save comparison report

    Returns:
        Markdown comparison report as string
    """
    md = []

    # Header
    md.append("# HumanEval Benchmark Comparison\n")
    md.append(f"**Generated:** {datetime.now().isoformat()}\n")
    md.append(f"**Models Compared:** {len(benchmark_runs)}\n")

    # Overall Comparison
    md.append("## Overall Performance\n")
    md.append("| Model | Pass@k | Score | Total Cost | Cost per Task | Avg Latency |")
    md.append("|-------|--------|-------|------------|---------------|-------------|")

    for run in sorted(benchmark_runs, key=lambda x: x.overall_score, reverse=True):
        k = run.pass_at_k.get("k", 10)
        score = run.overall_score
        md.append(
            f"| {run.model_name} | "
            f"pass@{k} | "
            f"**{score:.4f}** ({score*100:.2f}%) | "
            f"${run.total_cost_usd:.4f} | "
            f"${run.avg_cost_per_task:.6f} | "
            f"{run.avg_latency_seconds:.3f}s |"
        )
    md.append("")

    # Cost Efficiency (Score per Dollar)
    md.append("## Cost Efficiency (Score per Dollar)\n")
    md.append("| Model | Score | Cost | Score/$1 | Samples/$1 |")
    md.append("|-------|-------|------|----------|------------|")

    efficiency_data = []
    for run in benchmark_runs:
        if run.total_cost_usd > 0:
            score_per_dollar = run.overall_score / run.total_cost_usd
            samples_per_dollar = run.total_samples / run.total_cost_usd
            efficiency_data.append(
                {
                    "model": run.model_name,
                    "score": run.overall_score,
                    "cost": run.total_cost_usd,
                    "score_per_dollar": score_per_dollar,
                    "samples_per_dollar": samples_per_dollar,
                }
            )

    for data in sorted(
        efficiency_data, key=lambda x: x["score_per_dollar"], reverse=True
    ):
        md.append(
            f"| {data['model']} | "
            f"{data['score']:.4f} | "
            f"${data['cost']:.4f} | "
            f"**{data['score_per_dollar']:.2f}** | "
            f"{data['samples_per_dollar']:.0f} |"
        )
    md.append("")

    # Token Usage Comparison
    md.append("## Token Usage\n")
    md.append(
        "| Model | Total Tokens | Input Tokens | Output Tokens | Avg Tokens/Sample |"
    )
    md.append(
        "|-------|--------------|--------------|---------------|-------------------|"
    )

    for run in benchmark_runs:
        avg_tokens = (
            run.total_tokens / run.total_samples if run.total_samples > 0 else 0
        )
        md.append(
            f"| {run.model_name} | "
            f"{run.total_tokens:,} | "
            f"{run.total_input_tokens:,} | "
            f"{run.total_output_tokens:,} | "
            f"{avg_tokens:.1f} |"
        )
    md.append("")

    # Performance vs Cost Chart (text-based)
    md.append("## Performance vs Cost\n")
    md.append("```")
    md.append("Score (pass@k)  vs  Total Cost")
    md.append("-" * 50)

    max_cost = max(run.total_cost_usd for run in benchmark_runs)
    for run in sorted(benchmark_runs, key=lambda x: x.overall_score, reverse=True):
        score_bar = "█" * int(run.overall_score * 50)
        cost_bar = (
            "■" * int((run.total_cost_usd / max_cost) * 30) if max_cost > 0 else ""
        )
        md.append(
            f"{run.model_name:20s} "
            f"[{run.overall_score:.3f}] {score_bar:50s} "
            f"[${run.total_cost_usd:.2f}] {cost_bar}"
        )
    md.append("```")
    md.append("")

    report = "\n".join(md)

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Saved comparison report to: {output_path}")

    return report


def generate_cost_breakdown(
    benchmark_run: BenchmarkRun, top_n: int = 20
) -> dict[str, Any]:
    """
    Generate detailed cost breakdown by task.

    Args:
        benchmark_run: BenchmarkRun object
        top_n: Number of top tasks to include

    Returns:
        Dictionary with cost breakdown data
    """
    tasks_by_cost = sorted(
        benchmark_run.per_task_results,
        key=lambda x: x.get("total_cost_usd", 0),
        reverse=True,
    )

    top_tasks = tasks_by_cost[:top_n]
    top_tasks_cost = sum(t.get("total_cost_usd", 0) for t in top_tasks)
    top_tasks_pct = (
        top_tasks_cost / benchmark_run.total_cost_usd * 100
        if benchmark_run.total_cost_usd > 0
        else 0
    )

    return {
        "total_cost": benchmark_run.total_cost_usd,
        "total_tasks": benchmark_run.total_tasks,
        "avg_cost_per_task": benchmark_run.avg_cost_per_task,
        "top_n_tasks": top_n,
        "top_n_cost": top_tasks_cost,
        "top_n_percentage": top_tasks_pct,
        "top_tasks": [
            {
                "task_id": t["task_id"],
                "cost": t["total_cost_usd"],
                "tokens": t["total_input_tokens"] + t["total_output_tokens"],
                "samples": t["samples_generated"],
            }
            for t in top_tasks
        ],
    }


def print_console_summary(benchmark_run: BenchmarkRun):
    """
    Print a concise summary to console.

    Args:
        benchmark_run: BenchmarkRun object
    """
    k = benchmark_run.pass_at_k.get("k", 10)
    score = benchmark_run.overall_score

    print("\n" + "=" * 70)
    print(f"  HumanEval Benchmark Results: {benchmark_run.model_name}")
    print("=" * 70)
    print(f"\n  Pass@{k}:              {score:.4f} ({score*100:.2f}%)")
    print(f"  Total Cost:           ${benchmark_run.total_cost_usd:.4f}")
    print(f"  Total Tasks:          {benchmark_run.total_tasks}")
    print(f"  Total Samples:        {benchmark_run.total_samples:,}")
    print(f"  Total Tokens:         {benchmark_run.total_tokens:,}")
    print(f"  Avg Latency:          {benchmark_run.avg_latency_seconds:.3f}s")
    print(f"  Success Rate:         {benchmark_run.success_rate:.2%}")

    if benchmark_run.model_selection_stats:
        print("\n  Model Selection:")
        for model, data in benchmark_run.model_selection_stats.get(
            "models", {}
        ).items():
            pct = data.get("percentage", 0)
            print(f"    - {model}: {pct:.2f}%")

    print("\n" + "=" * 70 + "\n")
