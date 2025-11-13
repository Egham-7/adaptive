"""
Utility modules for HumanEval benchmarking.

This module exports utilities for response parsing, result tracking, reporting, and validation.
"""

from .pass_at_k import (
    calculate_overall_pass_at_k,
    calculate_pass_at_k,
    extract_task_results_from_benchmark,
)
from .reporting import (
    compare_benchmarks,
    generate_cost_breakdown,
    generate_markdown_report,
    print_console_summary,
)
from .response_parser import (
    PricingCalculator,
    ResponseMetrics,
    aggregate_metrics,
    estimate_tokens,
    parse_adaptive_response,
    parse_claude_response,
    parse_glm_response,
    parse_openai_response,
)
from .result_tracker import BenchmarkRun, ResultTracker
from .validators import (
    ValidationError,
    validate_benchmark_run,
    validate_response_metrics,
    validate_task_metrics,
)

__all__ = [
    # Response parsing
    "ResponseMetrics",
    "parse_claude_response",
    "parse_glm_response",
    "parse_adaptive_response",
    "parse_openai_response",
    "estimate_tokens",
    "aggregate_metrics",
    "PricingCalculator",
    # Result tracking
    "BenchmarkRun",
    "ResultTracker",
    # Reporting
    "generate_markdown_report",
    "compare_benchmarks",
    "generate_cost_breakdown",
    "print_console_summary",
    # Validation
    "ValidationError",
    "validate_response_metrics",
    "validate_task_metrics",
    "validate_benchmark_run",
    # Pass@k calculation (workaround for DeepEval pandas bug)
    "calculate_pass_at_k",
    "calculate_overall_pass_at_k",
    "extract_task_results_from_benchmark",
]
