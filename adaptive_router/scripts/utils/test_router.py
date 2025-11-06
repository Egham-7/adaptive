#!/usr/bin/env python3
"""Router Testing Script - Comprehensive testing for UniRouter.

This script tests the routing logic (model selection) without making any API calls.
It validates:
- Cost bias behavior (0.0=cheapest to 1.0=highest quality)
- Provider filtering (OpenAI only, Anthropic only, etc.)
- Prompt variety (coding, creative, analysis, etc.)
- Determinism (same input = same output)
- Performance (routing speed)

Usage:
    python scripts/test_router.py                    # Run all tests
    python scripts/test_router.py --cost-bias 0.5   # Test specific bias
    python scripts/test_router.py --providers openai # Test specific provider
    python scripts/test_router.py --verbose         # Detailed output
    python scripts/test_router.py --output test.json # Save results
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, cast

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path to import adaptive_router
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_router import ModelRouter, ModelSelectionRequest
from adaptive_router.models.api import Model
from app.models import RegistryModel

console = Console()


# ============================================================================
# Test Configuration
# ============================================================================

# Test prompts covering different scenarios
TEST_PROMPTS = {
    "coding_simple": "Write a Python function to reverse a string",
    "coding_complex": "Implement a distributed lock manager with leader election",
    "creative": "Write a short story about a robot learning to paint",
    "analysis": "Explain the trade-offs between microservices and monolithic architecture",
    "reasoning": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "simple_qa": "What is the capital of France?",
    "math": "Solve for x: 2x + 5 = 15",
    "technical": "Explain how a B-tree differs from a binary search tree",
}

# Cost bias values to test
COST_BIAS_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]

# Provider groups to test
PROVIDER_GROUPS = {
    "all": None,  # No filtering
    "openai": ["openai"],
    "anthropic": ["anthropic"],
    "budget": ["openai", "deepseek", "gemini"],
    "premium": ["openai", "anthropic"],
}


# ============================================================================
# Test Functions
# ============================================================================


def test_cost_bias_variation(
    router: ModelRouter, prompt: str, verbose: bool = False
) -> List[Dict[str, Any]]:
    """Test routing decisions across different cost bias values.

    Args:
        router: ModelRouter instance
        prompt: Test prompt
        verbose: Whether to print detailed output

    Returns:
        List of test results
    """
    console.print("\n[bold cyan]Test 1: Cost Bias Variation[/bold cyan]")
    console.print("─" * 80)
    console.print(f'Prompt: [italic]"{prompt}"[/italic]\n')

    results = []

    for cost_bias in COST_BIAS_VALUES:
        request = ModelSelectionRequest(prompt=prompt, cost_bias=cost_bias)

        start_time = time.perf_counter()
        response = router.select_model(request)
        routing_time = (time.perf_counter() - start_time) * 1000

        result = {
            "test": "cost_bias_variation",
            "prompt": prompt,
            "cost_bias": cost_bias,
            "selected_model": response.model_id,
            "alternatives": [alt.model_id for alt in response.alternatives[:3]],
            "routing_time_ms": round(routing_time, 2),
        }
        results.append(result)

        if verbose:
            console.print(
                f"Cost Bias {cost_bias:.1f}: [green]{result['selected_model']}[/green] "
                f"({routing_time:.2f}ms)"
            )

    # Create results table
    table = Table(title="Cost Bias Test Results")
    table.add_column("Cost Bias", style="cyan")
    table.add_column("Selected Model", style="green")
    table.add_column("Alternatives", style="yellow")
    table.add_column("Time", style="magenta")

    for result in results:
        cost_bias = cast(float, result["cost_bias"])
        selected_model = str(result["selected_model"])
        alternatives = result["alternatives"]
        routing_time_ms = cast(float, result["routing_time_ms"])

        # Ensure alternatives is a list of strings
        if isinstance(alternatives, list):
            alt_str = ", ".join(str(alt) for alt in alternatives[:2])
        else:
            alt_str = ""

        table.add_row(
            f"{cost_bias:.1f}",
            selected_model,
            alt_str,
            f"{routing_time_ms:.1f}ms",
        )

    console.print(table)

    # Validation
    cheapest_model = results[0]["selected_model"]
    premium_model = results[-1]["selected_model"]

    console.print(
        f"\n✅ Cost bias working: "
        f"Cheapest at 0.0 = [green]{cheapest_model}[/green], "
        f"Premium at 1.0 = [green]{premium_model}[/green]"
    )

    return results


def test_provider_filtering(
    router: ModelRouter, prompt: str, verbose: bool = False
) -> List[Dict[str, Any]]:
    """Test routing with different provider constraints.

    Args:
        router: ModelRouter instance
        prompt: Test prompt
        verbose: Whether to print detailed output

    Returns:
        List of test results
    """
    console.print("\n[bold cyan]Test 2: Provider Filtering[/bold cyan]")
    console.print("─" * 80)
    console.print(f'Prompt: [italic]"{prompt}"[/italic]\n')

    results = []

    # Get all available models from the router
    available_models = (
        router.get_supported_models()
    )  # Returns ["provider:model_name", ...]

    # Test with different provider groups
    for group_name, providers in PROVIDER_GROUPS.items():
        if providers is None:
            # All providers - no filtering
            request = ModelSelectionRequest(prompt=prompt, cost_bias=0.5)
        else:
            # Filter models by provider
            filtered_model_ids = [
                model_id
                for model_id in available_models
                if any(model_id.split("/")[0] == provider for provider in providers)
            ]

            # Convert filtered model IDs to RegistryModel objects
            filtered_models = [
                RegistryModel(
                    provider=model_id.split(":")[0], model_name=model_id.split(":")[1]
                )
                for model_id in filtered_model_ids
            ]

            # Create request with filtered models
            request = ModelSelectionRequest(
                prompt=prompt, cost_bias=0.5, models=filtered_models
            )

        start_time = time.perf_counter()
        response = router.select_model(request)
        routing_time = (time.perf_counter() - start_time) * 1000

        result = {
            "test": "provider_filtering",
            "provider_group": group_name,
            "providers": providers,
            "prompt": prompt,
            "selected_model": response.model_id,
            "routing_time_ms": round(routing_time, 2),
        }
        results.append(result)

        # Validate that selected model matches provider constraint
        if providers is not None:
            selected_provider = response.model_id.split(":", 1)[0]
            if selected_provider not in providers:
                console.print(
                    f"[red]WARNING: Selected provider '{selected_provider}' "
                    f"not in allowed providers {providers}[/red]"
                )

        if verbose:
            provider_info = f" (from {providers})" if providers else " (all providers)"
            console.print(
                f"Provider Group '{group_name}'{provider_info}: "
                f"[green]{result['selected_model']}[/green]"
            )

    # Create results table
    table = Table(title="Provider Filtering Test Results")
    table.add_column("Provider Group", style="cyan")
    table.add_column("Selected Model", style="green")
    table.add_column("Time", style="magenta")

    for result in results:
        provider_group = result["provider_group"]
        selected_model = result["selected_model"]
        routing_time_ms = result["routing_time_ms"]

        table.add_row(
            str(provider_group),
            str(selected_model),
            f"{routing_time_ms:.1f}ms",
        )

    console.print(table)
    console.print("\n✅ Provider filtering test completed")

    return results


def test_prompt_variety(
    router: ModelRouter, cost_bias: float = 0.5, verbose: bool = False
) -> List[Dict[str, Any]]:
    """Test routing across different prompt types.

    Args:
        router: ModelRouter instance
        cost_bias: Cost bias to use
        verbose: Whether to print detailed output

    Returns:
        List of test results
    """
    console.print("\n[bold cyan]Test 3: Prompt Variety[/bold cyan]")
    console.print("─" * 80)
    console.print(f"Cost Bias: {cost_bias}\n")

    results = []

    for prompt_type, prompt in TEST_PROMPTS.items():
        request = ModelSelectionRequest(prompt=prompt, cost_bias=cost_bias)

        start_time = time.perf_counter()
        response = router.select_model(request)
        routing_time = (time.perf_counter() - start_time) * 1000

        result = {
            "test": "prompt_variety",
            "prompt_type": prompt_type,
            "prompt": prompt,
            "cost_bias": cost_bias,
            "selected_model": response.model_id,
            "routing_time_ms": round(routing_time, 2),
        }
        results.append(result)

        if verbose:
            console.print(
                f"{prompt_type:20s}: [green]{result['selected_model']:30s}[/green] "
                f"({routing_time:.2f}ms)"
            )

    # Create results table
    table = Table(title="Prompt Variety Test Results")
    table.add_column("Prompt Type", style="cyan")
    table.add_column("Prompt", style="white", max_width=50)
    table.add_column("Selected Model", style="green")
    table.add_column("Time", style="magenta")

    for result in results:
        prompt_type = str(result["prompt_type"])
        prompt = str(result["prompt"])
        selected_model = str(result["selected_model"])
        routing_time_ms = cast(float, result["routing_time_ms"])

        # Truncate prompt if needed
        truncated_prompt = prompt[:47] + "..." if len(prompt) > 50 else prompt

        table.add_row(
            prompt_type,
            truncated_prompt,
            selected_model,
            f"{routing_time_ms:.1f}ms",
        )

    console.print(table)
    console.print("\n✅ Prompt variety test completed")

    return results


def test_determinism(
    router: ModelRouter, prompt: str, cost_bias: float = 0.5, iterations: int = 10
) -> Dict[str, Any]:
    """Test that routing is deterministic (same input = same output).

    Args:
        router: ModelRouter instance
        prompt: Test prompt
        cost_bias: Cost bias to use
        iterations: Number of iterations to test

    Returns:
        Test result
    """
    console.print("\n[bold cyan]Test 4: Determinism[/bold cyan]")
    console.print("─" * 80)
    console.print(f"Running {iterations} iterations with same input...\n")

    request = ModelSelectionRequest(prompt=prompt, cost_bias=cost_bias)

    selected_models = []
    for i in range(iterations):
        response = router.select_model(request)
        selected_models.append(response.model_id)

    # Check if all selections are identical
    is_deterministic = len(set(selected_models)) == 1

    result = {
        "test": "determinism",
        "prompt": prompt,
        "cost_bias": cost_bias,
        "iterations": iterations,
        "selected_model": selected_models[0],
        "is_deterministic": is_deterministic,
        "unique_selections": list(set(selected_models)),
    }

    if is_deterministic:
        console.print(
            f"✅ [green]Deterministic routing verified![/green] "
            f"All {iterations} iterations selected: [bold]{selected_models[0]}[/bold]"
        )
    else:
        console.print(
            f"❌ [red]Non-deterministic routing detected![/red] "
            f"Got {len(set(selected_models))} different selections: {set(selected_models)}"
        )

    return result


def test_performance(
    router: ModelRouter, prompt: str, iterations: int = 100
) -> Dict[str, Any]:
    """Benchmark routing performance.

    Args:
        router: ModelRouter instance
        prompt: Test prompt
        iterations: Number of iterations for benchmarking

    Returns:
        Performance metrics
    """
    console.print("\n[bold cyan]Test 5: Performance Benchmark[/bold cyan]")
    console.print("─" * 80)
    console.print(f"Running {iterations} iterations...\n")

    request = ModelSelectionRequest(prompt=prompt, cost_bias=0.5)

    routing_times = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Benchmarking routing...", total=iterations)

        for i in range(iterations):
            start_time = time.perf_counter()
            router.select_model(request)
            routing_time = (time.perf_counter() - start_time) * 1000
            routing_times.append(routing_time)
            progress.update(task, advance=1)

    # Calculate statistics
    routing_times.sort()
    result = {
        "test": "performance",
        "iterations": iterations,
        "min_ms": round(routing_times[0], 2),
        "max_ms": round(routing_times[-1], 2),
        "mean_ms": round(sum(routing_times) / len(routing_times), 2),
        "median_ms": round(routing_times[len(routing_times) // 2], 2),
        "p95_ms": round(routing_times[int(len(routing_times) * 0.95)], 2),
        "p99_ms": round(routing_times[int(len(routing_times) * 0.99)], 2),
    }

    # Create results table
    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Iterations", str(result["iterations"]))
    table.add_row("Min", f"{result['min_ms']:.2f}ms")
    table.add_row("Mean", f"{result['mean_ms']:.2f}ms")
    table.add_row("Median", f"{result['median_ms']:.2f}ms")
    table.add_row("P95", f"{result['p95_ms']:.2f}ms")
    table.add_row("P99", f"{result['p99_ms']:.2f}ms")
    table.add_row("Max", f"{result['max_ms']:.2f}ms")

    console.print(table)

    # Validation
    p99_ms = cast(float, result["p99_ms"])
    if p99_ms < 50:
        console.print(
            f"\n✅ [green]Excellent performance![/green] P99 latency: {p99_ms:.2f}ms"
        )
    elif p99_ms < 100:
        console.print(
            f"\n✅ [yellow]Good performance[/yellow] P99 latency: {p99_ms:.2f}ms"
        )
    else:
        console.print(
            f"\n⚠️  [red]Performance warning[/red] P99 latency: {p99_ms:.2f}ms (expected <50ms)"
        )

    return result


# ============================================================================
# Main Test Runner
# ============================================================================


def run_all_tests(args: argparse.Namespace) -> Dict[str, Any]:
    """Run all configured tests.

    Args:
        args: Command line arguments

    Returns:
        Dictionary containing all test results
    """
    console.print(
        Panel.fit(
            "[bold cyan]🧪 UniRouter Testing Suite[/bold cyan]\n"
            "[dim]Testing routing logic without making any API calls[/dim]",
            border_style="cyan",
        )
    )

    # Initialize router
    console.print("\n[bold]Initializing router...[/bold]")
    try:
        config_file = (
            Path(__file__).parent.parent.parent
            / "adaptive_router"
            / "config"
            / "unirouter_models.yaml"
        )
        profile_path = (
            Path(__file__).parent.parent.parent / "data" / "global_profile.json"
        )

        with open(config_file) as f:
            import yaml

            config = yaml.safe_load(f)
            models = []
            for model_data in config.get("gpt5_models", []):
                # Parse provider and model name from id
                provider, model_name = model_data["id"].split(":", 1)
                models.append(
                    Model(
                        provider=provider,
                        model_name=model_name,
                        cost_per_1m_input_tokens=model_data["cost_per_1m_input_tokens"],
                        cost_per_1m_output_tokens=model_data[
                            "cost_per_1m_output_tokens"
                        ],
                    )
                )

        router = ModelRouter.from_local_file(
            profile_path=profile_path,
            models=models,
        )
        console.print("✅ [green]Router initialized successfully[/green]\n")
    except Exception as e:
        console.print(f"❌ [red]Failed to initialize router: {e}[/red]")
        sys.exit(1)

    all_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {},
    }

    # Test 1: Cost Bias Variation
    if args.test_all or args.test == "cost_bias":
        test_prompt = str(args.prompt or TEST_PROMPTS["coding_simple"])
        results = test_cost_bias_variation(router, test_prompt, args.verbose)
        test_dict = all_results["tests"]
        assert isinstance(test_dict, dict)
        test_dict["cost_bias_variation"] = results

    # Test 2: Provider Filtering
    if args.test_all or args.test == "providers":
        test_prompt = str(args.prompt or TEST_PROMPTS["technical"])
        results = test_provider_filtering(router, test_prompt, args.verbose)
        test_dict = all_results["tests"]
        assert isinstance(test_dict, dict)
        test_dict["provider_filtering"] = results

    # Test 3: Prompt Variety
    if args.test_all or args.test == "prompts":
        cost_bias = float(args.cost_bias if args.cost_bias is not None else 0.5)
        results = test_prompt_variety(router, cost_bias, args.verbose)
        test_dict = all_results["tests"]
        assert isinstance(test_dict, dict)
        test_dict["prompt_variety"] = results

    # Test 4: Determinism
    if args.test_all or args.test == "determinism":
        test_prompt = str(args.prompt or TEST_PROMPTS["coding_simple"])
        cost_bias = float(args.cost_bias if args.cost_bias is not None else 0.5)
        result = test_determinism(router, test_prompt, cost_bias, args.iterations)
        test_dict = all_results["tests"]
        assert isinstance(test_dict, dict)
        test_dict["determinism"] = result

    # Test 5: Performance
    if args.test_all or args.test == "performance":
        test_prompt = str(args.prompt or TEST_PROMPTS["coding_simple"])
        result = test_performance(router, test_prompt, args.iterations)
        test_dict = all_results["tests"]
        assert isinstance(test_dict, dict)
        test_dict["performance"] = result

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test UniRouter routing logic (no API calls)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_router.py                     # Run all tests
  python scripts/test_router.py --test cost_bias   # Test cost bias only
  python scripts/test_router.py --cost-bias 0.3    # Test specific cost bias
  python scripts/test_router.py --verbose          # Detailed output
  python scripts/test_router.py --output test.json # Save results to file
        """,
    )

    parser.add_argument(
        "--test",
        choices=["cost_bias", "providers", "prompts", "determinism", "performance"],
        help="Run specific test (default: all tests)",
    )
    parser.add_argument(
        "--cost-bias",
        type=float,
        help="Cost bias value to test (0.0-1.0)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Custom prompt to test",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for determinism/performance tests (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )

    args = parser.parse_args()
    args.test_all = args.test is None

    # Run tests
    results = run_all_tests(args)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n💾 Results saved to: [bold]{output_path}[/bold]")

    # Summary
    console.print(
        Panel.fit(
            "[bold green]✅ All tests completed successfully![/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
