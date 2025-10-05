#!/usr/bin/env python3
"""
GenAI Performance Benchmarking Tool
Consolidated script for running benchmarks and analyzing results.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import seaborn as sns
import typer

# Set style for better looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

app = typer.Typer(help="GenAI Performance Benchmarking Tool")
console = Console()


@dataclass
class BenchmarkParameters:
    """Data class to hold all benchmark parameters"""

    # Core parameters
    url: str
    model: str
    concurrency: str | None = None
    check_health: bool = True

    # Audio Input
    audio_length_mean: float | None = None
    audio_length_stddev: float | None = None
    audio_format: str | None = None
    audio_depths: str | None = None
    audio_sample_rates: str | None = None
    audio_num_channels: int | None = None

    # Endpoint
    model_selection_strategy: str | None = None
    backend: str | None = None
    endpoint: str | None = None
    endpoint_type: str | None = None
    server_metrics_url: str | None = None
    streaming: bool = False

    # Image Input
    image_width_mean: int | None = None
    image_width_stddev: int | None = None
    image_height_mean: int | None = None
    image_height_stddev: int | None = None
    image_format: str | None = None

    # Input
    batch_size_audio: int | None = None
    batch_size_image: int | None = None
    batch_size_text: int | None = None
    extra_inputs: str | None = None
    goodput: str | None = None
    header: str | None = None
    input_file: str | None = None
    num_dataset_entries: int | None = None
    num_prefix_prompts: int | None = None
    output_tokens_mean: int | None = None
    output_tokens_mean_deterministic: bool = False
    output_tokens_stddev: int | None = None
    random_seed: int | None = None
    grpc_method: str | None = None
    synthetic_input_tokens_mean: int | None = None
    synthetic_input_tokens_stddev: int | None = None
    prefix_prompt_length: int | None = None
    warmup_request_count: int | None = None

    # Other
    verbose: bool = False

    # Output
    artifact_dir: str | None = None
    checkpoint_dir: str | None = None
    generate_plots: bool = False
    enable_checkpointing: bool = False
    profile_export_file: str | None = None

    # Profiling
    measurement_interval: int | None = None
    request_count: int | None = None
    request_rate: float | None = None
    fixed_schedule: str | None = None
    stability_percentage: float | None = None

    # Session
    num_sessions: int | None = None
    session_concurrency: int | None = None
    session_delay_ratio: float | None = None
    session_turn_delay_mean: float | None = None
    session_turn_delay_stddev: float | None = None
    session_turns_mean: int | None = None
    session_turns_stddev: int | None = None

    # Tokenizer
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    tokenizer_trust_remote_code: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for easy parameter passing"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class GenAIPerfAnalyzer:
    def __init__(self, results_dir: str = "./results") -> None:
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def parse_genai_perf_csv(self, csv_path: str) -> dict[str, Any] | None:
        """Parse GenAI-Perf CSV results file"""
        try:
            df = pd.read_csv(csv_path, skiprows=8)
            metrics = {}
            for _, row in df.iterrows():
                metric_name = row["Metric"]
                if metric_name == "Throughput":
                    metrics["throughput_tps"] = row["Mean"]
                elif metric_name == "Time to First Token":
                    metrics["ttft_ms"] = row["Mean"]
                elif metric_name == "Inter Token Latency":
                    metrics["itl_ms"] = row["Mean"]
                elif metric_name == "Request Latency":
                    metrics["e2e_latency_ms"] = row["Mean"]
                elif metric_name == "Time Per Output Token":
                    metrics["time_per_output_token_ms"] = row["Mean"]
            return metrics
        except Exception as e:
            console.print(f"[red]Error parsing {csv_path}: {e}[/red]")
            return None

    def parse_genai_perf_json(self, json_path: str) -> dict[str, Any] | None:
        """Parse GenAI-Perf JSON results file"""
        try:
            with open(json_path) as f:
                data = json.load(f)
            metrics = {}

            # Handle new GenAI-Perf JSON format
            if "request_throughput" in data:
                metrics["throughput_tps"] = data["request_throughput"].get("avg", 0)
            if "time_to_first_token" in data:
                metrics["ttft_ms"] = data["time_to_first_token"].get("avg", 0)
            if "inter_token_latency" in data:
                metrics["itl_ms"] = data["inter_token_latency"].get("avg", 0)
            if "request_latency" in data:
                metrics["e2e_latency_ms"] = data["request_latency"].get("avg", 0)
            if "time_per_output_token" in data:
                metrics["time_per_output_token_ms"] = data["time_per_output_token"].get(
                    "avg", 0
                )

            # Fallback for older format
            if not metrics and "service_kind" in data:
                for key, value in data.items():
                    if key == "request_throughput":
                        metrics["throughput_tps"] = value
                    elif key == "time_to_first_token":
                        metrics["ttft_ms"] = value.get("mean", 0)
                    elif key == "inter_token_latency":
                        metrics["itl_ms"] = value.get("mean", 0)
                    elif key == "request_latency":
                        metrics["e2e_latency_ms"] = value.get("mean", 0)

            # If still no metrics, use the data as-is
            if not metrics:
                metrics = data

            return metrics
        except Exception as e:
            console.print(f"[red]Error parsing {json_path}: {e}[/red]")
            return None

    def extract_test_info(self, filepath: str) -> dict[str, Any]:
        """Extract test information from file path"""
        path_parts = Path(filepath).parts
        filename = Path(filepath).stem

        concurrency = 1
        if "_c" in filename:
            try:
                concurrency = int(filename.split("_c")[-1])
            except ValueError:
                pass

        test_type = "unknown"
        for part in path_parts:
            if any(
                keyword in part.lower()
                for keyword in ["quick", "code", "text", "question", "long"]
            ):
                test_type = part
                break

        if test_type == "unknown" and "_" in filename:
            test_type = filename.split("_")[0]

        return {"concurrency": concurrency, "test_type": test_type}

    def scan_results_directory(self) -> pd.DataFrame:
        """Scan results directory for benchmark files"""
        results = []

        csv_files = list(self.results_dir.rglob("*_genai_perf.csv"))
        for csv_file in csv_files:
            metrics = self.parse_genai_perf_csv(str(csv_file))
            if metrics:
                test_info = self.extract_test_info(str(csv_file))
                result = {**metrics, **test_info, "source_file": str(csv_file)}
                results.append(result)

        # Look for GenAI-Perf JSON files
        json_files = list(self.results_dir.rglob("*_genai_perf.json"))
        for json_file in json_files:
            metrics = self.parse_genai_perf_json(str(json_file))
            if metrics:
                test_info = self.extract_test_info(str(json_file))
                result = {**metrics, **test_info, "source_file": str(json_file)}
                results.append(result)

        return pd.DataFrame(results)

    def generate_performance_plots(self, df: pd.DataFrame) -> None:
        """Generate comprehensive performance visualization plots"""
        if df.empty:
            console.print("[yellow]No data to plot[/yellow]")
            return

        # Throughput vs Concurrency
        plt.figure(figsize=(12, 8))
        if "test_type" in df.columns and len(df["test_type"].unique()) > 1:
            for test_type in df["test_type"].unique():
                test_data = df[df["test_type"] == test_type]
                if not test_data.empty:
                    plt.plot(
                        test_data["concurrency"],
                        test_data["throughput_tps"],
                        "o-",
                        label=test_type,
                        linewidth=2,
                        markersize=8,
                    )
        else:
            plt.plot(
                df["concurrency"], df["throughput_tps"], "o-", linewidth=2, markersize=8
            )

        plt.xlabel("Concurrency Level", fontsize=12)
        plt.ylabel("Throughput (tokens/second)", fontsize=12)
        plt.title(
            "Go LLM API: Throughput vs Concurrency", fontsize=14, fontweight="bold"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "throughput_vs_concurrency.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Latency Metrics
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(
            df["concurrency"],
            df["ttft_ms"],
            "o-",
            color="red",
            linewidth=2,
            markersize=8,
        )
        plt.xlabel("Concurrency")
        plt.ylabel("Time to First Token (ms)")
        plt.title("TTFT vs Concurrency")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(
            df["concurrency"],
            df["e2e_latency_ms"],
            "o-",
            color="blue",
            linewidth=2,
            markersize=8,
        )
        plt.xlabel("Concurrency")
        plt.ylabel("End-to-End Latency (ms)")
        plt.title("E2E Latency vs Concurrency")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        if "itl_ms" in df.columns:
            plt.plot(
                df["concurrency"],
                df["itl_ms"],
                "o-",
                color="green",
                linewidth=2,
                markersize=8,
            )
            plt.xlabel("Concurrency")
            plt.ylabel("Inter Token Latency (ms)")
            plt.title("ITL vs Concurrency")
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "latency_metrics.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        console.print(f"[green]✓ Performance plots saved to {self.plots_dir}/[/green]")

    def generate_summary_report(self, df: pd.DataFrame) -> None:
        """Generate a comprehensive summary report"""
        if df.empty:
            console.print("[yellow]No data available for summary report[/yellow]")
            return

        report_path = self.results_dir / "benchmark_summary_report.txt"

        with open(report_path, "w") as f:
            f.write("Go LLM API Benchmark Summary Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Test Runs: {len(df)}\n\n")

            f.write("Overall Performance Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Max Throughput: {df['throughput_tps'].max():.2f} tokens/second\n")
            f.write(f"Min TTFT: {df['ttft_ms'].min():.2f} ms\n")
            f.write(f"Min E2E Latency: {df['e2e_latency_ms'].min():.2f} ms\n")
            f.write(
                f"Optimal Concurrency: {df.loc[df['throughput_tps'].idxmax(), 'concurrency']}\n\n"
            )

            if "test_type" in df.columns:
                f.write("Performance by Test Type:\n")
                f.write("-" * 30 + "\n")
                for test_type in df["test_type"].unique():
                    test_data = df[df["test_type"] == test_type]
                    f.write(f"\n{test_type}:\n")
                    f.write(
                        f"  Max Throughput: {test_data['throughput_tps'].max():.2f} tokens/second\n"
                    )
                    f.write(f"  Min TTFT: {test_data['ttft_ms'].min():.2f} ms\n")
                    f.write(
                        f"  Avg E2E Latency: {test_data['e2e_latency_ms'].mean():.2f} ms\n"
                    )

            f.write("\nConcurrency Analysis:\n")
            f.write("-" * 20 + "\n")
            concurrency_stats = (
                df.groupby("concurrency")
                .agg(
                    {
                        "throughput_tps": "mean",
                        "ttft_ms": "mean",
                        "e2e_latency_ms": "mean",
                    }
                )
                .round(2)
            )

            for concurrency, stats in concurrency_stats.iterrows():
                f.write(f"Concurrency {concurrency}:\n")
                f.write(f"  Throughput: {stats['throughput_tps']:.2f} tokens/second\n")
                f.write(f"  TTFT: {stats['ttft_ms']:.2f} ms\n")
                f.write(f"  E2E Latency: {stats['e2e_latency_ms']:.2f} ms\n\n")

        console.print(f"[green]✓ Summary report saved to {report_path}[/green]")

    def save_detailed_results(self, df: pd.DataFrame) -> None:
        """Save detailed results to CSV"""
        if df.empty:
            console.print("[yellow]No data to save[/yellow]")
            return

        csv_path = self.results_dir / "go_api_benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        console.print(f"[green]✓ Detailed results saved to {csv_path}[/green]")

    def run_analysis(self) -> None:
        """Run complete analysis pipeline"""
        console.print("[blue]Starting GenAI-Perf Results Analysis...[/blue]")
        console.print(f"Results directory: {self.results_dir}")

        df = self.scan_results_directory()

        if df.empty:
            console.print("[red]No benchmark results found![/red]")
            console.print("Make sure you have run benchmarks first.")
            return

        console.print(f"[green]Found {len(df)} benchmark results[/green]")

        console.print("\n[bold]Basic Statistics:[/bold]")
        console.print(
            f"Throughput range: {df['throughput_tps'].min():.2f} - {df['throughput_tps'].max():.2f} tokens/second"
        )
        console.print(
            f"TTFT range: {df['ttft_ms'].min():.2f} - {df['ttft_ms'].max():.2f} ms"
        )
        console.print(
            f"Concurrency levels tested: {sorted(df['concurrency'].unique())}"
        )

        self.generate_performance_plots(df)
        self.generate_summary_report(df)
        self.save_detailed_results(df)

        console.print("\n[green]Analysis Complete![/green]")
        console.print(f"📊 Plots: {self.plots_dir}/")
        console.print(f"📄 Summary: {self.results_dir}/benchmark_summary_report.txt")
        console.print(f"📊 CSV Data: {self.results_dir}/go_api_benchmark_results.csv")


def _run_benchmark_with_params(params: BenchmarkParameters) -> None:
    """Run benchmark with given parameters"""
    console.print(f"Router URL: {params.url}")
    console.print(f"Model: {params.model}")

    benchmarker = GenAIPerfBenchmarker(params.url, params.model)

    if params.check_health:
        console.print("\n[yellow]Checking API health...[/yellow]")
        if not benchmarker.check_api_health():
            console.print(f"[red]API not accessible at {params.url}[/red]")
            console.print("Please ensure your API is running and accessible.")
            raise typer.Exit(1)
        console.print("[green]✓ API is accessible[/green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running benchmarks...", total=None)
        benchmarker.run_simple_benchmark(params)
        progress.update(task, description="Benchmarks completed")

    console.print("\n[green]Benchmarking completed![/green]")
    console.print(f"Results saved in: {benchmarker.results_dir}")


def create_benchmark_params(**kwargs: Any) -> BenchmarkParameters:
    """Create BenchmarkParameters from keyword arguments"""
    return BenchmarkParameters(
        url=kwargs.get("url", "http://localhost:8080"),
        model=kwargs.get("model", ""),
        concurrency=kwargs.get("concurrency"),
        check_health=kwargs.get("check_health", True),
        audio_length_mean=kwargs.get("audio_length_mean"),
        audio_length_stddev=kwargs.get("audio_length_stddev"),
        audio_format=kwargs.get("audio_format"),
        audio_depths=kwargs.get("audio_depths"),
        audio_sample_rates=kwargs.get("audio_sample_rates"),
        audio_num_channels=kwargs.get("audio_num_channels"),
        model_selection_strategy=kwargs.get("model_selection_strategy"),
        backend=kwargs.get("backend"),
        endpoint=kwargs.get("endpoint"),
        endpoint_type=kwargs.get("endpoint_type"),
        server_metrics_url=kwargs.get("server_metrics_url"),
        streaming=kwargs.get("streaming", False),
        image_width_mean=kwargs.get("image_width_mean"),
        image_width_stddev=kwargs.get("image_width_stddev"),
        image_height_mean=kwargs.get("image_height_mean"),
        image_height_stddev=kwargs.get("image_height_stddev"),
        image_format=kwargs.get("image_format"),
        batch_size_audio=kwargs.get("batch_size_audio"),
        batch_size_image=kwargs.get("batch_size_image"),
        batch_size_text=kwargs.get("batch_size_text"),
        extra_inputs=kwargs.get("extra_inputs"),
        goodput=kwargs.get("goodput"),
        header=kwargs.get("header"),
        input_file=kwargs.get("input_file"),
        num_dataset_entries=kwargs.get("num_dataset_entries"),
        num_prefix_prompts=kwargs.get("num_prefix_prompts"),
        output_tokens_mean=kwargs.get("output_tokens_mean"),
        output_tokens_mean_deterministic=kwargs.get(
            "output_tokens_mean_deterministic", False
        ),
        output_tokens_stddev=kwargs.get("output_tokens_stddev"),
        random_seed=kwargs.get("random_seed"),
        grpc_method=kwargs.get("grpc_method"),
        synthetic_input_tokens_mean=kwargs.get("synthetic_input_tokens_mean"),
        synthetic_input_tokens_stddev=kwargs.get("synthetic_input_tokens_stddev"),
        prefix_prompt_length=kwargs.get("prefix_prompt_length"),
        warmup_request_count=kwargs.get("warmup_request_count"),
        verbose=kwargs.get("verbose", False),
        artifact_dir=kwargs.get("artifact_dir"),
        checkpoint_dir=kwargs.get("checkpoint_dir"),
        generate_plots=kwargs.get("generate_plots", False),
        enable_checkpointing=kwargs.get("enable_checkpointing", False),
        profile_export_file=kwargs.get("profile_export_file"),
        measurement_interval=kwargs.get("measurement_interval"),
        request_count=kwargs.get("request_count"),
        request_rate=kwargs.get("request_rate"),
        fixed_schedule=kwargs.get("fixed_schedule"),
        stability_percentage=kwargs.get("stability_percentage"),
        num_sessions=kwargs.get("num_sessions"),
        session_concurrency=kwargs.get("session_concurrency"),
        session_delay_ratio=kwargs.get("session_delay_ratio"),
        session_turn_delay_mean=kwargs.get("session_turn_delay_mean"),
        session_turn_delay_stddev=kwargs.get("session_turn_delay_stddev"),
        session_turns_mean=kwargs.get("session_turns_mean"),
        session_turns_stddev=kwargs.get("session_turns_stddev"),
        tokenizer=kwargs.get("tokenizer"),
        tokenizer_revision=kwargs.get("tokenizer_revision"),
        tokenizer_trust_remote_code=kwargs.get("tokenizer_trust_remote_code", False),
    )


class GenAIPerfBenchmarker:
    def __init__(
        self,
        router_url: str = "http://localhost:8080",
        model_name: str = "adaptive-go-api",
    ):
        self.router_url = router_url
        self.model_name = model_name
        self.results_dir = Path("./results")
        self.results_dir.mkdir(exist_ok=True)

    def check_api_health(self) -> bool:
        """Check if the Go API is accessible"""
        try:
            # Use the full URL as provided
            health_url = f"{self.router_url.rstrip('/')}/health"
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except (requests.RequestException, ConnectionError, requests.Timeout) as e:
            console.print(f"[dim]Health check failed for {self.router_url}: {e}[/dim]")
            return False

    def run_genai_perf_command(self, **kwargs: Any) -> bool:
        """Run genai-perf command with given parameters"""
        cmd = [
            "uv",
            "run",
            "genai-perf",
            "profile",
            "-m",
            "auto",  # Placeholder model name since API auto-selects
            "--endpoint-type",
            "chat",
            "--tokenizer",
            "gpt2",
            "-u",
            self.router_url,
        ]

        # Handle parameters that need special processing
        for key, value in kwargs.items():
            if value is None:
                continue

            if key.startswith("extra_inputs_"):
                cmd.extend(["--extra-inputs", f"{key[13:]}:{value}"])
            elif key == "profile_export_file":
                cmd.extend(["--profile-export-file", str(value)])
            elif key == "artifact_dir":
                cmd.extend(["--artifact-dir", str(value)])
            elif key in [
                "streaming",
                "verbose",
                "generate_plots",
                "enable_checkpointing",
                "output_tokens_mean_deterministic",
                "tokenizer_trust_remote_code",
            ]:
                # Boolean flags - only add if True
                if value:
                    cmd.append(f"--{key.replace('_', '-')}")
            elif key in [
                "audio_depths",
                "audio_sample_rates",
                "server_metrics_url",
                "goodput",
                "header",
            ]:
                # Parameters that can have multiple values
                if isinstance(value, str) and "," in value:
                    for item in value.split(","):
                        cmd.extend([f"--{key.replace('_', '-')}", item.strip()])
                else:
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])
            elif key not in [
                "url",
                "model",
                "check_health",
            ]:  # Skip these as they're handled elsewhere
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        console.print(f"[dim]Running command: {' '.join(cmd)}[/dim]")

        try:
            result = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, timeout=300, check=False
            )
            if result.returncode != 0:
                console.print(f"[red]Command failed with stdout: {result.stdout}[/red]")
                console.print(f"[red]Command failed with stderr: {result.stderr}[/red]")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            console.print("[red]Benchmark timed out[/red]")
            return False
        except (subprocess.SubprocessError, OSError) as e:
            console.print(f"[red]Error running benchmark: {e}[/red]")
            return False

    def run_simple_benchmark(self, params: BenchmarkParameters) -> None:
        """Run simple benchmark without tokenizer dependencies"""
        console.print("[blue]Running Simple Benchmark (No Tokenizer)[/blue]")

        # Extract concurrency levels from params
        if params.concurrency:
            try:
                concurrency_levels = [
                    int(c.strip()) for c in params.concurrency.split(",")
                ]
            except ValueError:
                console.print("[red]Invalid concurrency levels format[/red]")
                concurrency_levels = [1, 5, 10, 25]
        else:
            concurrency_levels = [1, 5, 10, 25]

        test_scenarios = [
            ("quick_response", 50),
            ("medium_response", 150),
            ("long_response", 300),
        ]

        for test_name, max_tokens in test_scenarios:
            console.print(
                f"\n[bold]Testing: {test_name} (max_tokens: {max_tokens})[/bold]"
            )

            for concurrency_level in concurrency_levels:
                console.print(f"  Concurrency: {concurrency_level}")

                # Build command parameters, starting with defaults
                cmd_params = {
                    "num_dataset_entries": 30,
                    "concurrency": concurrency_level,
                    "extra_inputs_max_tokens": max_tokens,
                    "extra_inputs_temperature": 0.7,
                    "measurement_interval": 8000,
                    "profile_export_file": f"simple_{test_name}_c{concurrency_level}.json",
                    "artifact_dir": self.results_dir
                    / f"simple_{test_name}_c{concurrency_level}_artifacts",
                }

                # Override with any CLI parameters provided
                params_dict = params.to_dict()
                for key, value in params_dict.items():
                    if (
                        key != "concurrency"
                    ):  # Skip concurrency as we handle it specially
                        cmd_params[key] = value

                success = self.run_genai_perf_command(**cmd_params)

                if success:
                    console.print(
                        f"    [green]✓ Completed concurrency {concurrency_level}[/green]"
                    )
                else:
                    console.print(
                        f"    [red]✗ Failed concurrency {concurrency_level}[/red]"
                    )

                time.sleep(2)  # Small delay between tests

            console.print(f"[green]✓ Completed {test_name}[/green]")


@app.command()
def benchmark(
    url: str = typer.Option(
        "http://localhost:8080", "--url", "-u", help="Full Router URL with protocol"
    ),
    model: str = typer.Option("adaptive-go-api", "--model", "-m", help="Model name"),
    concurrency: str | None = typer.Option(
        None,
        "--concurrency",
        "-c",
        help="Comma-separated concurrency levels (e.g., 1,5,10)",
    ),
    check_health: bool = typer.Option(
        True, "--check-health/--no-check-health", help="Check API health before running"
    ),
    # Audio Input
    audio_length_mean: float | None = typer.Option(
        None, help="Mean length of audio data in seconds"
    ),
    audio_length_stddev: float | None = typer.Option(
        None, help="Standard deviation of audio length"
    ),
    audio_format: str | None = typer.Option(None, help="Audio format (wav, mp3)"),
    audio_depths: str | None = typer.Option(
        None, help="Comma-separated audio bit depths"
    ),
    audio_sample_rates: str | None = typer.Option(
        None, help="Comma-separated audio sample rates"
    ),
    audio_num_channels: int | None = typer.Option(
        None, help="Number of audio channels (1 or 2)"
    ),
    # Endpoint
    model_selection_strategy: str | None = typer.Option(
        None, help="Model selection strategy (round_robin, random)"
    ),
    backend: str | None = typer.Option(None, help="Backend (tensorrtllm, vllm)"),
    endpoint: str | None = typer.Option(None, help="Custom endpoint"),
    endpoint_type: str | None = typer.Option(None, help="Endpoint type"),
    server_metrics_url: str | None = typer.Option(
        None, help="Comma-separated server metrics URLs"
    ),
    streaming: bool = typer.Option(False, help="Enable streaming API"),
    # Image Input
    image_width_mean: int | None = typer.Option(None, help="Mean width of images"),
    image_width_stddev: int | None = typer.Option(
        None, help="Standard deviation of image width"
    ),
    image_height_mean: int | None = typer.Option(None, help="Mean height of images"),
    image_height_stddev: int | None = typer.Option(
        None, help="Standard deviation of image height"
    ),
    image_format: str | None = typer.Option(None, help="Image format (png, jpeg)"),
    # Input
    batch_size_audio: int | None = typer.Option(None, help="Audio batch size"),
    batch_size_image: int | None = typer.Option(None, help="Image batch size"),
    batch_size_text: int | None = typer.Option(None, help="Text batch size"),
    extra_inputs: str | None = typer.Option(
        None, help="Extra inputs in 'key:value' format"
    ),
    goodput: str | None = typer.Option(None, help="Goodput constraints"),
    header: str | None = typer.Option(None, help="Custom headers"),
    input_file: str | None = typer.Option(None, help="Input file path"),
    num_dataset_entries: int | None = typer.Option(
        None, "--num-dataset-entries", "--num-prompts", help="Number of unique payloads"
    ),
    num_prefix_prompts: int | None = typer.Option(
        None, help="Number of prefix prompts"
    ),
    output_tokens_mean: int | None = typer.Option(
        None, help="Mean number of output tokens"
    ),
    output_tokens_mean_deterministic: bool = typer.Option(
        False, help="Enable deterministic output tokens"
    ),
    output_tokens_stddev: int | None = typer.Option(
        None, help="Standard deviation of output tokens"
    ),
    random_seed: int | None = typer.Option(None, help="Random seed"),
    grpc_method: str | None = typer.Option(None, help="gRPC method name"),
    synthetic_input_tokens_mean: int | None = typer.Option(
        None, help="Mean synthetic input tokens"
    ),
    synthetic_input_tokens_stddev: int | None = typer.Option(
        None, help="Standard deviation of synthetic input tokens"
    ),
    prefix_prompt_length: int | None = typer.Option(None, help="Prefix prompt length"),
    warmup_request_count: int | None = typer.Option(
        None, help="Number of warmup requests"
    ),
    # Other
    verbose: bool = typer.Option(False, help="Enable verbose mode"),
    # Output
    artifact_dir: str | None = typer.Option(None, help="Artifact directory"),
    checkpoint_dir: str | None = typer.Option(None, help="Checkpoint directory"),
    generate_plots: bool = typer.Option(False, help="Generate plots"),
    enable_checkpointing: bool = typer.Option(False, help="Enable checkpointing"),
    profile_export_file: str | None = typer.Option(
        None, help="Profile export file path"
    ),
    # Profiling
    measurement_interval: int | None = typer.Option(
        None, help="Measurement interval in milliseconds"
    ),
    request_count: int | None = typer.Option(None, help="Number of requests"),
    request_rate: float | None = typer.Option(None, help="Request rate"),
    fixed_schedule: str | None = typer.Option(None, help="Fixed schedule file"),
    stability_percentage: float | None = typer.Option(
        None, help="Stability percentage"
    ),
    # Session
    num_sessions: int | None = typer.Option(None, help="Number of sessions"),
    session_concurrency: int | None = typer.Option(None, help="Session concurrency"),
    session_delay_ratio: float | None = typer.Option(None, help="Session delay ratio"),
    session_turn_delay_mean: float | None = typer.Option(
        None, help="Mean session turn delay"
    ),
    session_turn_delay_stddev: float | None = typer.Option(
        None, help="Standard deviation of session turn delay"
    ),
    session_turns_mean: int | None = typer.Option(None, help="Mean session turns"),
    session_turns_stddev: int | None = typer.Option(
        None, help="Standard deviation of session turns"
    ),
    # Tokenizer
    tokenizer: str | None = typer.Option(None, help="Tokenizer name or path"),
    tokenizer_revision: str | None = typer.Option(None, help="Tokenizer revision"),
    tokenizer_trust_remote_code: bool = typer.Option(
        False, help="Trust remote tokenizer code"
    ),
) -> None:
    """Run GenAI-Perf benchmarks"""
    console.print(
        Panel("[bold blue]GenAI-Perf Benchmarking Tool[/bold blue]", expand=False)
    )

    # Create benchmark parameters
    params = create_benchmark_params(
        url=url,
        model=model,
        concurrency=concurrency,
        check_health=check_health,
        audio_length_mean=audio_length_mean,
        audio_length_stddev=audio_length_stddev,
        audio_format=audio_format,
        audio_depths=audio_depths,
        audio_sample_rates=audio_sample_rates,
        audio_num_channels=audio_num_channels,
        model_selection_strategy=model_selection_strategy,
        backend=backend,
        endpoint=endpoint,
        endpoint_type=endpoint_type,
        server_metrics_url=server_metrics_url,
        streaming=streaming,
        image_width_mean=image_width_mean,
        image_width_stddev=image_width_stddev,
        image_height_mean=image_height_mean,
        image_height_stddev=image_height_stddev,
        image_format=image_format,
        batch_size_audio=batch_size_audio,
        batch_size_image=batch_size_image,
        batch_size_text=batch_size_text,
        extra_inputs=extra_inputs,
        goodput=goodput,
        header=header,
        input_file=input_file,
        num_dataset_entries=num_dataset_entries,
        num_prefix_prompts=num_prefix_prompts,
        output_tokens_mean=output_tokens_mean,
        output_tokens_mean_deterministic=output_tokens_mean_deterministic,
        output_tokens_stddev=output_tokens_stddev,
        random_seed=random_seed,
        grpc_method=grpc_method,
        synthetic_input_tokens_mean=synthetic_input_tokens_mean,
        synthetic_input_tokens_stddev=synthetic_input_tokens_stddev,
        prefix_prompt_length=prefix_prompt_length,
        warmup_request_count=warmup_request_count,
        verbose=verbose,
        artifact_dir=artifact_dir,
        checkpoint_dir=checkpoint_dir,
        generate_plots=generate_plots,
        enable_checkpointing=enable_checkpointing,
        profile_export_file=profile_export_file,
        measurement_interval=measurement_interval,
        request_count=request_count,
        request_rate=request_rate,
        fixed_schedule=fixed_schedule,
        stability_percentage=stability_percentage,
        num_sessions=num_sessions,
        session_concurrency=session_concurrency,
        session_delay_ratio=session_delay_ratio,
        session_turn_delay_mean=session_turn_delay_mean,
        session_turn_delay_stddev=session_turn_delay_stddev,
        session_turns_mean=session_turns_mean,
        session_turns_stddev=session_turns_stddev,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        tokenizer_trust_remote_code=tokenizer_trust_remote_code,
    )

    _run_benchmark_with_params(params)


@app.command()
def analyze(
    results_dir: str = typer.Option(
        "./results",
        "--results-dir",
        "-r",
        help="Directory containing benchmark results",
    ),
    plots_only: bool = typer.Option(False, "--plots-only", help="Generate plots only"),
    summary_only: bool = typer.Option(
        False, "--summary-only", help="Generate summary report only"
    ),
) -> None:
    """Analyze benchmark results"""
    console.print(
        Panel("[bold blue]GenAI-Perf Results Analysis[/bold blue]", expand=False)
    )

    analyzer = GenAIPerfAnalyzer(results_dir)

    if plots_only:
        df = analyzer.scan_results_directory()
        analyzer.generate_performance_plots(df)
    elif summary_only:
        df = analyzer.scan_results_directory()
        analyzer.generate_summary_report(df)
    else:
        analyzer.run_analysis()


@app.command()
def status(
    results_dir: str = typer.Option(
        "./results",
        "--results-dir",
        "-r",
        help="Directory containing benchmark results",
    )
) -> None:
    """Show status of benchmark results"""
    results_path = Path(results_dir)

    if not results_path.exists():
        console.print(f"[red]Results directory {results_dir} does not exist[/red]")
        raise typer.Exit(1)

    # Count files
    json_files = list(results_path.rglob("*.json"))
    csv_files = list(results_path.rglob("*.csv"))
    plot_files = list(results_path.rglob("*.png"))

    table = Table(title="Benchmark Results Status")
    table.add_column("File Type", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Latest", style="green")

    if json_files:
        latest_json = max(json_files, key=lambda f: f.stat().st_mtime)
        table.add_row("JSON Results", str(len(json_files)), str(latest_json.name))
    else:
        table.add_row("JSON Results", "0", "None")

    if csv_files:
        latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
        table.add_row("CSV Results", str(len(csv_files)), str(latest_csv.name))
    else:
        table.add_row("CSV Results", "0", "None")

    if plot_files:
        latest_plot = max(plot_files, key=lambda f: f.stat().st_mtime)
        table.add_row("Plot Files", str(len(plot_files)), str(latest_plot.name))
    else:
        table.add_row("Plot Files", "0", "None")

    console.print(table)


@app.command()
def run_all(
    url: str = typer.Option(
        "http://localhost:8080", "--url", "-u", help="Full Router URL with protocol"
    ),
    model: str = typer.Option("adaptive-go-api", "--model", "-m", help="Model name"),
    concurrency: str | None = typer.Option(
        None, "--concurrency", "-c", help="Comma-separated concurrency levels"
    ),
    # Audio Input
    audio_length_mean: float | None = typer.Option(
        None, help="Mean length of audio data in seconds"
    ),
    audio_length_stddev: float | None = typer.Option(
        None, help="Standard deviation of audio length"
    ),
    audio_format: str | None = typer.Option(None, help="Audio format (wav, mp3)"),
    audio_depths: str | None = typer.Option(
        None, help="Comma-separated audio bit depths"
    ),
    audio_sample_rates: str | None = typer.Option(
        None, help="Comma-separated audio sample rates"
    ),
    audio_num_channels: int | None = typer.Option(
        None, help="Number of audio channels (1 or 2)"
    ),
    # Endpoint
    model_selection_strategy: str | None = typer.Option(
        None, help="Model selection strategy (round_robin, random)"
    ),
    backend: str | None = typer.Option(None, help="Backend (tensorrtllm, vllm)"),
    endpoint: str | None = typer.Option(None, help="Custom endpoint"),
    endpoint_type: str | None = typer.Option(None, help="Endpoint type"),
    server_metrics_url: str | None = typer.Option(
        None, help="Comma-separated server metrics URLs"
    ),
    streaming: bool = typer.Option(False, help="Enable streaming API"),
    # Image Input
    image_width_mean: int | None = typer.Option(None, help="Mean width of images"),
    image_width_stddev: int | None = typer.Option(
        None, help="Standard deviation of image width"
    ),
    image_height_mean: int | None = typer.Option(None, help="Mean height of images"),
    image_height_stddev: int | None = typer.Option(
        None, help="Standard deviation of image height"
    ),
    image_format: str | None = typer.Option(None, help="Image format (png, jpeg)"),
    # Input
    batch_size_audio: int | None = typer.Option(None, help="Audio batch size"),
    batch_size_image: int | None = typer.Option(None, help="Image batch size"),
    batch_size_text: int | None = typer.Option(None, help="Text batch size"),
    extra_inputs: str | None = typer.Option(
        None, help="Extra inputs in 'key:value' format"
    ),
    goodput: str | None = typer.Option(None, help="Goodput constraints"),
    header: str | None = typer.Option(None, help="Custom headers"),
    input_file: str | None = typer.Option(None, help="Input file path"),
    num_dataset_entries: int | None = typer.Option(
        None, "--num-dataset-entries", "--num-prompts", help="Number of unique payloads"
    ),
    num_prefix_prompts: int | None = typer.Option(
        None, help="Number of prefix prompts"
    ),
    output_tokens_mean: int | None = typer.Option(
        None, help="Mean number of output tokens"
    ),
    output_tokens_mean_deterministic: bool = typer.Option(
        False, help="Enable deterministic output tokens"
    ),
    output_tokens_stddev: int | None = typer.Option(
        None, help="Standard deviation of output tokens"
    ),
    random_seed: int | None = typer.Option(None, help="Random seed"),
    grpc_method: str | None = typer.Option(None, help="gRPC method name"),
    synthetic_input_tokens_mean: int | None = typer.Option(
        None, help="Mean synthetic input tokens"
    ),
    synthetic_input_tokens_stddev: int | None = typer.Option(
        None, help="Standard deviation of synthetic input tokens"
    ),
    prefix_prompt_length: int | None = typer.Option(None, help="Prefix prompt length"),
    warmup_request_count: int | None = typer.Option(
        None, help="Number of warmup requests"
    ),
    # Other
    verbose: bool = typer.Option(False, help="Enable verbose mode"),
    # Output
    artifact_dir: str | None = typer.Option(None, help="Artifact directory"),
    checkpoint_dir: str | None = typer.Option(None, help="Checkpoint directory"),
    generate_plots: bool = typer.Option(False, help="Generate plots"),
    enable_checkpointing: bool = typer.Option(False, help="Enable checkpointing"),
    profile_export_file: str | None = typer.Option(
        None, help="Profile export file path"
    ),
    # Profiling
    measurement_interval: int | None = typer.Option(
        None, help="Measurement interval in milliseconds"
    ),
    request_count: int | None = typer.Option(None, help="Number of requests"),
    request_rate: float | None = typer.Option(None, help="Request rate"),
    fixed_schedule: str | None = typer.Option(None, help="Fixed schedule file"),
    stability_percentage: float | None = typer.Option(
        None, help="Stability percentage"
    ),
    # Session
    num_sessions: int | None = typer.Option(None, help="Number of sessions"),
    session_concurrency: int | None = typer.Option(None, help="Session concurrency"),
    session_delay_ratio: float | None = typer.Option(None, help="Session delay ratio"),
    session_turn_delay_mean: float | None = typer.Option(
        None, help="Mean session turn delay"
    ),
    session_turn_delay_stddev: float | None = typer.Option(
        None, help="Standard deviation of session turn delay"
    ),
    session_turns_mean: int | None = typer.Option(None, help="Mean session turns"),
    session_turns_stddev: int | None = typer.Option(
        None, help="Standard deviation of session turns"
    ),
    # Tokenizer
    tokenizer: str | None = typer.Option(None, help="Tokenizer name or path"),
    tokenizer_revision: str | None = typer.Option(None, help="Tokenizer revision"),
    tokenizer_trust_remote_code: bool = typer.Option(
        False, help="Trust remote tokenizer code"
    ),
) -> None:
    """Run benchmarks and analyze results in one command"""
    console.print(
        Panel(
            "[bold blue]Running Complete Benchmark Pipeline[/bold blue]", expand=False
        )
    )

    # Create benchmark parameters
    params = create_benchmark_params(
        url=url,
        model=model,
        concurrency=concurrency,
        check_health=True,  # Always check health in run_all
        audio_length_mean=audio_length_mean,
        audio_length_stddev=audio_length_stddev,
        audio_format=audio_format,
        audio_depths=audio_depths,
        audio_sample_rates=audio_sample_rates,
        audio_num_channels=audio_num_channels,
        model_selection_strategy=model_selection_strategy,
        backend=backend,
        endpoint=endpoint,
        endpoint_type=endpoint_type,
        server_metrics_url=server_metrics_url,
        streaming=streaming,
        image_width_mean=image_width_mean,
        image_width_stddev=image_width_stddev,
        image_height_mean=image_height_mean,
        image_height_stddev=image_height_stddev,
        image_format=image_format,
        batch_size_audio=batch_size_audio,
        batch_size_image=batch_size_image,
        batch_size_text=batch_size_text,
        extra_inputs=extra_inputs,
        goodput=goodput,
        header=header,
        input_file=input_file,
        num_dataset_entries=num_dataset_entries,
        num_prefix_prompts=num_prefix_prompts,
        output_tokens_mean=output_tokens_mean,
        output_tokens_mean_deterministic=output_tokens_mean_deterministic,
        output_tokens_stddev=output_tokens_stddev,
        random_seed=random_seed,
        grpc_method=grpc_method,
        synthetic_input_tokens_mean=synthetic_input_tokens_mean,
        synthetic_input_tokens_stddev=synthetic_input_tokens_stddev,
        prefix_prompt_length=prefix_prompt_length,
        warmup_request_count=warmup_request_count,
        verbose=verbose,
        artifact_dir=artifact_dir,
        checkpoint_dir=checkpoint_dir,
        generate_plots=generate_plots,
        enable_checkpointing=enable_checkpointing,
        profile_export_file=profile_export_file,
        measurement_interval=measurement_interval,
        request_count=request_count,
        request_rate=request_rate,
        fixed_schedule=fixed_schedule,
        stability_percentage=stability_percentage,
        num_sessions=num_sessions,
        session_concurrency=session_concurrency,
        session_delay_ratio=session_delay_ratio,
        session_turn_delay_mean=session_turn_delay_mean,
        session_turn_delay_stddev=session_turn_delay_stddev,
        session_turns_mean=session_turns_mean,
        session_turns_stddev=session_turns_stddev,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        tokenizer_trust_remote_code=tokenizer_trust_remote_code,
    )

    # Run benchmarks
    console.print("\n[blue]Step 1: Running benchmarks[/blue]")
    _run_benchmark_with_params(params)

    # Analyze results
    console.print("\n[blue]Step 2: Analyzing results[/blue]")
    analyze(results_dir="./results")

    console.print("\n[green]Complete pipeline finished![/green]")


def main() -> None:
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()
