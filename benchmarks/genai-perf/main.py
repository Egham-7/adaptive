#!/usr/bin/env python3
"""
GenAI Performance Benchmarking Tool
Consolidated script for running benchmarks and analyzing results.
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import requests
import subprocess
import time
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from typing import Optional, List

# Set style for better looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

app = typer.Typer(help="GenAI Performance Benchmarking Tool")
console = Console()


class GenAIPerfAnalyzer:
    def __init__(self, results_dir="./results"):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def parse_genai_perf_csv(self, csv_path):
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

    def parse_genai_perf_json(self, json_path):
        """Parse GenAI-Perf JSON results file"""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            metrics = {}
            if "service_kind" in data:
                for key, value in data.items():
                    if key == "request_throughput":
                        metrics["throughput_tps"] = value
                    elif key == "time_to_first_token":
                        metrics["ttft_ms"] = value.get("mean", 0)
                    elif key == "inter_token_latency":
                        metrics["itl_ms"] = value.get("mean", 0)
                    elif key == "request_latency":
                        metrics["e2e_latency_ms"] = value.get("mean", 0)
            else:
                metrics = data
            return metrics
        except Exception as e:
            console.print(f"[red]Error parsing {json_path}: {e}[/red]")
            return None

    def extract_test_info(self, filepath):
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

    def scan_results_directory(self):
        """Scan results directory for benchmark files"""
        results = []

        csv_files = list(self.results_dir.rglob("*_genai_perf.csv"))
        for csv_file in csv_files:
            metrics = self.parse_genai_perf_csv(csv_file)
            if metrics:
                test_info = self.extract_test_info(csv_file)
                result = {**metrics, **test_info, "source_file": str(csv_file)}
                results.append(result)

        json_files = list(self.results_dir.rglob("*.json"))
        for json_file in json_files:
            if "genai_perf" not in json_file.stem:
                metrics = self.parse_genai_perf_json(json_file)
                if metrics:
                    test_info = self.extract_test_info(json_file)
                    result = {**metrics, **test_info, "source_file": str(json_file)}
                    results.append(result)

        return pd.DataFrame(results)

    def generate_performance_plots(self, df):
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

    def generate_summary_report(self, df):
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

    def save_detailed_results(self, df):
        """Save detailed results to CSV"""
        if df.empty:
            console.print("[yellow]No data to save[/yellow]")
            return

        csv_path = self.results_dir / "go_api_benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        console.print(f"[green]✓ Detailed results saved to {csv_path}[/green]")

    def run_analysis(self):
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

    def run_genai_perf_command(self, **kwargs) -> bool:
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

        # Add optional parameters
        for key, value in kwargs.items():
            if key.startswith("extra_inputs_"):
                cmd.extend(["--extra-inputs", f"{key[13:]}:{value}"])
            elif key == "profile_export_file":
                cmd.extend(["--profile-export-file", str(value)])
            elif key == "artifact_dir":
                cmd.extend(["--artifact-dir", str(value)])
            else:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        console.print(f"[dim]Running command: {' '.join(cmd)}[/dim]")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
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

    def run_simple_benchmark(self, concurrency_levels: List[int] = [1, 5, 10, 25]):
        """Run simple benchmark without tokenizer dependencies"""
        console.print("[blue]Running Simple Benchmark (No Tokenizer)[/blue]")

        test_scenarios = [
            ("quick_response", 50),
            ("medium_response", 150),
            ("long_response", 300),
        ]

        for test_name, max_tokens in test_scenarios:
            console.print(
                f"\n[bold]Testing: {test_name} (max_tokens: {max_tokens})[/bold]"
            )

            for concurrency in concurrency_levels:
                console.print(f"  Concurrency: {concurrency}")

                success = self.run_genai_perf_command(
                    num_prompts=30,
                    concurrency=concurrency,
                    extra_inputs_max_tokens=max_tokens,
                    extra_inputs_temperature=0.7,
                    measurement_interval=8000,
                    profile_export_file=f"simple_{test_name}_c{concurrency}.json",
                    artifact_dir=self.results_dir
                    / f"simple_{test_name}_c{concurrency}_artifacts",
                )

                if success:
                    console.print(
                        f"    [green]✓ Completed concurrency {concurrency}[/green]"
                    )
                else:
                    console.print(f"    [red]✗ Failed concurrency {concurrency}[/red]")

                time.sleep(2)  # Small delay between tests

            console.print(f"[green]✓ Completed {test_name}[/green]")


@app.command()
def benchmark(
    url: str = typer.Option(
        "http://localhost:8080", "--url", "-u", help="Full Router URL with protocol"
    ),
    model: str = typer.Option("adaptive-go-api", "--model", "-m", help="Model name"),
    concurrency: Optional[str] = typer.Option(
        None,
        "--concurrency",
        "-c",
        help="Comma-separated concurrency levels (e.g., 1,5,10)",
    ),
    check_health: bool = typer.Option(
        True, "--check-health/--no-check-health", help="Check API health before running"
    ),
):
    """Run GenAI-Perf benchmarks"""
    console.print(
        Panel("[bold blue]GenAI-Perf Benchmarking Tool[/bold blue]", expand=False)
    )
    console.print(f"Router URL: {url}")
    console.print(f"Model: {model}")

    benchmarker = GenAIPerfBenchmarker(url, model)

    if check_health:
        console.print("\n[yellow]Checking API health...[/yellow]")
        if not benchmarker.check_api_health():
            console.print(f"[red]API not accessible at {url}[/red]")
            console.print("Please ensure your API is running and accessible.")
            raise typer.Exit(1)
        console.print("[green]✓ API is accessible[/green]")

    concurrency_levels = [1, 5, 10, 25]
    if concurrency:
        try:
            concurrency_levels = [int(c.strip()) for c in concurrency.split(",")]
        except ValueError:
            console.print("[red]Invalid concurrency levels format[/red]")
            raise typer.Exit(1)

    console.print(f"\nConcurrency levels: {concurrency_levels}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running benchmarks...", total=None)
        benchmarker.run_simple_benchmark(concurrency_levels)
        progress.update(task, description="Benchmarks completed")

    console.print("\n[green]Benchmarking completed![/green]")
    console.print(f"Results saved in: {benchmarker.results_dir}")


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
):
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
):
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
    concurrency: Optional[str] = typer.Option(
        None, "--concurrency", "-c", help="Comma-separated concurrency levels"
    ),
):
    """Run benchmarks and analyze results in one command"""
    console.print(
        Panel(
            "[bold blue]Running Complete Benchmark Pipeline[/bold blue]", expand=False
        )
    )

    # Run benchmarks
    console.print("\n[blue]Step 1: Running benchmarks[/blue]")
    benchmark(url=url, model=model, concurrency=concurrency)

    # Analyze results
    console.print("\n[blue]Step 2: Analyzing results[/blue]")
    analyze()

    console.print("\n[green]Complete pipeline finished![/green]")


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()
