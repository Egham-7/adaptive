"""Command-line interface for web performance testing."""

import sys
from pathlib import Path
from typing import Optional, Union
import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .base import SyncEndpointTester, AsyncEndpointTester
from .metrics import MetricsCollector, ReportGenerator

app = typer.Typer(help="Web Performance Testing Suite")
console = Console()


@app.command()
def validate(
    config_path: str = typer.Option(
        "config/endpoints.yaml", "--config", "-c", help="Path to configuration file"
    )
):
    """Validate configuration file."""
    try:
        config = load_config(config_path)
        issues = config.validate_configuration()

        if issues:
            console.print("[red]Configuration validation failed:[/red]")
            for issue in issues:
                console.print(f"  ❌ {issue}")
            sys.exit(1)
        else:
            console.print("[green]✅ Configuration validation passed[/green]")

            # Show configuration summary
            enabled_endpoints = config.get_enabled_endpoints()

            table = Table(title="Configuration Summary")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Base URL", config.base_url)
            table.add_row("Enabled Endpoints", str(len(enabled_endpoints)))
            table.add_row("Total Scenarios", str(len(config.scenarios)))
            table.add_row("Global Timeout", f"{config.global_settings.timeout}s")

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)


@app.command()
def list_endpoints(
    config_path: str = typer.Option(
        "config/endpoints.yaml", "--config", "-c", help="Path to configuration file"
    )
):
    """List configured endpoints."""
    try:
        config = load_config(config_path)

        table = Table(title="Configured Endpoints")
        table.add_column("Name", style="cyan")
        table.add_column("Method", style="green")
        table.add_column("Path", style="blue")
        table.add_column("Weight", style="magenta")
        table.add_column("Status", style="yellow")

        for name, endpoint in config.endpoints.items():
            status = "✅ Enabled" if endpoint.enabled else "❌ Disabled"
            table.add_row(
                name, endpoint.method, endpoint.path, str(endpoint.weight), status
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def list_scenarios(
    config_path: str = typer.Option(
        "config/endpoints.yaml", "--config", "-c", help="Path to configuration file"
    )
):
    """List available test scenarios."""
    try:
        config = load_config(config_path)

        table = Table(title="Test Scenarios")
        table.add_column("Name", style="cyan")
        table.add_column("Users", style="green")
        table.add_column("Spawn Rate", style="blue")
        table.add_column("Duration", style="magenta")

        for name, scenario in config.scenarios.items():
            table.add_row(
                name,
                str(scenario.users),
                str(scenario.spawn_rate),
                f"{scenario.duration}s",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def test(
    scenario: str = typer.Argument(..., help="Scenario name to run"),
    config_path: str = typer.Option(
        "config/endpoints.yaml", "--config", "-c", help="Path to configuration file"
    ),
    engine: str = typer.Option(
        "async", "--engine", "-e", help="Test engine to use: sync, async, or locust"
    ),
    report: bool = typer.Option(
        True, "--report/--no-report", help="Generate performance report"
    ),
    output_dir: str = typer.Option(
        "results", "--output", "-o", help="Output directory for results"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging (detailed request/response logs)",
    ),
):
    """Run performance test with specified scenario."""
    try:
        config = load_config(config_path)

        # Override verbose setting from CLI
        if verbose:
            config.global_settings.verbose = True

        # Validate scenario exists
        if scenario not in config.scenarios:
            console.print(f"[red]Scenario '{scenario}' not found[/red]")
            console.print("Available scenarios:")
            for name in config.scenarios.keys():
                console.print(f"  • {name}")
            sys.exit(1)

        # Validate configuration
        issues = config.validate_configuration()
        if issues:
            console.print("[red]Configuration issues found:[/red]")
            for issue in issues:
                console.print(f"  ❌ {issue}")
            sys.exit(1)

        console.print(f"[green]Starting performance test: {scenario}[/green]")
        console.print(f"Engine: {engine}")
        console.print(f"Configuration: {config_path}")

        # Run test based on engine
        if engine == "locust":
            from .locust_tests import ConfigurableLocustRunner

            runner = ConfigurableLocustRunner(config_path)

            report_file: Optional[str] = None
            if report:
                report_file = f"{output_dir}/locust_report_{scenario}.html"
                Path(output_dir).mkdir(exist_ok=True)

            runner.run_scenario(scenario, headless=True, html_report=report_file)

            console.print("[green]✅ Test completed[/green]")
            if report_file:
                console.print(f"Report generated: {report_file}")

        else:
            # Use built-in testers
            tester: Union[SyncEndpointTester, AsyncEndpointTester]
            if engine == "sync":
                tester = SyncEndpointTester(config, console)
            elif engine == "async":
                tester = AsyncEndpointTester(config, console)
            else:
                console.print(f"[red]Unknown engine: {engine}[/red]")
                sys.exit(1)

            # Run test
            session = tester.run_test(scenario)

            console.print("[green]✅ Test completed[/green]")
            console.print(f"Total requests: {session.total_requests}")
            console.print(f"Success rate: {session.success_rate:.2f}%")
            console.print(
                f"Average response time: {session.average_response_time:.2f}ms"
            )

            # Generate report if requested
            if report:
                metrics_collector = MetricsCollector(config)
                metrics_collector.add_session(session)

                report_generator = ReportGenerator(metrics_collector, output_dir)

                html_report = report_generator.generate_html_report()
                csv_report = report_generator.generate_csv_report()
                json_report = report_generator.generate_json_report()
                chart_files = report_generator.generate_charts()

                console.print("[green]Reports generated:[/green]")
                console.print(f"  HTML: {html_report}")
                console.print(f"  CSV: {csv_report}")
                console.print(f"  JSON: {json_report}")

                if chart_files:
                    console.print("  Charts:")
                    for chart in chart_files:
                        console.print(f"    • {chart}")

    except Exception as e:
        console.print(f"[red]Error running test: {e}[/red]")
        sys.exit(1)


@app.command()
def generate_locustfile(
    config_path: str = typer.Option(
        "config/endpoints.yaml", "--config", "-c", help="Path to configuration file"
    ),
    output: str = typer.Option(
        "locustfile.py", "--output", "-o", help="Output path for locustfile.py"
    ),
):
    """Generate standalone locustfile.py for use with locust command."""
    try:
        # Validate config first
        config = load_config(config_path)
        issues = config.validate_configuration()

        if issues:
            console.print("[red]Configuration issues found:[/red]")
            for issue in issues:
                console.print(f"  ❌ {issue}")
            sys.exit(1)

        from .locust_tests import create_locust_file

        create_locust_file(config_path, output)

        console.print(f"[green]✅ Locustfile created: {output}[/green]")
        console.print(f"Run with: [cyan]locust -f {output}[/cyan]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def init(
    path: str = typer.Option(
        "config/endpoints.yaml",
        "--path",
        "-p",
        help="Path where to create configuration file",
    )
):
    """Initialize a new configuration file."""
    config_path = Path(path)

    if config_path.exists():
        overwrite = typer.confirm(
            f"Configuration file {path} already exists. Overwrite?"
        )
        if not overwrite:
            console.print("[yellow]Aborted[/yellow]")
            return

    # Create directory if needed
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create sample configuration
    sample_config = """# Web Performance Testing Configuration

base_url: "https://llmadaptive.uk"

# Global test settings
global_settings:
  timeout: 30
  max_retries: 3
  verify_ssl: true
  headers:
    User-Agent: "WebPerf-Test-Suite/0.1.0"
    Content-Type: "application/json"

# Test scenarios
scenarios:
  quick_test:
    users: 5
    spawn_rate: 1
    duration: 30
  
  load_test:
    users: 50
    spawn_rate: 5
    duration: 300

# Endpoint definitions
endpoints:
  select_model:
    path: "/api/v1/select-model"
    method: "POST"
    description: "Model selection endpoint"
    test_data:
      - prompt: "What is the capital of France?"
        max_tokens: 100
        temperature: 0.7
    weight: 100
    enabled: true

# Performance thresholds
thresholds:
  response_time:
    p50: 500   # milliseconds
    p95: 2000
    p99: 5000
  success_rate: 95  # percentage
  rps:
    minimum: 1
    target: 10

# Reporting
reporting:
  output_dir: "results"
  generate_charts: true
  export_formats: ["html", "csv", "json"]
  chart_formats: ["png"]
"""

    with open(config_path, "w") as f:
        f.write(sample_config)

    console.print(f"[green]✅ Configuration file created: {path}[/green]")
    console.print(
        "Edit the configuration file to match your API endpoints and requirements."
    )
    console.print(f"Validate with: [cyan]web-perf validate --config {path}[/cyan]")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
