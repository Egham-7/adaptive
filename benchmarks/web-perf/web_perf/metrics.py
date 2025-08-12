"""Performance metrics collection and reporting."""

import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import asdict

from .base import TestSession
from .config import Config


class MetricsCollector:
    """Collect and analyze performance metrics."""

    def __init__(self, config: Config):
        self.config = config
        self.sessions: List[TestSession] = []

    def add_session(self, session: TestSession):
        """Add a test session to metrics collection."""
        self.sessions.append(session)

    def get_endpoint_metrics(self, endpoint_name: str) -> Dict[str, Any]:
        """Get metrics for a specific endpoint across all sessions."""
        endpoint_results = []

        for session in self.sessions:
            endpoint_results.extend(
                [r for r in session.results if r.endpoint_name == endpoint_name]
            )

        if not endpoint_results:
            return {"error": f"No results found for endpoint '{endpoint_name}'"}

        response_times = [r.response_time for r in endpoint_results]
        successful_results = [r for r in endpoint_results if r.success]

        return {
            "endpoint_name": endpoint_name,
            "total_requests": len(endpoint_results),
            "successful_requests": len(successful_results),
            "success_rate": (len(successful_results) / len(endpoint_results)) * 100,
            "response_times": {
                "mean": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "p95": self._percentile(response_times, 95),
                "p99": self._percentile(response_times, 99),
                "min": min(response_times),
                "max": max(response_times),
                "std": (
                    statistics.stdev(response_times) if len(response_times) > 1 else 0
                ),
            },
            "error_rate": (
                (len(endpoint_results) - len(successful_results))
                / len(endpoint_results)
            )
            * 100,
            "average_request_size": statistics.mean(
                [r.request_size for r in endpoint_results]
            ),
            "average_response_size": statistics.mean(
                [r.response_size for r in endpoint_results]
            ),
        }

    def get_overall_metrics(self) -> Dict[str, Any]:
        """Get overall metrics across all sessions and endpoints."""
        all_results = []
        for session in self.sessions:
            all_results.extend(session.results)

        if not all_results:
            return {"error": "No test results available"}

        response_times = [r.response_time for r in all_results]
        successful_results = [r for r in all_results if r.success]

        # Group by endpoint
        endpoint_stats = defaultdict(list)
        for result in all_results:
            endpoint_stats[result.endpoint_name].append(result)

        return {
            "total_requests": len(all_results),
            "successful_requests": len(successful_results),
            "overall_success_rate": (len(successful_results) / len(all_results)) * 100,
            "overall_response_times": {
                "mean": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "p95": self._percentile(response_times, 95),
                "p99": self._percentile(response_times, 99),
                "min": min(response_times),
                "max": max(response_times),
            },
            "endpoints_tested": len(endpoint_stats),
            "endpoint_breakdown": {
                name: len(results) for name, results in endpoint_stats.items()
            },
            "total_sessions": len(self.sessions),
            "total_duration": sum(session.duration for session in self.sessions),
        }

    def get_session_comparison(self) -> List[Dict[str, Any]]:
        """Compare metrics across sessions."""
        comparison = []

        for session in self.sessions:
            session_metrics = {
                "session_id": session.session_id,
                "start_time": session.start_time.isoformat(),
                "duration": session.duration,
                "total_requests": session.total_requests,
                "success_rate": session.success_rate,
                "average_response_time": session.average_response_time,
                "percentiles": session.get_percentiles(),
                "config": session.config,
            }
            comparison.append(session_metrics)

        return comparison

    def check_thresholds(self) -> Dict[str, Any]:
        """Check if metrics meet configured thresholds."""
        overall_metrics = self.get_overall_metrics()

        if "error" in overall_metrics:
            return {"error": "No metrics available for threshold checking"}

        thresholds = self.config.thresholds
        results: Dict[str, Any] = {"passed": True, "checks": {}}

        # Check success rate
        success_rate_check = (
            overall_metrics["overall_success_rate"] >= thresholds.success_rate
        )
        results["checks"]["success_rate"] = {
            "passed": success_rate_check,
            "actual": overall_metrics["overall_success_rate"],
            "threshold": thresholds.success_rate,
        }
        if not success_rate_check:
            results["passed"] = False

        # Check response time thresholds
        response_times = overall_metrics["overall_response_times"]

        for percentile in ["p50", "p95", "p99"]:
            actual = response_times.get(
                percentile.replace("p", "median" if percentile == "p50" else percentile)
            )
            threshold = getattr(thresholds.response_time, percentile)

            if percentile == "p50":
                actual = response_times["median"]
            else:
                actual = response_times[percentile]

            check_passed = actual <= threshold
            results["checks"][f"response_time_{percentile}"] = {
                "passed": check_passed,
                "actual": actual,
                "threshold": threshold,
            }
            if not check_passed:
                results["passed"] = False

        return results

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        n = len(sorted_data)
        index = (percentile / 100.0) * (n - 1)

        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to pandas DataFrame for analysis."""
        all_results = []

        for session in self.sessions:
            for result in session.results:
                row = asdict(result)
                row["session_id"] = session.session_id
                row["session_start"] = session.start_time
                row["timestamp"] = result.timestamp
                all_results.append(row)

        return pd.DataFrame(all_results)


class ReportGenerator:
    """Generate performance test reports."""

    def __init__(
        self, metrics_collector: MetricsCollector, output_dir: str = "results"
    ):
        self.metrics = metrics_collector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_html_report(self, filename: Optional[str] = None) -> str:
        """Generate HTML report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.html"

        report_path = self.output_dir / filename

        overall_metrics = self.metrics.get_overall_metrics()
        threshold_check = self.metrics.check_thresholds()

        # Get endpoint metrics
        endpoint_metrics = {}
        for session in self.metrics.sessions:
            for result in session.results:
                if result.endpoint_name not in endpoint_metrics:
                    endpoint_metrics[result.endpoint_name] = (
                        self.metrics.get_endpoint_metrics(result.endpoint_name)
                    )

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .metric-box {{ display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .pass {{ background-color: #d4edda; border-color: #c3e6cb; }}
                .fail {{ background-color: #f8d7da; border-color: #f5c6cb; }}
                .endpoint {{ margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Test Report</h1>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Total Sessions: {len(self.metrics.sessions)}</p>
            </div>
            
            <h2>Overall Metrics</h2>
            <div class="metric-box">
                <strong>Total Requests:</strong> {overall_metrics.get('total_requests', 0)}
            </div>
            <div class="metric-box">
                <strong>Success Rate:</strong> {overall_metrics.get('overall_success_rate', 0):.2f}%
            </div>
            <div class="metric-box">
                <strong>Average Response Time:</strong> {overall_metrics.get('overall_response_times', {}).get('mean', 0):.2f}ms
            </div>
            <div class="metric-box">
                <strong>P95 Response Time:</strong> {overall_metrics.get('overall_response_times', {}).get('p95', 0):.2f}ms
            </div>
            
            <h2>Threshold Checks</h2>
            <div class="{'pass' if threshold_check.get('passed') else 'fail'}">
                <strong>Overall Status:</strong> {'PASSED' if threshold_check.get('passed') else 'FAILED'}
            </div>
            
            <h2>Endpoint Breakdown</h2>
        """

        for name, metrics in endpoint_metrics.items():
            if "error" not in metrics:
                html_content += f"""
                <div class="endpoint">
                    <h3>{name}</h3>
                    <p><strong>Total Requests:</strong> {metrics['total_requests']}</p>
                    <p><strong>Success Rate:</strong> {metrics['success_rate']:.2f}%</p>
                    <p><strong>Average Response Time:</strong> {metrics['response_times']['mean']:.2f}ms</p>
                    <p><strong>P95 Response Time:</strong> {metrics['response_times']['p95']:.2f}ms</p>
                </div>
                """

        html_content += """
            </body>
            </html>
        """

        with open(report_path, "w") as f:
            f.write(html_content)

        return str(report_path)

    def generate_csv_report(self, filename: Optional[str] = None) -> str:
        """Generate CSV report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_data_{timestamp}.csv"

        report_path = self.output_dir / filename
        df = self.metrics.to_dataframe()
        df.to_csv(report_path, index=False)

        return str(report_path)

    def generate_charts(self) -> List[str]:
        """Generate performance charts."""
        df = self.metrics.to_dataframe()

        if df.empty:
            return []

        chart_files = []

        # Set style
        plt.style.use("seaborn-v0_8")

        # Response time distribution
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(df["response_time"], bins=50, alpha=0.7, edgecolor="black")
        plt.xlabel("Response Time (ms)")
        plt.ylabel("Frequency")
        plt.title("Response Time Distribution")

        # Response time by endpoint
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x="endpoint_name", y="response_time")
        plt.xlabel("Endpoint")
        plt.ylabel("Response Time (ms)")
        plt.title("Response Time by Endpoint")
        plt.xticks(rotation=45)

        plt.tight_layout()
        chart_path = (
            self.output_dir
            / f"response_times_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close()
        chart_files.append(str(chart_path))

        # Success rate over time (if multiple sessions)
        if len(self.metrics.sessions) > 1:
            plt.figure(figsize=(10, 6))

            session_success_rates = []
            session_times = []

            for session in self.metrics.sessions:
                session_success_rates.append(session.success_rate)
                session_times.append(session.start_time)

            import matplotlib.dates as mdates

            # Convert datetime to matplotlib dates
            session_times_mpl = mdates.date2num(session_times)
            plt.plot(session_times_mpl, session_success_rates, marker="o")
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            plt.xlabel("Session Start Time")
            plt.ylabel("Success Rate (%)")
            plt.title("Success Rate Over Time")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            chart_path = (
                self.output_dir
                / f"success_rate_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()
            chart_files.append(str(chart_path))

        return chart_files

    def generate_json_report(self, filename: Optional[str] = None) -> str:
        """Generate JSON report with all metrics."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_metrics_{timestamp}.json"

        report_path = self.output_dir / filename

        # Collect all metrics
        report_data: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "overall_metrics": self.metrics.get_overall_metrics(),
            "threshold_checks": self.metrics.check_thresholds(),
            "session_comparison": self.metrics.get_session_comparison(),
            "endpoint_metrics": {},
        }

        # Add endpoint-specific metrics
        for session in self.metrics.sessions:
            for result in session.results:
                endpoint_name = result.endpoint_name
                if endpoint_name not in report_data["endpoint_metrics"]:
                    report_data["endpoint_metrics"][endpoint_name] = (
                        self.metrics.get_endpoint_metrics(endpoint_name)
                    )

        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        return str(report_path)
