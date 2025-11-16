"""
SWE-bench CLI integration for submitting and retrieving evaluation results.

This module provides utilities for interacting with the SWE-bench API via sb-cli.
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SWEBenchClient:
    """Client for interacting with SWE-bench API via sb-cli."""

    def __init__(self, api_key: str):
        """
        Initialize SWE-bench client.

        Args:
            api_key: SWE-bench API key
        """
        self.api_key = api_key

    def test_connection(self) -> bool:
        """
        Test connection to SWE-bench API.

        Returns:
            True if connection successful
        """
        try:
            result = subprocess.run(
                ["sb-cli", "get-quotas", "--api_key", self.api_key],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                logger.info("✓ SWE-bench API connection successful")
                logger.info(f"Quotas: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"Connection failed: {result.stderr}")
                return False

        except FileNotFoundError:
            logger.error(
                "sb-cli not found. Install with: pip install sb-cli\n"
                "Or add to pyproject.toml and run: uv sync"
            )
            return False
        except Exception as e:
            logger.error(f"Error testing connection: {str(e)}")
            return False

    def submit_predictions(
        self,
        dataset: str,
        split: str,
        predictions_path: Path,
        run_id: str,
    ) -> bool:
        """
        Submit predictions to SWE-bench for evaluation.

        Args:
            dataset: Dataset name (e.g., "swe-bench_lite", "swe-bench_verified")
            split: Dataset split ("dev" or "test")
            predictions_path: Path to predictions JSON file
            run_id: Unique identifier for this run

        Returns:
            True if submission successful
        """
        try:
            logger.info("Submitting predictions to SWE-bench...")
            logger.info(f"  Dataset: {dataset} ({split})")
            logger.info(f"  Run ID: {run_id}")
            logger.info(f"  Predictions: {predictions_path}")

            result = subprocess.run(
                [
                    "sb-cli",
                    "submit",
                    dataset,
                    split,
                    "--predictions_path",
                    str(predictions_path),
                    "--run_id",
                    run_id,
                    "--api_key",
                    self.api_key,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                logger.info("✓ Predictions submitted successfully")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"Submission failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error submitting predictions: {str(e)}")
            return False

    def get_report(
        self,
        dataset: str,
        split: str,
        run_id: str,
        output_path: Path | None = None,
    ) -> dict[str, Any] | None:
        """
        Get evaluation report from SWE-bench.

        Args:
            dataset: Dataset name
            split: Dataset split
            run_id: Run identifier
            output_path: Optional path to save report JSON

        Returns:
            Report dictionary or None if failed
        """
        try:
            logger.info(f"Retrieving report for run: {run_id}")

            result = subprocess.run(
                [
                    "sb-cli",
                    "get-report",
                    dataset,
                    split,
                    run_id,
                    "--api_key",
                    self.api_key,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                # Parse JSON output
                report_text = result.stdout.strip()

                # sb-cli may output status messages before JSON
                # Try to find the JSON part
                json_start = report_text.find("{")
                if json_start >= 0:
                    report_json = report_text[json_start:]
                    report = json.loads(report_json)
                else:
                    # If no JSON found, it might be a status message
                    logger.info(f"Report status: {report_text}")
                    return None

                # Save if requested
                if output_path:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "w") as f:
                        json.dump(report, f, indent=2)

                return dict(report)
            else:
                logger.warning(f"Could not retrieve report: {result.stderr}")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing report JSON: {str(e)}")
            logger.debug(f"Output was: {result.stdout}")
            return None
        except Exception as e:
            logger.error(f"Error getting report: {str(e)}")
            return None

    def wait_for_results(
        self,
        dataset: str,
        split: str,
        run_id: str,
        timeout_seconds: int = 3600,
        poll_interval: int = 30,
    ) -> dict[str, Any] | None:
        """
        Wait for evaluation results to be ready.

        Args:
            dataset: Dataset name
            split: Dataset split
            run_id: Run identifier
            timeout_seconds: Maximum time to wait
            poll_interval: Seconds between status checks

        Returns:
            Report dictionary when ready, or None if timeout
        """
        start_time = time.time()
        attempts = 0

        logger.info(f"Waiting for evaluation results (timeout: {timeout_seconds}s)...")

        while (time.time() - start_time) < timeout_seconds:
            attempts += 1

            # Try to get report
            report = self.get_report(dataset, split, run_id)

            if report:
                # Check if evaluation is complete
                status = report.get("status", "unknown")

                if status == "completed":
                    logger.info(f"✓ Evaluation completed after {attempts} attempts")
                    return report
                elif status in ["failed", "error"]:
                    logger.error(f"Evaluation failed with status: {status}")
                    return None
                else:
                    logger.info(f"  Status: {status} (attempt {attempts})")

            # Wait before next check
            time.sleep(poll_interval)

        logger.error(f"Timeout waiting for results after {timeout_seconds}s")
        return None

    def list_runs(self, dataset: str, split: str) -> list[str]:
        """
        List all runs for a dataset.

        Args:
            dataset: Dataset name
            split: Dataset split

        Returns:
            List of run IDs
        """
        try:
            result = subprocess.run(
                [
                    "sb-cli",
                    "list-runs",
                    dataset,
                    split,
                    "--api_key",
                    self.api_key,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                # Parse run IDs from output
                runs = []
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        runs.append(line.strip())
                return runs
            else:
                logger.error(f"Failed to list runs: {result.stderr}")
                return []

        except Exception as e:
            logger.error(f"Error listing runs: {str(e)}")
            return []


def create_predictions_file(predictions: list[dict[str, str]], output_path: Path) -> None:
    """
    Create predictions file in SWE-bench format.

    Args:
        predictions: List of predictions with instance_id and model_patch
        output_path: Path to save predictions JSON

    Format:
        [
            {
                "instance_id": "django__django-11099",
                "model_patch": "diff --git a/file.py ..."
            },
            ...
        ]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    logger.info(f"Saved {len(predictions)} predictions to {output_path}")
