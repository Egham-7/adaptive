"""
Dataset loading utilities for SWE-bench.

This module provides functions to load SWE-bench datasets from various sources.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_dataset_from_file(dataset_path: Path, max_instances: int = -1) -> list[dict[str, Any]]:
    """
    Load SWE-bench dataset from a local JSON file.

    Args:
        dataset_path: Path to dataset JSON file
        max_instances: Maximum number of instances to load (-1 for all)

    Returns:
        List of instance dictionaries
    """
    try:
        with open(dataset_path) as f:
            instances = json.load(f)

        logger.info(f"Loaded {len(instances)} instances from {dataset_path}")

        # Limit if requested
        if max_instances > 0 and len(instances) > max_instances:
            instances = instances[:max_instances]
            logger.info(f"Limited to {max_instances} instances")

        return instances

    except FileNotFoundError:
        logger.error(f"Dataset file not found: {dataset_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in dataset file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def download_dataset(dataset: str, output_dir: Path = Path("data")) -> Path:
    """
    Download SWE-bench dataset using sb-cli or provide instructions.

    Args:
        dataset: Dataset name (e.g., "lite", "verified", "full")
        output_dir: Directory to save dataset

    Returns:
        Path to downloaded dataset file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map dataset names
    dataset_map = {
        "lite": "swe-bench_lite",
        "verified": "swe-bench_verified",
        "full": "swe-bench",
    }

    dataset_name = dataset_map.get(dataset, dataset)
    output_file = output_dir / f"{dataset_name}.json"

    if output_file.exists():
        logger.info(f"Dataset already exists: {output_file}")
        return output_file

    logger.info(f"Dataset {dataset_name} not found locally.")
    logger.info("\nTo download the dataset, you have two options:")
    logger.info("\n1. Download from Hugging Face:")
    logger.info(f"   - Visit: https://huggingface.co/datasets/princeton-nlp/SWE-bench")
    logger.info(f"   - Download the {dataset_name} split")
    logger.info(f"   - Save to: {output_file}")
    logger.info("\n2. Use the SWE-bench repository:")
    logger.info(f"   git clone https://github.com/princeton-nlp/SWE-bench")
    logger.info(f"   # Follow instructions in their README to download datasets")

    raise FileNotFoundError(
        f"Dataset {dataset_name} not found. Please download it to {output_file}"
    )


def get_dataset_instances(dataset: str, max_instances: int = -1) -> list[dict[str, Any]]:
    """
    Get dataset instances, loading from file or using fallback.

    Args:
        dataset: Dataset name ("lite", "verified", "full")
        max_instances: Maximum number of instances (-1 for all)

    Returns:
        List of instance dictionaries
    """
    # Try to load from local file
    dataset_map = {
        "lite": "swe-bench_lite",
        "verified": "swe-bench_verified",
        "full": "swe-bench",
    }

    dataset_name = dataset_map.get(dataset, dataset)
    dataset_path = Path("data") / f"{dataset_name}.json"

    try:
        return load_dataset_from_file(dataset_path, max_instances)
    except FileNotFoundError:
        logger.warning(f"Dataset file not found: {dataset_path}")
        logger.info("Attempting to download...")
        dataset_path = download_dataset(dataset)
        return load_dataset_from_file(dataset_path, max_instances)


def validate_instance(instance: dict[str, Any]) -> bool:
    """
    Validate that an instance has required fields.

    Args:
        instance: Instance dictionary

    Returns:
        True if valid
    """
    required_fields = [
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
    ]

    for field in required_fields:
        if field not in instance:
            logger.error(f"Instance missing required field: {field}")
            return False

    return True


def get_instance_summary(instances: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Get summary statistics about a dataset.

    Args:
        instances: List of instances

    Returns:
        Dictionary with summary stats
    """
    repos = set()
    avg_problem_length = 0

    for instance in instances:
        if "repo" in instance:
            repos.add(instance["repo"])
        if "problem_statement" in instance:
            avg_problem_length += len(instance["problem_statement"])

    avg_problem_length = avg_problem_length / len(instances) if instances else 0

    return {
        "total_instances": len(instances),
        "unique_repos": len(repos),
        "repos": sorted(list(repos)),
        "avg_problem_length": int(avg_problem_length),
    }
