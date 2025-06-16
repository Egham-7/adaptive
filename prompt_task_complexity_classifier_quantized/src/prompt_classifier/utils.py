"""
Utility functions for the prompt task complexity classifier.

This module provides helper functions for model configuration,
file validation, and other common operations.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

# This is a forward reference to prevent circular imports during type checking
if TYPE_CHECKING:
    from .classifier import QuantizedPromptClassifier


def load_model_config(model_path: str | Path) -> dict[str, Any]:
    """
    Load model configuration from config.json file.

    Args:
        model_path: Path to the model directory

    Returns:
        Dictionary containing model configuration

    Raises:
        FileNotFoundError: If config.json is not found
        json.JSONDecodeError: If config.json is invalid
    """
    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    return cast(dict[str, Any], config)


def validate_model_files(model_path: str | Path) -> list[str]:
    """
    Validate that all required model files are present.

    Args:
        model_path: Path to the model directory

    Returns:
        List of missing files (empty if all files present)
    """
    model_path = Path(model_path)

    required_files = [
        "model_quantized.onnx",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spm.model",
    ]

    missing_files: list[str] = []
    for file_name in required_files:
        if not (model_path / file_name).exists():
            missing_files.append(file_name)

    return missing_files


def get_file_size(file_path: str | Path) -> str:
    """
    Get human-readable file size.

    Args:
        file_path: Path to the file

    Returns:
        Human-readable file size string
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return "N/A"

    size_bytes: float = float(file_path.stat().st_size)

    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            # For Bytes, don't show a decimal point
            if unit == "B":
                return f"{int(size_bytes)} {unit}"
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024

    return f"{size_bytes:.1f} TB"


def validate_config_structure(config: dict[str, Any]) -> bool:
    """
    Validate that config has required fields for the classifier.

    Args:
        config: Configuration dictionary

    Returns:
        True if config is valid, False otherwise
    """
    required_fields = [
        "target_sizes",
        "task_type_map",
        "weights_map",
        "divisor_map",
    ]

    for field in required_fields:
        if field not in config:
            return False

    # Validate target_sizes structure
    if not isinstance(config["target_sizes"], dict):
        return False

    expected_targets = [
        "task_type",
        "creativity_scope",
        "reasoning",
        "contextual_knowledge",
        "number_of_few_shots",
        "domain_knowledge",
        "no_label_reason",
        "constraint_ct",
    ]

    for target in expected_targets:
        if target not in config["target_sizes"]:
            return False

    return True


def format_results_for_display(results: dict[str, Any]) -> str:
    """
    Format classification results for human-readable display.

    Args:
        results: Classification results from the model for a single prompt.

    Returns:
        Formatted string representation
    """
    output_lines: list[str] = []

    # Task type
    if "task_type_1" in results:
        task_type = results["task_type_1"]
        confidence = results.get("task_type_prob", 0.0)
        output_lines.append(f"Task Type: {task_type} (confidence: {confidence:.3f})")

        if "task_type_2" in results and results["task_type_2"] != "NA":
            output_lines.append(f"Secondary Task: {results['task_type_2']}")

    # Complexity score
    if "prompt_complexity_score" in results:
        complexity = results["prompt_complexity_score"]
        output_lines.append(f"Complexity Score: {complexity:.3f}")

    # Individual dimensions
    dimensions = [
        ("Creativity", "creativity_scope"),
        ("Reasoning", "reasoning"),
        ("Context Knowledge", "contextual_knowledge"),
        ("Domain Knowledge", "domain_knowledge"),
        ("Few-shot Learning", "number_of_few_shots"),
        ("Constraints", "constraint_ct"),
    ]

    output_lines.append("-" * 20)
    for display_name, key in dimensions:
        if key in results:
            value = results[key]
            output_lines.append(f"{display_name:<20}: {value:.3f}")

    return "\n".join(output_lines)


def create_model_summary(model_path: str | Path) -> dict[str, Any]:
    """
    Create a summary of model information.

    Args:
        model_path: Path to the model directory

    Returns:
        Dictionary with model summary information
    """
    model_path = Path(model_path)

    summary: dict[str, Any] = {
        "model_path": str(model_path.resolve()),
        "files_present": [],
        "files_missing": [],
        "model_size": "N/A",
        "config_valid": False,
        "total_parameters": "N/A",
    }

    # Check files
    missing_files = validate_model_files(model_path)
    summary["files_missing"] = missing_files

    all_files = [
        "model_quantized.onnx",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spm.model",
    ]
    summary["files_present"] = [f for f in all_files if f not in missing_files]

    # Get model size
    onnx_path = model_path / "model_quantized.onnx"
    if onnx_path.exists():
        summary["model_size"] = get_file_size(onnx_path)

    # Validate config
    try:
        config = load_model_config(model_path)
        summary["config_valid"] = validate_config_structure(config)

        if "target_sizes" in config and isinstance(config["target_sizes"], dict):
            total_outputs = sum(config["target_sizes"].values())
            summary["total_parameters"] = f"{total_outputs} output dimensions"

    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return summary


def setup_logging(level: str = "INFO") -> None:
    """
    Setup basic logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    import logging

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def benchmark_inference_speed(
    classifier: "QuantizedPromptClassifier",
    test_prompts: list[str],
    num_runs: int = 5,
    warmup_runs: int = 2,
) -> dict[str, float]:
    """
    Benchmark inference speed of the classifier.

    Args:
        classifier: QuantizedPromptClassifier instance
        test_prompts: List of prompts to test with
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs

    Returns:
        Dictionary with timing statistics
    """
    import time

    import numpy as np

    # Warmup
    for _ in range(warmup_runs):
        classifier.classify_prompts(test_prompts)

    # Benchmark
    times: list[float] = []
    for _ in range(num_runs):
        start_time = time.time()
        classifier.classify_prompts(test_prompts)
        end_time = time.time()
        times.append(end_time - start_time)

    times_arr = np.array(times)
    mean_time = float(np.mean(times_arr))

    return {
        "mean_time": mean_time,
        "std_time": float(np.std(times_arr)),
        "min_time": float(np.min(times_arr)),
        "max_time": float(np.max(times_arr)),
        "throughput": len(test_prompts) / mean_time,
        "avg_per_prompt": mean_time / len(test_prompts),
    }
