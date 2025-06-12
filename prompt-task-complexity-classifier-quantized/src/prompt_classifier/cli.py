#!/usr/bin/env python3
"""
Command Line Interface for the Prompt Task Complexity Classifier - Quantized

This module provides CLI commands for quantization, and uploading
the quantized model.
"""

import argparse
from pathlib import Path
import sys

from .classifier import QuantizedPromptClassifier
from .utils import (
    create_model_summary,
    format_results_for_display,
    setup_logging,
)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prompt Task Complexity Classifier - Quantized",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Quantize command
    quantize_parser = subparsers.add_parser(
        "quantize", help="Quantize the original model to ONNX format using static quantization"
    )
    quantize_parser.add_argument(
        "--model-id",
        type=str,
        default="nvidia/prompt-task-and-complexity-classifier",
        help="Hugging Face model ID to quantize",
    )
    quantize_parser.add_argument(
        "--output-dir",
        type=str,
        default="./quantized_model",
        help="Output directory for quantized model",
    )
    quantize_parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for ONNX export"
    )
    quantize_parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
        help="Sequence length for ONNX export",
    )
    quantize_parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=5000,
        help="Number of samples to use for static quantization calibration",
    )

    # Classify command
    classify_parser = subparsers.add_parser(
        "classify", help="Classify prompts using the quantized model"
    )
    classify_parser.add_argument("prompts", nargs="+", help="Prompts to classify")
    classify_parser.add_argument(
        "--model-path",
        type=str,
        default="./quantized_model",
        help="Path to the quantized model directory",
    )
    classify_parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )

    # Upload command
    upload_parser = subparsers.add_parser(
        "upload", help="Upload quantized model to Hugging Face Hub"
    )
    upload_parser.add_argument(
        "repo_name",
        type=str,
        help="Repository name (e.g., 'username/model-name')",
    )
    upload_parser.add_argument(
        "--model-path",
        type=str,
        default="./",
        help="Path to the quantized model directory",
    )
    upload_parser.add_argument(
        "--private", action="store_true", help="Make repository private"
    )
    upload_parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload quantized ONNX model",
        help="Commit message",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Show information about a quantized model"
    )
    info_parser.add_argument(
        "--model-path",
        type=str,
        default="./",
        help="Path to the quantized model directory",
    )

    # Global arguments
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress output except errors"
    )

    args = parser.parse_args()

    # Setup logging
    if args.quiet:
        log_level = "ERROR"
    elif args.verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    setup_logging(log_level)

    # Handle commands
    if args.command == "quantize":
        cmd_quantize(args)
    elif args.command == "classify":
        cmd_classify(args)
    elif args.command == "upload":
        cmd_upload(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_quantize(args: argparse.Namespace) -> None:
    """Handle quantize command."""
    from .scripts.quantization import main as quantize_main

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run quantization
    sys.argv = [
        "quantization.py",
        args.model_id,
        "--output_dir",
        str(output_dir),
        "--num_calibration_samples",
        str(args.num_calibration_samples),
    ]

    try:
        quantize_main()
        print(f"✅ Quantization completed! Files saved to: {output_dir}")
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        sys.exit(1)


def cmd_classify(args: argparse.Namespace) -> None:
    """Handle classify command."""
    import json

    print("🔍 Classifying prompts...")

    try:
        # Load model
        classifier = QuantizedPromptClassifier.from_pretrained(args.model_path)

        # Classify prompts
        results = classifier.classify_prompts(args.prompts)

        # Output results
        if args.output_format == "json":
            print(json.dumps(results, indent=2))
        else:
            for i, (prompt, result) in enumerate(
                zip(args.prompts, results, strict=False)
            ):
                print(f"\n📋 Prompt {i+1}: {prompt}")
                print("-" * 50)
                print(format_results_for_display(result))

    except Exception as e:
        print(f"❌ Classification failed: {e}")
        sys.exit(1)


def cmd_upload(args: argparse.Namespace) -> None:
    """Handle upload command."""
    import os

    from .scripts.upload import main as upload_main

    print(f"📤 Uploading model to {args.repo_name}...")

    # Change to model directory if specified
    original_cwd = os.getcwd()
    if args.model_path != "./":
        os.chdir(args.model_path)

    try:
        # Prepare arguments for upload script (without --model-path)
        sys.argv = [
            "upload_to_hf.py",
            args.repo_name,
            "--commit-message",
            args.commit_message,
        ]

        if args.private:
            sys.argv.append("--private")

        upload_main()
        print(
            "✅ Upload completed! Model available at: "
            f"https://huggingface.co/{args.repo_name}"
        )
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def cmd_info(args: argparse.Namespace) -> None:
    """Handle info command."""
    print("i  Model Information")
    print("=" * 50)

    try:
        summary = create_model_summary(args.model_path)

        print(f"Model Path: {summary['model_path']}")
        print(f"Model Size: {summary['model_size']}")
        print(f"Config Valid: {summary['config_valid']}")
        print(f"Parameters: {summary['total_parameters']}")

        print(f"\nFiles Present ({len(summary['files_present'])}):")
        for file in summary["files_present"]:
            print(f"  ✅ {file}")

        if summary["files_missing"]:
            print(f"\nFiles Missing ({len(summary['files_missing'])}):")
            for file in summary["files_missing"]:
                print(f"  ❌ {file}")

    except Exception as e:
        print(f"❌ Error getting model info: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
