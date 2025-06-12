#!/usr/bin/env python3
"""
Upload script for the quantized prompt task and complexity classifier.

This script uploads the quantized model files to Hugging Face Hub with
proper configuration and documentation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi, create_repo, upload_file
    from transformers import AutoTokenizer, AutoConfig
except ImportError as e:
    print(
        f"CRITICAL: Missing required packages: {e}\n"
        "Please install: pip install huggingface_hub transformers",
        file=sys.stderr,
    )
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_model_files() -> bool:
    """Validate that all required model files are present."""
    required_files = [
        "model_quantized.onnx",
        "config.json",
        "README.md",
        ".gitattributes"
    ]

    missing_files = []
    for file_name in required_files:
        if not Path(file_name).exists():
            missing_files.append(file_name)

    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False

    logger.info("✅ All required files found")
    return True


def check_model_size() -> str:
    """Check and report the quantized model size."""
    model_path = Path("model_quantized.onnx")
    if not model_path.exists():
        return "Unknown"

    size_bytes = model_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    if size_mb < 1024:
        return f"{size_mb:.1f} MB"
    else:
        size_gb = size_mb / 1024
        return f"{size_gb:.2f} GB"


def create_tokenizer_files():
    """Create tokenizer files from the original model if they don't exist."""
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt"
    ]

    missing_tokenizer_files = [f for f in tokenizer_files if not Path(f).exists()]

    if missing_tokenizer_files:
        logger.info("Creating missing tokenizer files from original model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "nvidia/prompt-task-and-complexity-classifier"
            )
            tokenizer.save_pretrained("./")
            logger.info("✅ Tokenizer files created successfully")
        except Exception as e:
            logger.error(f"Failed to create tokenizer files: {e}")
            return False

    return True


def upload_model_to_hub(
    repo_name: str,
    private: bool = False,
    commit_message: str = "Upload quantized ONNX model"
) -> bool:
    """
    Upload the model files to Hugging Face Hub.

    Args:
        repo_name: Repository name (e.g., "username/model-name")
        private: Whether to make the repository private
        commit_message: Commit message for the upload

    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize API
        api = HfApi()

        # Create repository
        logger.info(f"Creating repository: {repo_name}")
        create_repo(
            repo_id=repo_name,
            token=None,  # Uses HF_TOKEN env var or cached token
            private=private,
            repo_type="model",
            exist_ok=True
        )

        # Files to upload
        files_to_upload = [
            "model_quantized.onnx",
            "config.json",
            "README.md",
            ".gitattributes",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt"
        ]

        # Optional files
        optional_files = ["test_model.py"]

        # Add optional files if they exist
        for file_name in optional_files:
            if Path(file_name).exists():
                files_to_upload.append(file_name)

        # Upload each file
        logger.info("Starting file uploads...")
        for file_name in files_to_upload:
            file_path = Path(file_name)
            if file_path.exists():
                logger.info(f"📤 Uploading {file_name}")
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_name,
                    repo_id=repo_name,
                    commit_message=f"{commit_message} - {file_name}"
                )
            else:
                logger.warning(f"⚠️  File not found, skipping: {file_name}")

        logger.info(f"✅ Upload completed successfully!")
        logger.info(f"🔗 Model available at: https://huggingface.co/{repo_name}")
        return True

    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload quantized prompt task complexity classifier to Hugging Face Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "repo_name",
        type=str,
        help="Repository name (e.g., 'username/prompt-task-complexity-classifier-quantized')"
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )

    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload quantized ONNX model for fast CPU inference",
        help="Commit message for the upload"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip validation checks and force upload"
    )

    args = parser.parse_args()

    logger.info("🚀 Starting upload process for quantized model")
    logger.info(f"📦 Target repository: {args.repo_name}")

    # Validate repository name format
    if "/" not in args.repo_name:
        logger.error("Repository name must be in format 'username/model-name'")
        sys.exit(1)

    # Check current directory
    if not Path("config.json").exists():
        logger.error(
            "This script must be run from the model directory containing config.json\n"
            "Current directory does not appear to contain model files."
        )
        sys.exit(1)

    # Validate files unless forced
    if not args.force:
        if not validate_model_files():
            logger.error("File validation failed. Use --force to skip validation.")
            sys.exit(1)

    # Create tokenizer files if needed
    if not create_tokenizer_files():
        logger.error("Failed to create tokenizer files")
        sys.exit(1)

    # Report model information
    model_size = check_model_size()
    logger.info(f"📊 Model size: {model_size}")

    # Load and display model config info
    try:
        with open("config.json", "r") as f:
            config = json.load(f)

        logger.info(f"🔧 Model type: {config.get('model_type', 'Unknown')}")
        logger.info(f"🎯 Quantized: {config.get('quantized', False)}")
        logger.info(f"💻 Optimized for: {config.get('optimized_for', 'Unknown')}")

        if "target_sizes" in config:
            total_outputs = sum(config["target_sizes"].values())
            logger.info(f"📈 Total output dimensions: {total_outputs}")

    except Exception as e:
        logger.warning(f"Could not read config details: {e}")

    # Upload to Hub
    success = upload_model_to_hub(
        repo_name=args.repo_name,
        private=args.private,
        commit_message=args.commit_message
    )

    if success:
        logger.info("🎉 Upload completed successfully!")
        logger.info("\n" + "="*60)
        logger.info("📋 Next steps:")
        logger.info(f"   1. Visit: https://huggingface.co/{args.repo_name}")
        logger.info("   2. Test the model with the provided test script")
        logger.info("   3. Update the model card if needed")
        logger.info("   4. Share your quantized model!")
        logger.info("="*60)
    else:
        logger.error("❌ Upload failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
