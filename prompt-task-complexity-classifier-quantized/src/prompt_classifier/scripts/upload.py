#!/usr/bin/env python3
"""
Upload script for the quantized prompt task and complexity classifier.

This script prepares and uploads the quantized model files to the Hugging Face Hub
with proper configuration, documentation, and best practices.
"""

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys
import traceback

try:
    from huggingface_hub import HfApi
except ImportError as e:
    print(
        f"CRITICAL: Missing required packages: {e}\n"
        "Please install: pip install huggingface_hub transformers",
        file=sys.stderr,
    )
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _create_model_card_template(repo_id: str, original_model: str) -> str:
    """Creates a template for the README.md model card."""
    return f"""---
license: apache-2.0
language: en
library_name: optimum
tags:
- onnx
- quantized
- text-classification
- nvidia
- nemotron
pipeline_tag: text-classification
---

# Quantized ONNX model for {repo_id}

This repository contains the quantized ONNX version of the \
[{original_model}](https://huggingface.co/{original_model}) model.

## Model Description

This is a multi-headed model which classifies English text prompts across task \
types and complexity dimensions. This version has been quantized to `INT8` \
using dynamic quantization with the [🤗 Optimum](https://github.com/huggingface/optimum) \
library, resulting in a smaller footprint and faster CPU inference.

For more details on the model architecture, tasks, and complexity dimensions, \
please refer to the [original model card]\
(https://huggingface.co/{original_model}).

## How to Use

You can use this model directly with `optimum.onnxruntime` for accelerated \
inference.

First, install the required libraries:
```bash
pip install optimum[onnxruntime] transformers
```

Then, you can use the model in a pipeline:
```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

repo_id = "{repo_id}"
model = ORTModelForSequenceClassification.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# Note: The pipeline task is a simplification.
# For full multi-headed output, you need to process the logits manually.
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

prompt = "Write a mystery set in a small town where an everyday object goes missing."
results = classifier(prompt)
print(results)
```
"""


def _prepare_upload_folder(source_dir: Path, upload_dir: Path, repo_id: str) -> bool:
    """
    Prepares a temporary folder with all necessary files for upload to Hugging Face Hub.

    Args:
        source_dir: The local directory containing the quantized model and its
                    associated files (e.g., config.json, tokenizer files).
        upload_dir: The temporary directory where files will be staged for upload.
        repo_id: The full repository ID (e.g., "username/model-name").

    Returns:
        True if the folder was prepared successfully, False otherwise.
    """
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True)

    logger.info(f"Preparing files for upload in temporary directory: {upload_dir}")

    # 1. Copy essential files
    files_to_copy = [
        "model_quantized.onnx",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spm.model",  # For DeBERTa tokenizer
    ]
    for filename in files_to_copy:
        source_file = source_dir / filename
        if source_file.exists():
            shutil.copy2(source_file, upload_dir / filename)
        else:
            logger.warning(f"⚠️ Optional file not found, skipping: {filename}")

    # 2. Create or copy README.md
    source_readme = source_dir / "README.md"
    if source_readme.exists():
        shutil.copy2(source_readme, upload_dir / "README.md")
        logger.info("✅ Using existing README.md")
    else:
        logger.info("📄 No README.md found, generating a template...")
        original_model = "nvidia/prompt-task-and-complexity-classifier"
        readme_content = _create_model_card_template(repo_id, original_model)
        (upload_dir / "README.md").write_text(readme_content)

    # 3. Create .gitattributes for LFS
    git_attributes_path = upload_dir / ".gitattributes"
    git_attributes_content = "*.onnx filter=lfs diff=lfs merge=lfs -text\n"
    git_attributes_path.write_text(git_attributes_content)
    logger.info("✅ Created .gitattributes for ONNX LFS handling")

    # 4. Update config.json with ONNX-specific info
    try:
        config_path = upload_dir / "config.json"
        with config_path.open("r+") as f:
            config_data = json.load(f)
            config_data["framework"] = "onnx"
            config_data["model_type"] = "deberta-v2"
            if "tags" not in config_data:
                config_data["tags"] = []
            for tag in ["onnx", "quantized"]:
                if tag not in config_data["tags"]:
                    config_data["tags"].append(tag)

            f.seek(0)
            json.dump(config_data, f, indent=2)
            f.truncate()
        logger.info("🔧 Updated config.json with ONNX tags")
    except Exception as e:
        logger.error(f"❌ Could not update config.json: {e}")
        return False

    return True


def main() -> None:
    """
    Main function to parse arguments and execute the upload process.
    """
    parser = argparse.ArgumentParser(
        description="Upload quantized model to Hugging Face Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="Repository ID on the Hub (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make the repository private"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload quantized ONNX model",
        help="Commit message for the upload",
    )
    args = parser.parse_args()

    logger.info(f"🚀 Starting upload process for repository: {args.repo_id}")

    # Correctly set the source_directory to where the quantized model is outputted
    source_directory: Path = Path("./quantized_model_output")
    upload_directory: Path = source_directory / "upload_temp"

    # Basic validation
    if not (source_directory / "model_quantized.onnx").exists():
        logger.error(
            f"❌ `model_quantized.onnx` not found in the expected directory: {source_directory}"
        )
        logger.error("Please ensure the quantization script has run successfully.")
        sys.exit(1)

    try:
        # Prepare all files in a temporary folder
        if not _prepare_upload_folder(source_directory, upload_directory, args.repo_id):
            raise RuntimeError("Failed to prepare files for upload.")

        # Upload the entire prepared folder
        api = HfApi()
        logger.info(f"📤 Uploading folder contents to {args.repo_id}...")
        api.upload_folder(
            folder_path=str(upload_directory),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
            create_pr=False,
        )

        logger.info("=" * 60)
        logger.info("🎉 Upload complete!")
        logger.info(
            f"🔗 Your model is now available at: https://huggingface.co/{args.repo_id}"
        )
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during upload: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Clean up the temporary directory
        if upload_directory.exists():
            logger.info("Cleaning up temporary upload directory...")
            shutil.rmtree(upload_directory)


if __name__ == "__main__":
    main()
