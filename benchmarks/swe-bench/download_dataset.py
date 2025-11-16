#!/usr/bin/env python3
"""
Download SWE-bench Lite dataset from Hugging Face.

This script downloads the dataset using the datasets library.
"""

import json
import sys
from pathlib import Path

try:
    from datasets import load_dataset  # type: ignore[import-untyped]

    print("Downloading SWE-bench Lite dataset from Hugging Face...")
    print("This may take a few minutes on first run...\n")

    # Load the dataset
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    print(f"✓ Downloaded {len(dataset)} instances")

    # Convert to list of dicts
    instances = []
    for item in dataset:
        instances.append(dict(item))

    # Save to JSON
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "swe-bench_lite.json"

    with open(output_file, "w") as f:
        json.dump(instances, f, indent=2)

    print(f"✓ Saved to: {output_file}")
    print("\nDataset ready! You can now run:")
    print("  uv run python run_benchmark.py --quick")

except ImportError:
    print("Error: 'datasets' library not found")
    print("\nInstalling required package...")
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True)
    print("\nPlease run this script again:")
    print("  uv run python download_dataset.py")
    sys.exit(1)

except Exception as e:
    print(f"Error downloading dataset: {str(e)}")
    print("\nAlternative: Download manually from:")
    print("  https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite")
    sys.exit(1)
