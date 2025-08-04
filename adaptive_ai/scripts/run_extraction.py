#!/usr/bin/env python3
"""
Simple runner script for model extraction with environment loading.
"""

import os
import sys
from pathlib import Path

# Add the script directory to Python path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Load environment variables if .env file exists
env_file = script_dir / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

# Import and run the main extraction script
from extract_provider_models import main

if __name__ == "__main__":
    main()