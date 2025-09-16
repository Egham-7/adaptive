#!/usr/bin/env python3
"""
Base configuration for LangGraph model extraction system.
Shared settings used by the LangGraph workflow.
"""

import os

# Budget Configuration
BUDGET_MODE = True
MAX_COST_LIMIT = 4.00  # Hard stop at $4 (user request)
TARGET_COST_RANGE = (2.50, 4.00)  # Expected cost range
MODEL = "gpt-4o-mini"  # Optimal cost/quality balance

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Note: API key will be checked at runtime, not import time

# Processing Configuration
CACHE_ENABLED = True  # Cache results to avoid reprocessing

# Fields to Extract (focus on gaps in YAML files)
FIELDS_TO_EXTRACT = [
    "description",  # Always empty in YAML files
    "max_context_tokens",  # Always null in YAML files
    "max_output_tokens",  # Always null in YAML files
    "task_type",  # Always empty in YAML files
    "complexity",  # Always empty in YAML files
    "supports_function_calling",  # Always null in YAML files
    "model_size_params",  # Always empty in YAML files
    "latency_tier",  # Always empty in YAML files
]

# Input/Output Paths
STRUCTURED_MODELS_PATH = "data/provider_models"
OUTPUT_PATH = "data/enhanced_models"
CACHE_PATH = "data/cache"

# Logging Configuration
LOG_LEVEL = "INFO"
SHOW_PROGRESS = True
SHOW_COSTS = True

# Model Pricing (per 1M tokens)
PRICING = {
    "gpt-4o-mini": {
        "input": 0.150,  # $0.150 per 1M input tokens
        "output": 0.600,  # $0.600 per 1M output tokens
    }
}

# Base Quality Control (detailed config in langgraph_config.py)
MIN_CONFIDENCE_SCORE = 0.7
