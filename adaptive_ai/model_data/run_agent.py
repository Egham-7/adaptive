#!/usr/bin/env python3
"""
Model Enrichment Agent Runner

Simple entry point for the LangGraph model enrichment agent.
"""

import logging

from agent import run_enrichment
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    if not config.OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY not set")
        exit(1)

    run_enrichment()
