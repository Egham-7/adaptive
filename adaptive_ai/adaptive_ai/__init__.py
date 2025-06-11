"""
Adaptive AI - Intelligent LLM Infrastructure with Smart Model Selection

This package provides intelligent model selection and routing capabilities
for Large Language Models (LLMs) across multiple providers.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "you@example.com"

from .main import create_app, AdaptiveModelSelectionAPI

__all__ = ["create_app", "AdaptiveModelSelectionAPI", "__version__"]
