"""
AI Model Metadata Enrichment Tools

A sophisticated toolkit for automatically enriching AI model metadata using
multiple research sources including HuggingFace, provider APIs, and web documentation.
"""

from .base_research_tool import BaseResearchTool, ModelMetadata
from .huggingface_research import HuggingFaceResearchTool
from .web_research import WebResearchTool
from .provider_research import ProviderResearchTool
from .quality_validator import QualityValidator

__version__ = "1.0.0"

__all__ = [
    "BaseResearchTool",
    "ModelMetadata", 
    "HuggingFaceResearchTool",
    "WebResearchTool",
    "ProviderResearchTool",
    "QualityValidator",
]