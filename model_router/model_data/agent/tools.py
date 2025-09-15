"""
Search and Analysis Tools

LangChain tools for web search and model analysis.
"""

import logging
import os
from typing import Any

from langchain_community.tools import DuckDuckGoSearchRun  # type: ignore
from langchain_community.utilities import GoogleSerperAPIWrapper  # type: ignore

logger = logging.getLogger(__name__)


def search_model_documentation(provider: str, model_name: str) -> str:
    """Search for official model documentation and specifications."""
    try:
        # Try Serper first if API key available
        if os.getenv("SERPER_API_KEY"):
            search: GoogleSerperAPIWrapper | DuckDuckGoSearchRun = (
                GoogleSerperAPIWrapper()
            )
        else:
            search = DuckDuckGoSearchRun()

        queries = [
            f"{provider} {model_name} official documentation specifications",
            f"{model_name} context window tokens maximum capacity",
            f"{provider} {model_name} API capabilities features",
        ]

        results = []
        for query in queries:
            try:
                if hasattr(search, "invoke"):
                    result = search.invoke(query)
                else:
                    result = search.run(query)
                results.append(f"Query: {query}\\nResult: {result}\\n---")
            except Exception as e:
                logger.warning(f"Search failed for '{query}': {e}")

        return "\\n".join(results) if results else "No search results found."

    except Exception as e:
        return f"Search error: {e!s}"


def extract_technical_specs(search_results: str, model_name: str) -> dict[str, Any]:
    """Return empty dict - AI will handle all technical specification extraction."""
    # No static pattern matching - let AI analyze everything
    return {}


def classify_model_capabilities(
    model_name: str, search_results: str, provider: str
) -> dict[str, Any]:
    """Return minimal defaults - AI will handle all classification intelligently."""
    # No static pattern matching - let AI make all decisions
    return {"confidence_score": 0.5}  # Base confidence, will be updated by AI
