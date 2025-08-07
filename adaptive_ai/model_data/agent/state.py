"""
LangGraph State Models

Defines the state structure for the model enrichment workflow.
"""

from typing import Any, TypedDict


class ModelExtractionState(TypedDict):
    """State definition for the LangGraph workflow."""

    provider: str
    model_name: str
    model_data: dict[str, Any]
    search_results: list[dict[str, Any]]
    extracted_info: dict[str, Any] | None
    confidence_score: float
    error_message: str | None
    retry_count: int
