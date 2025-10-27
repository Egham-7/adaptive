"""FastAPI server for adaptive_router.

This module provides HTTP API endpoints for model selection.
"""

from adaptive_router.api.app import app, create_app

__all__ = ["app", "create_app"]
