"""Adaptive Router Application Package.

This package contains the FastAPI application and all infrastructure-specific code
for the Adaptive Router service.
"""

from app.main import create_app

__version__ = "0.1.0"

__all__ = ["create_app"]
