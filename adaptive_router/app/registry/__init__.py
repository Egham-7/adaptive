"""Registry client package for Adaptive Router application.

This package provides HTTP client functionality for communicating with the
Adaptive model registry service.
"""

from app.registry.client import RegistryClient
from app.registry.models import ModelRegistry

__all__ = ["RegistryClient", "ModelRegistry"]
