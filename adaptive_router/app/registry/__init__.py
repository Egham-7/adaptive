"""Registry client package for Adaptive Router application.

This package provides HTTP client functionality for communicating with the
Adaptive model registry service.
"""

from app.registry.client import RegistryClient
from app.registry.registry import ModelRegistry

__all__ = ["RegistryClient", "ModelRegistry"]
