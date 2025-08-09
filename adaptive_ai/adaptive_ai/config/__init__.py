"""
Configuration module - Main interface aggregating all model configurations.

This module serves as the primary entry point for all model-related configurations,
importing from organized submodules while maintaining backward compatibility.
"""

# Import from organized submodules
from .domain_mappings import minion_domains
from .providers import provider_model_capabilities, provider_model_names
from .task_mappings import task_model_mappings_data

# Export everything
__all__ = [
    "minion_domains",
    "provider_model_capabilities",  # Kept for backward compatibility
    "provider_model_names",  # New simplified structure
    "task_model_mappings_data",
]
