"""
Model catalog - Main interface aggregating all model configurations.

This module serves as the primary entry point for all model-related configurations,
importing from organized submodules while maintaining backward compatibility.
"""

# Import from organized submodules
from .domain_mappings import minion_domains
from .providers import ACTIVE_PROVIDERS, provider_model_capabilities
from .task_mappings import task_model_mappings_data

# Re-export everything for backward compatibility
__all__ = [
    "ACTIVE_PROVIDERS",
    "minion_domains",
    "provider_model_capabilities",
    "task_model_mappings_data",
]
