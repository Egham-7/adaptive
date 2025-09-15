"""Prompt Task Complexity Classifier Package.

Clean package containing the prompt task complexity classifier model implementation
for Modal deployment.

Modules:
    - task_complexity_model: Complete ML model implementation with GPU acceleration
    - config: Configuration management
    - models: Pydantic models for API requests and responses
"""

__version__ = "1.0.0"

# Export available components - use lazy imports to avoid dependency issues
__all__ = ["task_complexity_model", "config", "models"]
