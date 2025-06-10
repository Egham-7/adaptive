"""
Configuration loader for model capabilities, mappings, and parameters.
Uses the core settings system for configuration management.
"""

from typing import Dict, Any, cast
from .types import ModelCapability, TaskType, TaskModelMapping, TaskTypeParametersType
from core.config import get_settings


def get_model_capabilities() -> Dict[str, ModelCapability]:
    """Get all model capabilities from config"""
    settings = get_settings()
    capabilities = settings.get_model_capabilities()

    # Validate and cast each capability
    validated_capabilities: Dict[str, ModelCapability] = {}
    for model_name, capability in capabilities.items():
        validated_capabilities[model_name] = cast(
            ModelCapability,
            {
                "description": str(capability.get("description", "")),
                "provider": str(capability.get("provider", "Unknown")),
            },
        )

    return validated_capabilities


def get_task_model_mappings() -> Dict[TaskType, TaskModelMapping]:
    """Get task to model mappings from config"""
    settings = get_settings()
    mappings = settings.get_task_model_mappings()

    # Validate and cast mappings
    validated_mappings: Dict[TaskType, TaskModelMapping] = {}
    for task_type, mapping in mappings.items():
        if task_type in [
            "Open QA",
            "Closed QA",
            "Summarization",
            "Text Generation",
            "Code Generation",
            "Chatbot",
            "Classification",
            "Rewrite",
            "Brainstorming",
            "Extraction",
            "Other",
        ]:
            validated_mappings[cast(TaskType, task_type)] = cast(
                TaskModelMapping,
                {
                    "easy": {
                        "model": str(mapping["easy"]["model"]),
                        "complexity_threshold": float(
                            mapping["easy"]["complexity_threshold"]
                        ),
                    },
                    "medium": {
                        "model": str(mapping["medium"]["model"]),
                        "complexity_threshold": float(
                            mapping["medium"]["complexity_threshold"]
                        ),
                    },
                    "hard": {
                        "model": str(mapping["hard"]["model"]),
                        "complexity_threshold": float(
                            mapping["hard"]["complexity_threshold"]
                        ),
                    },
                },
            )

    return validated_mappings


def get_task_parameters() -> Dict[TaskType, TaskTypeParametersType]:
    """Get task parameters from config"""
    settings = get_settings()
    parameters = settings.get_task_parameters()

    # Validate and cast parameters
    validated_parameters: Dict[TaskType, TaskTypeParametersType] = {}
    for task_type, params in parameters.items():
        if task_type in [
            "Open QA",
            "Closed QA",
            "Summarization",
            "Text Generation",
            "Code Generation",
            "Chatbot",
            "Classification",
            "Rewrite",
            "Brainstorming",
            "Extraction",
            "Other",
        ]:
            validated_parameters[cast(TaskType, task_type)] = cast(
                TaskTypeParametersType,
                {
                    "Temperature": float(params.get("Temperature", 0.5)),
                    "TopP": float(params.get("TopP", 0.8)),
                    "PresencePenalty": float(params.get("PresencePenalty", 0.0)),
                    "FrequencyPenalty": float(params.get("FrequencyPenalty", 0.0)),
                    "MaxCompletionTokens": int(params.get("MaxCompletionTokens", 1000)),
                    "N": int(params.get("N", 1)),
                },
            )

    return validated_parameters


def get_models_by_provider(provider: str) -> Dict[str, ModelCapability]:
    """Get all models for a specific provider"""
    capabilities = get_model_capabilities()
    return {
        name: capability
        for name, capability in capabilities.items()
        if capability["provider"] == provider
    }


def get_all_providers() -> set[str]:
    """Get all available providers"""
    capabilities = get_model_capabilities()
    return {capability["provider"] for capability in capabilities.values()}


def get_task_difficulty_config(task_type: TaskType, difficulty: str) -> Dict[str, Any]:
    """Get configuration for a specific task type and difficulty"""
    mappings = get_task_model_mappings()

    if task_type not in mappings:
        raise ValueError(f"Unknown task type: {task_type}")

    mapping = mappings[task_type]
    if difficulty not in ["easy", "medium", "hard"]:
        raise ValueError(
            f"Unknown difficulty: {difficulty}. Must be one of: easy, medium, hard"
        )

    # Use bracket notation with cast for TypedDict access
    if difficulty == "easy":
        return cast(Dict[str, Any], mapping["easy"])
    elif difficulty == "medium":
        return cast(Dict[str, Any], mapping["medium"])
    else:  # difficulty == "hard"
        return cast(Dict[str, Any], mapping["hard"])


def get_all_task_types() -> list[TaskType]:
    """Get all available task types"""
    mappings = get_task_model_mappings()
    return list(mappings.keys())
