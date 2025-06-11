from abc import ABC, abstractmethod
from typing import Union, Dict, cast
from pydantic import BaseModel, Field, validator

from adaptive_ai.core.config import get_settings
from adaptive_ai.models.types import TaskTypeParametersType, TaskType


class LLMProviderParameters(ABC):
    @abstractmethod
    def adjust_parameters(
        self, task_type: str, prompt_scores: Dict[str, list]
    ) -> Dict[str, Union[float, int]]:
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Union[float, int]]:
        pass


class OpenAIParameters(BaseModel, LLMProviderParameters):
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=4096)
    n: int = Field(default=1, ge=1, le=10)

    class Config:
        validate_assignment = True

    @validator("temperature", "top_p", "presence_penalty", "frequency_penalty")
    def round_float_values(cls, v):
        return round(v, 2)

    def adjust_parameters(
        self, task_type: str, prompt_scores: Dict[str, list]
    ) -> Dict[str, Union[float, int]]:
        settings = get_settings()
        task_type_parameters = settings.get_task_parameters()

        if task_type not in task_type_parameters:
            raise ValueError(
                "Invalid task type. Choose from: "
                + ", ".join(task_type_parameters.keys())
            )

        # Cast task_type to TaskType literal for proper indexing
        task_type_literal: TaskType = cast(TaskType, task_type)
        base: TaskTypeParametersType = task_type_parameters[task_type_literal]

        # Extract prompt scores
        creativity_scope: float = prompt_scores.get("creativity_scope", [0.5])[0]
        reasoning: float = prompt_scores.get("reasoning", [0.5])[0]
        contextual_knowledge: float = prompt_scores.get("contextual_knowledge", [0.5])[
            0
        ]
        prompt_complexity_score: float = prompt_scores.get(
            "prompt_complexity_score", [0.5]
        )[0]
        domain_knowledge: float = prompt_scores.get("domain_knowledge", [0.5])[0]

        # Compute adjustments
        self.temperature = float(base["temperature"] + (creativity_scope - 0.5) * 0.5)
        self.top_p = float(base["top_p"] + (creativity_scope - 0.5) * 0.3)
        self.presence_penalty = float(
            base["presence_penalty"] + (domain_knowledge - 0.5) * 0.4
        )
        self.frequency_penalty = float(
            base["frequency_penalty"] + (reasoning - 0.5) * 0.4
        )
        self.max_tokens = base["max_completion_tokens"] + int(
            (contextual_knowledge - 0.5) * 500
        )

        # Explicitly calculate n as an integer
        base_n: int = base["n"]
        adjustment: float = (prompt_complexity_score - 0.5) * 2
        rounded_value: int = round(base_n + adjustment)
        self.n = max(1, rounded_value)

        return self.get_parameters()

    def get_parameters(self) -> Dict[str, Union[float, int]]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": self.max_tokens,
            "n": self.n,
        }


class LLMParameterService:
    """Service for managing LLM parameters across different providers."""

    def __init__(self):
        self.providers = {
            "openai": OpenAIParameters
        }

    def get_parameters(self, provider: str, model: str, task_type: str, prompt_scores: Dict[str, list]) -> Dict[str, Union[float, int]]:
        """Get adjusted parameters for a specific provider and model."""
        if provider.lower() not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")

        parameter_class = self.providers[provider.lower()]
        parameters = parameter_class(model=model)
        return parameters.adjust_parameters(task_type, prompt_scores)

    def add_provider(self, name: str, parameter_class):
        """Add a new provider parameter class."""
        self.providers[name.lower()] = parameter_class
