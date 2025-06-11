from typing import Union, cast

from pydantic import BaseModel, Field, validator

from adaptive_ai.core.config import get_settings
from adaptive_ai.models.types import TaskType, TaskTypeParametersType


class OpenAIParameters(BaseModel):
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
    def _round_floats(cls, self, v: float) -> float:
        return round(v, 2)

    def adjust_parameters(
        self,
        task_type: str,
        prompt_scores: dict[str, list[float]],
    ) -> dict[str, Union[float, int]]:
        settings = get_settings()
        task_params = settings.get_task_parameters()

        if task_type not in task_params:
            raise ValueError(
                "Invalid task type. Choose from: " + ", ".join(task_params.keys())
            )

        tt = cast(TaskType, task_type)
        base: TaskTypeParametersType = task_params[tt]

        creativity_scope = prompt_scores.get("creativity_scope", [0.5])[0]
        reasoning = prompt_scores.get("reasoning", [0.5])[0]
        contextual_knowledge = prompt_scores.get("contextual_knowledge", [0.5])[0]
        prompt_complexity_score = prompt_scores.get("prompt_complexity_score", [0.5])[0]
        domain_knowledge = prompt_scores.get("domain_knowledge", [0.5])[0]

        self.temperature = base["temperature"] + (creativity_scope - 0.5) * 0.5
        self.top_p = base["top_p"] + (creativity_scope - 0.5) * 0.3
        self.presence_penalty = (
            base["presence_penalty"] + (domain_knowledge - 0.5) * 0.4
        )
        self.frequency_penalty = base["frequency_penalty"] + (reasoning - 0.5) * 0.4
        self.max_tokens = base["max_completion_tokens"] + int(
            (contextual_knowledge - 0.5) * 500
        )

        base_n = base["n"]
        adjustment = (prompt_complexity_score - 0.5) * 2
        self.n = max(1, round(base_n + adjustment))

        return self.get_parameters()

    def get_parameters(self) -> dict[str, Union[float, int]]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": self.max_tokens,
            "n": self.n,
        }


class LLMParameterService:
    """Always uses OpenAIParameters under the hood."""

    def __init__(self) -> None:
        self._provider = OpenAIParameters

    def get_parameters(
        self,
        model: str,
        task_type: str,
        prompt_scores: dict[str, list[float]],
    ) -> dict[str, Union[float, int]]:
        params = self._provider(model=model)
        return params.adjust_parameters(task_type, prompt_scores)
