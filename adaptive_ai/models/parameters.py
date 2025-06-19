from typing import cast

from pydantic import BaseModel, Field, field_validator

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

    @field_validator("temperature", "top_p", "presence_penalty", "frequency_penalty")
    @classmethod
    def _round_floats(cls, v: float) -> float:
        return round(v, 2)

    def _clamp_value(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp a value to the specified range"""
        return max(min_val, min(max_val, value))

    def adjust_parameters(
        self,
        task_type: str,
        prompt_scores: dict[str, list[float]],
    ) -> dict[str, float | int | str]:
        settings = get_settings()
        task_params = settings.get_task_parameters()

        if task_type not in task_params:
            raise ValueError(
                "Invalid task type. Choose from: " + ", ".join(task_params.keys())
            )

        tt = cast(TaskType, task_type)
        base: TaskTypeParametersType = task_params[tt]

        # Extract and validate prompt scores (clamp to [0, 1] range)
        creativity_scope = self._clamp_value(
            prompt_scores.get("creativity_scope", [0.5])[0], 0.0, 1.0
        )
        reasoning = self._clamp_value(
            prompt_scores.get("reasoning", [0.5])[0], 0.0, 1.0
        )
        contextual_knowledge = self._clamp_value(
            prompt_scores.get("contextual_knowledge", [0.5])[0], 0.0, 1.0
        )
        prompt_complexity_score = self._clamp_value(
            prompt_scores.get("prompt_complexity_score", [0.5])[0], 0.0, 1.0
        )
        domain_knowledge = self._clamp_value(
            prompt_scores.get("domain_knowledge", [0.5])[0], 0.0, 1.0
        )

        # Calculate adjusted values with proper range validation

        # Temperature: 0.0 to 2.0
        adjusted_temperature = base["temperature"] + (creativity_scope - 0.5) * 0.5
        self.temperature = self._clamp_value(adjusted_temperature, 0.0, 2.0)

        # Top_p: 0.0 to 1.0
        adjusted_top_p = base["top_p"] + (creativity_scope - 0.5) * 0.3
        self.top_p = self._clamp_value(adjusted_top_p, 0.0, 1.0)

        # Presence penalty: -2.0 to 2.0
        adjusted_presence_penalty = (
            base["presence_penalty"] + (domain_knowledge - 0.5) * 0.4
        )
        self.presence_penalty = self._clamp_value(adjusted_presence_penalty, -2.0, 2.0)

        # Frequency penalty: -2.0 to 2.0
        adjusted_frequency_penalty = base["frequency_penalty"] + (reasoning - 0.5) * 0.4
        self.frequency_penalty = self._clamp_value(
            adjusted_frequency_penalty, -2.0, 2.0
        )

        # Max tokens: 1 to 4096 (OpenAI's common limit, though some models support more)
        adjusted_max_tokens = base["max_completion_tokens"] + int(
            (contextual_knowledge - 0.5) * 500
        )
        self.max_tokens = int(self._clamp_value(adjusted_max_tokens, 1, 4096))

        # N: 1 to 10 (reasonable limit for number of completions)
        base_n = base["n"]
        adjustment = (prompt_complexity_score - 0.5) * 2
        adjusted_n = base_n + adjustment
        self.n = int(self._clamp_value(adjusted_n, 1, 10))

        return self.get_parameters()

    def get_parameters(self) -> dict[str, float | int | str]:
        """Return OpenAI-compatible parameters dictionary"""
        return {
            "model": self.model,
            "temperature": round(self.temperature, 2),
            "top_p": round(self.top_p, 2),
            "presence_penalty": round(self.presence_penalty, 2),
            "frequency_penalty": round(self.frequency_penalty, 2),
            "max_tokens": self.max_tokens,
            "n": self.n,
        }
