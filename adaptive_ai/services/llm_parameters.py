from abc import ABC, abstractmethod
from typing import Union, Dict, Type, cast

from models.llms import domain_parameters, DomainParametersType


class LLMProviderParameters(ABC):
    @abstractmethod
    def adjust_parameters(self, domain: str, prompt_scores: dict) -> dict:
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        pass


class OpenAIParameters(LLMProviderParameters):
    def __init__(self, model: str):
        self.model = model
        self.temperature: float = 0.7
        self.top_p: float = 0.9
        self.presence_penalty: float = 0.0
        self.frequency_penalty: float = 0.0
        self.max_tokens: int = 1000
        self.n: int = 1

    def adjust_parameters(self, domain: str, prompt_scores: dict) -> dict:
        if domain not in domain_parameters:
            raise ValueError(
                "Invalid domain. Choose from: " + ", ".join(domain_parameters.keys())
            )

        base = cast(DomainParametersType, domain_parameters[domain])
        # Extract prompt scores
        creativity_scope = prompt_scores.get("creativity_scope", [0.5])[0]
        reasoning = prompt_scores.get("reasoning", [0.5])[0]
        contextual_knowledge = prompt_scores.get("contextual_knowledge", [0.5])[0]
        prompt_complexity_score = prompt_scores.get("prompt_complexity_score", [0.5])[0]
        domain_knowledge = prompt_scores.get("domain_knowledge", [0.5])[0]

        # Compute adjustments
        self.temperature = float(base["Temperature"] + (creativity_scope - 0.5) * 0.5)
        self.top_p = float(base["TopP"] + (creativity_scope - 0.5) * 0.3)
        self.presence_penalty = float(
            base["PresencePenalty"] + (domain_knowledge - 0.5) * 0.4
        )
        self.frequency_penalty = float(
            base["FrequencyPenalty"] + (reasoning - 0.5) * 0.4
        )
        self.max_tokens = base["MaxCompletionTokens"] + int(
            (contextual_knowledge - 0.5) * 500
        )

        # Explicitly calculate n as an integer
        base_n = base["N"]
        adjustment = (prompt_complexity_score - 0.5) * 2
        rounded_value = round(base_n + adjustment)
        self.n = max(1, rounded_value)

        # Round values for better output
        self._post_process_values()
        return self.get_parameters()

    def _post_process_values(self):
        self.temperature = round(self.temperature, 2)
        self.top_p = round(self.top_p, 2)
        self.presence_penalty = round(self.presence_penalty, 2)
        self.frequency_penalty = round(self.frequency_penalty, 2)
        self.max_tokens = int(self.max_tokens)

    def get_parameters(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": self.max_tokens,
            "n": self.n,
        }


class GroqParameters(LLMProviderParameters):
    def __init__(self, model: str):
        self.model = model
        self.temperature: float = 0.7
        self.top_p: float = 0.9
        self.presence_penalty: float = 0.0
        self.frequency_penalty: float = 0.0
        self.max_tokens: int = 1000
        self.n: int = 1

    def adjust_parameters(self, domain: str, prompt_scores: dict) -> dict:
        if domain not in domain_parameters:
            raise ValueError(
                "Invalid domain. Choose from: " + ", ".join(domain_parameters.keys())
            )

        base = cast(DomainParametersType, domain_parameters[domain])
        # Extract prompt scores
        creativity_scope = prompt_scores.get("creativity_scope", [0.5])[0]
        reasoning = prompt_scores.get("reasoning", [0.5])[0]
        contextual_knowledge = prompt_scores.get("contextual_knowledge", [0.5])[0]
        prompt_complexity_score = prompt_scores.get("prompt_complexity_score", [0.5])[0]
        domain_knowledge = prompt_scores.get("domain_knowledge", [0.5])[0]

        # Compute adjustments
        self.temperature = float(base["Temperature"] + (creativity_scope - 0.5) * 0.5)
        self.top_p = float(base["TopP"] + (creativity_scope - 0.5) * 0.3)
        self.presence_penalty = float(
            base["PresencePenalty"] + (domain_knowledge - 0.5) * 0.4
        )
        self.frequency_penalty = float(
            base["FrequencyPenalty"] + (reasoning - 0.5) * 0.4
        )
        self.max_tokens = base["MaxCompletionTokens"] + int(
            (contextual_knowledge - 0.5) * 500
        )

        # Explicitly calculate n as an integer
        base_n = base["N"]
        adjustment = (prompt_complexity_score - 0.5) * 2
        rounded_value = round(base_n + adjustment)
        self.n = max(1, rounded_value)

        # Round values for better output
        self._post_process_values()
        return self.get_parameters()

    def _post_process_values(self):
        self.temperature = round(self.temperature, 2)
        self.top_p = round(self.top_p, 2)
        self.presence_penalty = round(self.presence_penalty, 2)
        self.frequency_penalty = round(self.frequency_penalty, 2)
        self.max_tokens = int(self.max_tokens)

    def get_parameters(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": self.max_tokens,
            "n": self.n,
        }


class DeepSeekParameters(LLMProviderParameters):
    def __init__(self, model: str):
        self.model = model
        self.temperature: float = 0.7
        self.top_p: float = 0.9
        self.presence_penalty: float = 0.0
        self.frequency_penalty: float = 0.0
        self.max_tokens: int = 1000
        self.n: int = 1

    def adjust_parameters(self, domain: str, prompt_scores: dict) -> dict:
        if domain not in domain_parameters:
            raise ValueError(
                "Invalid domain. Choose from: " + ", ".join(domain_parameters.keys())
            )

        base = cast(DomainParametersType, domain_parameters[domain])
        # Extract prompt scores
        creativity_scope = prompt_scores.get("creativity_scope", [0.5])[0]
        reasoning = prompt_scores.get("reasoning", [0.5])[0]
        contextual_knowledge = prompt_scores.get("contextual_knowledge", [0.5])[0]
        prompt_complexity_score = prompt_scores.get("prompt_complexity_score", [0.5])[0]
        domain_knowledge = prompt_scores.get("domain_knowledge", [0.5])[0]

        # Compute adjustments
        self.temperature = float(base["Temperature"] + (creativity_scope - 0.5) * 0.5)
        self.top_p = float(base["TopP"] + (creativity_scope - 0.5) * 0.3)
        self.presence_penalty = float(
            base["PresencePenalty"] + (domain_knowledge - 0.5) * 0.4
        )
        self.frequency_penalty = float(
            base["FrequencyPenalty"] + (reasoning - 0.5) * 0.4
        )
        self.max_tokens = base["MaxCompletionTokens"] + int(
            (contextual_knowledge - 0.5) * 500
        )

        # Explicitly calculate n as an integer
        base_n = base["N"]
        adjustment = (prompt_complexity_score - 0.5) * 2
        rounded_value = round(base_n + adjustment)
        self.n = max(1, rounded_value)

        # Round values for better output
        self._post_process_values()
        return self.get_parameters()

    def _post_process_values(self):
        self.temperature = round(self.temperature, 2)
        self.top_p = round(self.top_p, 2)
        self.presence_penalty = round(self.presence_penalty, 2)
        self.frequency_penalty = round(self.frequency_penalty, 2)
        self.max_tokens = int(self.max_tokens)

    def get_parameters(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": self.max_tokens,
            "n": self.n,
        }


class LLMParametersFactory:
    """Factory class for creating LLM provider parameter objects."""

    @staticmethod
    def create(
        provider: str, model_name: str
    ) -> Union[OpenAIParameters, GroqParameters, DeepSeekParameters]:
        """
        Create and return an appropriate LLMProviderParameters instance based on provider.

        Args:
            provider: The LLM provider name (e.g., 'OpenAI', 'GROQ', 'DeepSeek')
            model_name: The name of the model to use

        Returns:
            An instance of a concrete LLMProviderParameters implementation

        Raises:
            ValueError: If the provider is not supported
        """
        provider_map: Dict[
            str, Type[Union[OpenAIParameters, GroqParameters, DeepSeekParameters]]
        ] = {
            "OpenAI": OpenAIParameters,
            "GROQ": GroqParameters,
            "DeepSeek": DeepSeekParameters,
        }

        if provider not in provider_map:
            raise ValueError(
                f"Provider '{provider}' not supported. Choose from: {
                    ', '.join(provider_map.keys())
                }"
            )

        # Explicitly cast the result to satisfy mypy
        return cast(
            Union[OpenAIParameters, GroqParameters, DeepSeekParameters],
            provider_map[provider](model_name),
        )
