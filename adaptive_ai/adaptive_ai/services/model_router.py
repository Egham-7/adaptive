from threading import RLock
from typing import Any, Protocol

import cachetools

from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import (
    ModelSelectionRequest,
)
from adaptive_ai.models.llm_orchestration_models import (
    OpenAIParameters,
)


class LitLoggerProtocol(Protocol):
    def log(self, key: str, value: Any) -> None: ...


class ModelRouter:
    MAX_TOKEN_COUNT = 10000  # Maximum reasonable token count for normalization

    # Pre-computed threshold constants for faster comparisons
    DEFAULT_COMPLEXITY_THRESHOLD = 0.4
    STANDARD_PROTOCOL_TOKEN_THRESHOLD = 60000  # Use standard protocol for long prompts

    def __init__(
        self,
        lit_logger: LitLoggerProtocol | None = None,
    ) -> None:
        self.lit_logger: LitLoggerProtocol | None = lit_logger

        # Cache for protocol decisions to avoid repeated computations
        self._protocol_decision_cache: cachetools.LRUCache[tuple[float, float, int], bool] = (  # type: ignore[type-arg]
            cachetools.LRUCache(maxsize=500)
        )
        # Thread safety lock for cache access
        self._cache_lock = RLock()

        self.log(
            "protocol_manager_init",
            {"rule_based": True, "caching_enabled": True},
        )

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                "protocol_decision_cache": {
                    "size": self._protocol_decision_cache.currsize,
                    "max_size": self._protocol_decision_cache.maxsize,
                }
            }

    def log(self, key: str, value: Any) -> None:
        if self.lit_logger:
            self.lit_logger.log(key, value)

    def _should_use_standard_protocol(
        self,
        classification_result: ClassificationResult,
        token_count: int,
        request: ModelSelectionRequest | None = None,
    ) -> bool:
        """Determine if standard protocol should be used based on complexity score and token length."""
        # If user explicitly provided models, always use standard protocol
        if request and request.model_router and request.model_router.models:
            return True

        # Use NVIDIA's professionally trained complexity score and prompt length
        complexity_score = classification_result.prompt_complexity_score[0]

        # Get configurable thresholds or use pre-computed defaults
        complexity_threshold = self.DEFAULT_COMPLEXITY_THRESHOLD
        token_threshold = self.STANDARD_PROTOCOL_TOKEN_THRESHOLD

        if request and request.model_router:
            if request.model_router.complexity_threshold is not None:
                complexity_threshold = request.model_router.complexity_threshold
            if request.model_router.token_threshold is not None:
                token_threshold = request.model_router.token_threshold

        # Create cache key based on actual decision factors
        cache_key = (complexity_score, complexity_threshold, token_count)

        # Check cache first (thread-safe)
        with self._cache_lock:
            cached_result = self._protocol_decision_cache.get(cache_key)
            if cached_result is not None:
                return bool(cached_result)

        # Decision based on complexity score OR token length
        decision = (
            complexity_score > complexity_threshold or token_count > token_threshold
        )

        # Cache the result (LRU cache handles eviction automatically, thread-safe)
        with self._cache_lock:
            self._protocol_decision_cache[cache_key] = decision

        return decision

    def select_best_protocol(
        self,
        classification_result: ClassificationResult,
        token_count: int,
        available_protocols: list[str],
        request: ModelSelectionRequest | None = None,
    ) -> str:
        """Select the best protocol based on NVIDIA's complexity score."""
        should_use_standard = self._should_use_standard_protocol(
            classification_result, token_count, request
        )

        # Prefer standard_llm if complexity/tokens are high and available
        if should_use_standard and "standard_llm" in available_protocols:
            return "standard_llm"
        elif "minion" in available_protocols:
            return "minion"
        else:
            # Fallback to first available protocol
            return available_protocols[0]

    def get_tuned_parameters(
        self, classification_result: ClassificationResult, task_type: str
    ) -> OpenAIParameters:
        """Get OpenAI parameters tuned based on classification features.

        Public interface for parameter tuning functionality.

        Args:
            classification_result: The classification result containing complexity scores
            task_type: The task type string for parameter customization

        Returns:
            OpenAIParameters: Tuned parameters optimized for the given task and complexity
        """
        return self._get_tuned_parameters(classification_result, task_type)

    def _get_tuned_parameters(
        self, classification_result: ClassificationResult, task_type: str
    ) -> OpenAIParameters:
        """Get OpenAI parameters tuned based on classification features (optimized)."""
        creativity = classification_result.creativity_scope[0]
        reasoning = classification_result.reasoning[0]
        complexity = classification_result.prompt_complexity_score[0]

        task_configs = self._get_task_config_optimized(task_type)

        temperature = max(
            0.1,
            min(
                1.0,
                task_configs["base_temp"]
                + task_configs["temp_factor"]
                * (
                    complexity
                    if task_type in {"Code Generation", "Classification"}
                    else creativity
                ),
            ),
        )

        max_tokens = min(
            2500,
            task_configs["base_tokens"]
            + int(
                task_configs["token_factor"]
                * (
                    complexity
                    if task_type in {"Code Generation", "Classification"}
                    else creativity
                )
            ),
        )

        frequency_penalty = min(
            0.5, reasoning * 0.2 if task_type == "Code Generation" else 0.0
        )
        presence_penalty = min(
            0.6, creativity * 0.4 if task_type == "Brainstorming" else 0.0
        )

        if task_type == "Text Generation":
            top_p = min(0.95, 0.85 + creativity * 0.1)
        elif task_type == "Brainstorming":
            top_p = min(0.98, 0.9 + creativity * 0.08)
        else:
            top_p = 0.9

        self.log(
            "parameter_tuning",
            {
                "task_type": task_type,
                "tuned_parameters": {
                    "temperature": round(temperature, 2),
                    "top_p": round(top_p, 2),
                    "max_tokens": max_tokens,
                    "frequency_penalty": round(frequency_penalty, 2),
                    "presence_penalty": round(presence_penalty, 2),
                },
            },
        )

        return OpenAIParameters(
            temperature=round(temperature, 2),
            top_p=round(top_p, 2),
            max_tokens=int(max_tokens),
            n=1,
            stop=None,
            frequency_penalty=round(frequency_penalty, 2),
            presence_penalty=round(presence_penalty, 2),
        )

    def _get_task_config_optimized(self, task_type: str) -> dict[str, float]:
        """Get task configuration with optimized lookup."""
        configs = {
            "Code Generation": {
                "base_temp": 0.5,
                "temp_factor": -0.3,
                "base_tokens": 1200,
                "token_factor": 800,
            },
            "Text Generation": {
                "base_temp": 0.6,
                "temp_factor": 0.4,
                "base_tokens": 1000,
                "token_factor": 1000,
            },
            "Classification": {
                "base_temp": 0.3,
                "temp_factor": -0.2,
                "base_tokens": 200,
                "token_factor": 300,
            },
            "Brainstorming": {
                "base_temp": 0.8,
                "temp_factor": 0.2,
                "base_tokens": 1200,
                "token_factor": 800,
            },
        }
        return configs.get(
            task_type,
            {
                "base_temp": 0.7,
                "temp_factor": 0.0,
                "base_tokens": 1000,
                "token_factor": 0,
            },
        )
