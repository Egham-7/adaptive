from typing import Any, Protocol

from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import (
    ModelEntry,
    ModelSelectionRequest,
)
from adaptive_ai.models.llm_enums import ProtocolType
from adaptive_ai.models.llm_orchestration_models import (
    Alternative,
    MinionInfo,
    OpenAIParameters,
    OrchestratorResponse,
    StandardLLMInfo,
)


class LitLoggerProtocol(Protocol):
    def log(self, key: str, value: Any) -> None: ...


class ProtocolManager:
    MAX_TOKEN_COUNT = 10000  # Maximum reasonable token count for normalization

    # Pre-computed threshold constants for faster comparisons
    DEFAULT_COMPLEXITY_THRESHOLD = 0.4
    STANDARD_PROTOCOL_TOKEN_THRESHOLD = 60000  # Use standard protocol for long prompts
    TOKEN_BUCKET_SIZE = 500  # Token bucket size for caching

    def __init__(
        self,
        lit_logger: LitLoggerProtocol | None = None,
        device: (
            str | None
        ) = None,  # Accept but ignore device parameter for compatibility
    ) -> None:
        self.lit_logger: LitLoggerProtocol | None = lit_logger

        # Cache for protocol decisions to avoid repeated computations
        self._protocol_decision_cache: dict[tuple[float, float, int], bool] = {}
        self._cache_max_size = 500

        self.log(
            "protocol_manager_init",
            {"rule_based": True, "device_ignored": device, "caching_enabled": True},
        )

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
        # Use NVIDIA's professionally trained complexity score and prompt length
        complexity_score = classification_result.prompt_complexity_score[0]

        # Get configurable thresholds or use pre-computed defaults
        complexity_threshold = self.DEFAULT_COMPLEXITY_THRESHOLD
        token_threshold = self.STANDARD_PROTOCOL_TOKEN_THRESHOLD

        if request and request.protocol_manager_config:
            if request.protocol_manager_config.complexity_threshold is not None:
                complexity_threshold = (
                    request.protocol_manager_config.complexity_threshold
                )
            if request.protocol_manager_config.token_threshold is not None:
                token_threshold = request.protocol_manager_config.token_threshold

        # Create cache key with token bucket for better cache hits
        token_bucket = (token_count // self.TOKEN_BUCKET_SIZE) * self.TOKEN_BUCKET_SIZE
        cache_key = (
            round(complexity_score, 2),
            round(complexity_threshold, 2),
            token_bucket,
        )

        # Check cache first
        if cache_key in self._protocol_decision_cache:
            return self._protocol_decision_cache[cache_key]

        # Decision based on complexity score OR token length
        decision = (
            complexity_score > complexity_threshold or token_count > token_threshold
        )

        # Cache the result if we have space
        if len(self._protocol_decision_cache) < self._cache_max_size:
            self._protocol_decision_cache[cache_key] = decision

        return decision

    def _select_best_protocol(
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

    def _get_tuned_parameters(
        self, classification_result: ClassificationResult, task_type: str
    ) -> OpenAIParameters:
        """Get OpenAI parameters tuned based on classification features (optimized)."""
        # Extract key features for parameter tuning (single access per feature)
        creativity = classification_result.creativity_scope[0]
        reasoning = classification_result.reasoning[0]
        complexity = classification_result.prompt_complexity_score[0]
        classification_result.domain_knowledge[0]
        classification_result.constraint_ct[0]

        # Use pre-computed task configurations for O(1) lookup
        task_configs = self._get_task_config_optimized(task_type)

        # Direct computation without intermediate variables
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

        # Pre-compute penalty values
        frequency_penalty = min(
            0.5, reasoning * 0.2 if task_type == "Code Generation" else 0.0
        )
        presence_penalty = min(
            0.6, creativity * 0.4 if task_type == "Brainstorming" else 0.0
        )

        # Optimized top_p calculation
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
        # Use dict.get with default for O(1) lookup instead of nested if-else
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

    def _log_protocol_decision(
        self,
        task_type: str,
        protocol_choice: str,
        classification_result: ClassificationResult,
        token_count: int,
        request: ModelSelectionRequest | None = None,
    ) -> None:
        """Log the protocol selection decision using NVIDIA's complexity score."""
        complexity_score = classification_result.prompt_complexity_score[0]

        # Extract configuration information if available
        cost_bias = None
        cost_bias_active = False
        complexity_threshold = self.DEFAULT_COMPLEXITY_THRESHOLD
        complexity_threshold_custom = False
        token_threshold = self.STANDARD_PROTOCOL_TOKEN_THRESHOLD
        token_threshold_custom = False

        if request and request.protocol_manager_config:
            if request.protocol_manager_config.cost_bias is not None:
                cost_bias = request.protocol_manager_config.cost_bias
                cost_bias_active = True
            if request.protocol_manager_config.complexity_threshold is not None:
                complexity_threshold = (
                    request.protocol_manager_config.complexity_threshold
                )
                complexity_threshold_custom = True
            if request.protocol_manager_config.token_threshold is not None:
                token_threshold = request.protocol_manager_config.token_threshold
                token_threshold_custom = True

        # Determine cost bias impact on model selection
        cost_bias_impact = None
        if (
            cost_bias_active
            and protocol_choice == "standard_llm"
            and cost_bias is not None
        ):
            if cost_bias <= 0.3:
                cost_bias_impact = "strongly_budget_focused"
            elif cost_bias <= 0.7:
                cost_bias_impact = "balanced_cost_performance"
            else:
                cost_bias_impact = "strongly_performance_focused"

        self.log(
            "nvidia_complexity_protocol_selection",
            {
                "task_type": task_type,
                "protocol_choice": protocol_choice,
                "nvidia_complexity_score": complexity_score,
                "token_count": token_count,
                "complexity_config": {
                    "complexity_threshold": complexity_threshold,
                    "complexity_threshold_custom": complexity_threshold_custom,
                    "complexity_threshold_exceeded": complexity_score
                    > complexity_threshold,
                },
                "token_config": {
                    "token_threshold": token_threshold,
                    "token_threshold_custom": token_threshold_custom,
                    "token_threshold_exceeded": token_count > token_threshold,
                },
                "cost_bias_info": {
                    "cost_bias": cost_bias,
                    "cost_bias_active": cost_bias_active,
                    "cost_bias_impact": cost_bias_impact,
                    "applies_to_protocol": protocol_choice == "standard_llm",
                },
                "decision_factors": {
                    "high_complexity": complexity_score > complexity_threshold,
                    "long_input": token_count > token_threshold,
                    "cost_optimization_enabled": cost_bias_active
                    and protocol_choice == "standard_llm",
                },
                "all_classification_features": {
                    "complexity_score": complexity_score,
                    "reasoning": classification_result.reasoning[0],
                    "creativity_scope": classification_result.creativity_scope[0],
                    "contextual_knowledge": classification_result.contextual_knowledge[
                        0
                    ],
                    "domain_knowledge": classification_result.domain_knowledge[0],
                    "constraint_ct": classification_result.constraint_ct[0],
                    "number_of_few_shots": classification_result.number_of_few_shots[0],
                },
            },
        )

    def _create_protocol_response(
        self,
        protocol_choice: str,
        candidates_map: dict[str, list[ModelEntry]],
        classification_result: ClassificationResult,
        task_type: str,
    ) -> OrchestratorResponse:
        """Create the appropriate protocol response with tuned parameters."""
        parameters = self._get_tuned_parameters(classification_result, task_type)

        candidates = candidates_map.get(protocol_choice)
        if not candidates:
            raise ValueError(f"No candidates available for protocol: {protocol_choice}")

        # Protocol handler mapping - easily extensible
        protocol_handlers = {
            "standard_llm": self._create_standard_response,
            "minion": self._create_minion_response,
            # Easy to add new protocols:
            # "specialist": self._create_specialist_response,
        }

        handler = protocol_handlers.get(protocol_choice)
        if not handler:
            raise ValueError(
                f"Unknown protocol '{protocol_choice}'. Available protocols: {list(protocol_handlers.keys())}"
            )

        return handler(candidates, parameters)

    def _create_standard_response(
        self, standard_candidates: list[ModelEntry], parameters: OpenAIParameters
    ) -> OrchestratorResponse:
        """Create standard LLM response."""
        first_standard = standard_candidates[0]
        primary_provider = first_standard.providers[0].value

        standard = StandardLLMInfo(
            provider=primary_provider,
            model=first_standard.model_name,
            parameters=parameters,
            alternatives=self._convert_model_entries_to_alternatives(
                standard_candidates[1:]
            ),
        )
        return OrchestratorResponse(
            protocol=ProtocolType.STANDARD_LLM, standard=standard
        )

    def _create_minion_response(
        self, minion_candidates: list[ModelEntry], parameters: OpenAIParameters
    ) -> OrchestratorResponse:
        """Create minion response."""
        first_minion = minion_candidates[0]
        primary_provider = first_minion.providers[0].value

        minion = MinionInfo(
            provider=primary_provider,
            model=first_minion.model_name,
            parameters=parameters,
            alternatives=self._convert_model_entries_to_alternatives(
                minion_candidates[1:]
            ),
        )
        return OrchestratorResponse(protocol=ProtocolType.MINION, minion=minion)

    def _convert_model_entries_to_alternatives(
        self, model_entries: list[ModelEntry]
    ) -> list[Alternative]:
        """Convert ModelEntry objects to Alternative objects with all providers."""
        alternatives = []
        for entry in model_entries:
            for provider in entry.providers:
                alternatives.append(
                    Alternative(provider=provider.value, model=entry.model_name)
                )
        return alternatives

    def select_protocol(
        self,
        standard_candidates: list[ModelEntry],
        minion_candidates: list[ModelEntry],
        classification_result: ClassificationResult,
        token_count: int = 0,
        request: ModelSelectionRequest | None = None,
    ) -> OrchestratorResponse:
        # Extract key decision factors
        task_type = (
            classification_result.task_type_1[0]
            if classification_result.task_type_1
            else "Other"
        )

        # Determine available protocols based on candidates
        available_protocols = []
        candidates_map = {}

        if standard_candidates:
            available_protocols.append("standard_llm")
            candidates_map["standard_llm"] = standard_candidates

        if minion_candidates:
            available_protocols.append("minion")
            candidates_map["minion"] = minion_candidates

        if not available_protocols:
            raise ValueError("No candidates available for any protocol")

        # Select best protocol
        protocol_choice = self._select_best_protocol(
            classification_result, token_count, available_protocols, request
        )

        # Log decision with full context
        self._log_protocol_decision(
            task_type, protocol_choice, classification_result, token_count, request
        )

        # Create and return response
        return self._create_protocol_response(
            protocol_choice, candidates_map, classification_result, task_type
        )
