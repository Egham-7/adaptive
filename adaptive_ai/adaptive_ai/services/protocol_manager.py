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

    def __init__(
        self,
        lit_logger: LitLoggerProtocol | None = None,
        device: (
            str | None
        ) = None,  # Accept but ignore device parameter for compatibility
    ) -> None:
        self.lit_logger: LitLoggerProtocol | None = lit_logger
        self.log(
            "protocol_manager_init",
            {"rule_based": True, "device_ignored": device},
        )

    def log(self, key: str, value: Any) -> None:
        if self.lit_logger:
            self.lit_logger.log(key, value)

    def _get_task_thresholds(self, task_type: str) -> dict[str, float]:
        """Get task-specific thresholds optimized for different task types."""
        # Task-specific thresholds based on task characteristics
        task_configs = {
            "Code Generation": {
                "complexity": 0.30,  # Lower - code tasks benefit from standard models
                "token_count": 2500,  # Lower - code context is important
                "few_shots": 3,  # Lower - examples matter for code
                "reasoning": 0.45,  # Lower - logic-heavy
            },
            "Summarization": {
                "complexity": 0.50,  # Higher - simpler task for minions
                "token_count": 4000,  # Higher - can handle longer text
                "few_shots": 5,  # Higher - pattern is straightforward
                "reasoning": 0.60,  # Higher - less reasoning needed
            },
            "Open QA": {
                "complexity": 0.35,  # Lower - knowledge-intensive
                "token_count": 2000,  # Lower - context matters
                "few_shots": 3,  # Lower - examples help
                "reasoning": 0.50,  # Medium - needs good reasoning
            },
            "Text Generation": {
                "complexity": 0.45,  # Medium-high - creative but structured
                "token_count": 3500,  # Medium-high - context less critical
                "few_shots": 4,  # Medium - some examples help
                "reasoning": 0.60,  # Higher - more creative than logical
            },
            "Classification": {
                "complexity": 0.55,  # Higher - good for minions
                "token_count": 5000,  # Higher - can handle lots of examples
                "few_shots": 6,  # Higher - pattern recognition
                "reasoning": 0.65,  # Higher - less reasoning needed
            },
        }

        # Default thresholds (current values)
        default_thresholds = {
            "complexity": 0.40,
            "token_count": 3000,
            "few_shots": 4,
            "reasoning": 0.55,
        }

        return task_configs.get(task_type, default_thresholds)

    def _calculate_composite_score(
        self,
        task_type: str,
        classification_result: ClassificationResult,
        token_count: int,
    ) -> float:
        """Calculate composite score using all available classification features."""
        # Extract all available features (taking first element since they're lists)
        complexity_score = classification_result.prompt_complexity_score[0]
        reasoning = classification_result.reasoning[0]
        creativity_scope = classification_result.creativity_scope[0]
        contextual_knowledge = classification_result.contextual_knowledge[0]
        domain_knowledge = classification_result.domain_knowledge[0]
        constraint_ct = classification_result.constraint_ct[0]
        number_of_few_shots = classification_result.number_of_few_shots[0]

        # Task-specific weights using ALL available features
        task_weights = {
            "Code Generation": {
                "complexity": 0.1,
                "reasoning": 0.25,
                "creativity": 0.05,
                "contextual": 0.15,
                "domain": 0.2,
                "constraints": 0.15,
                "few_shots": 0.05,
                "token_weight": 0.05,
            },
            "Open QA": {
                "complexity": 0.15,
                "reasoning": 0.2,
                "creativity": 0.1,
                "contextual": 0.2,
                "domain": 0.15,
                "constraints": 0.1,
                "few_shots": 0.05,
                "token_weight": 0.05,
            },
            "Summarization": {
                "complexity": 0.2,
                "reasoning": 0.15,
                "creativity": 0.1,
                "contextual": 0.25,
                "domain": 0.1,
                "constraints": 0.1,
                "few_shots": 0.05,
                "token_weight": 0.05,
            },
            "Text Generation": {
                "complexity": 0.25,
                "reasoning": 0.1,
                "creativity": 0.3,
                "contextual": 0.1,
                "domain": 0.05,
                "constraints": 0.1,
                "few_shots": 0.05,
                "token_weight": 0.05,
            },
            "Classification": {
                "complexity": 0.05,
                "reasoning": 0.2,
                "creativity": 0.05,
                "contextual": 0.15,
                "domain": 0.15,
                "constraints": 0.2,
                "few_shots": 0.15,
                "token_weight": 0.05,
            },
        }

        # Default weights
        default_weights = {
            "complexity": 0.15,
            "reasoning": 0.15,
            "creativity": 0.15,
            "contextual": 0.15,
            "domain": 0.1,
            "constraints": 0.15,
            "few_shots": 0.1,
            "token_weight": 0.05,
        }

        weights = task_weights.get(task_type, default_weights)

        # Normalize token count to 0-1 scale (assuming max reasonable is defined by MAX_TOKEN_COUNT)
        normalized_token_score = min(token_count / self.MAX_TOKEN_COUNT, 1.0)

        # Calculate composite score using all features (higher = more likely to use standard)
        composite_score = (
            weights["complexity"] * complexity_score
            + weights["reasoning"] * reasoning
            + weights["creativity"] * creativity_scope
            + weights["contextual"] * contextual_knowledge
            + weights["domain"] * domain_knowledge
            + weights["constraints"] * constraint_ct
            + weights["few_shots"] * number_of_few_shots
            + weights["token_weight"] * normalized_token_score
        )

        return float(min(composite_score, 1.0))  # Cap at 1.0 and ensure float type

    def _select_best_protocol(
        self,
        task_type: str,
        classification_result: ClassificationResult,
        token_count: int,
        available_protocols: list[str],
    ) -> str:
        """Select the best protocol from available options based on task requirements."""
        # Calculate composite score for each protocol
        protocol_scores = {}

        for protocol in available_protocols:
            score = self._calculate_protocol_score(
                protocol, task_type, classification_result, token_count
            )
            protocol_scores[protocol] = score

        # Return protocol with highest score
        return max(protocol_scores, key=lambda k: protocol_scores[k])

    def _calculate_protocol_score(
        self,
        protocol: str,
        task_type: str,
        classification_result: ClassificationResult,
        token_count: int,
    ) -> float:
        """Calculate how well a protocol fits the task requirements."""
        composite_score = self._calculate_composite_score(
            task_type, classification_result, token_count
        )

        # Protocol-specific scoring logic (expandable)
        protocol_multipliers = {
            "standard_llm": {
                "base_multiplier": 1.0,
                "complexity_bonus": 0.3,  # Bonus for complex tasks
                "token_penalty": (
                    -0.1 if token_count < 1000 else 0.0
                ),  # Penalty for short tasks
            },
            "minion": {
                "base_multiplier": 0.8,
                "complexity_penalty": -0.2,  # Penalty for complex tasks
                "token_bonus": (
                    0.2 if token_count < 1000 else 0.0
                ),  # Bonus for short tasks
            },
            # Easy to add new protocols:
            # "specialist": {
            #     "base_multiplier": 1.2,
            #     "domain_bonus": 0.4 if domain == "code" else 0.0,
            # },
        }

        if protocol not in protocol_multipliers:
            self.log(
                "unknown_protocol_error",
                {
                    "protocol": protocol,
                    "available_protocols": list(protocol_multipliers.keys()),
                    "task_type": task_type,
                },
            )
            raise ValueError(
                f"Unknown protocol '{protocol}'. Available protocols: {list(protocol_multipliers.keys())}"
            )

        multiplier_config = protocol_multipliers[protocol]
        score = composite_score * multiplier_config["base_multiplier"]

        # Apply bonuses/penalties
        complexity_score = classification_result.prompt_complexity_score[0]
        if "complexity_bonus" in multiplier_config:
            score += complexity_score * multiplier_config["complexity_bonus"]
        if "complexity_penalty" in multiplier_config:
            score += complexity_score * multiplier_config["complexity_penalty"]
        if "token_bonus" in multiplier_config:
            score += multiplier_config["token_bonus"]
        if "token_penalty" in multiplier_config:
            score += multiplier_config["token_penalty"]

        return max(0.0, min(1.0, score))  # Clamp between 0-1

    def _log_protocol_decision(
        self,
        task_type: str,
        protocol_choice: str,
        classification_result: ClassificationResult,
        token_count: int,
    ) -> None:
        """Log the protocol selection decision with all relevant factors."""
        task_thresholds = self._get_task_thresholds(task_type)
        composite_score = self._calculate_composite_score(
            task_type, classification_result, token_count
        )

        complexity_score = classification_result.prompt_complexity_score[0]
        reasoning = classification_result.reasoning[0]
        number_of_few_shots = classification_result.number_of_few_shots[0]

        self.log(
            "rule_based_protocol_selection",
            {
                "task_type": task_type,
                "protocol_choice": protocol_choice,
                "complexity_score": complexity_score,
                "token_count": token_count,
                "number_of_few_shots": number_of_few_shots,
                "reasoning": reasoning,
                "decision_factors": {
                    "high_complexity": complexity_score > task_thresholds["complexity"],
                    "long_input": token_count > task_thresholds["token_count"],
                    "many_few_shots": number_of_few_shots
                    > task_thresholds["few_shots"],
                    "high_reasoning": reasoning > task_thresholds["reasoning"],
                },
                "task_thresholds": task_thresholds,
                "composite_score": composite_score,
            },
        )

    def _create_protocol_response(
        self, protocol_choice: str, candidates_map: dict[str, list[ModelEntry]]
    ) -> OrchestratorResponse:
        """Create the appropriate protocol response based on decision."""
        parameters = OpenAIParameters(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
            n=1,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

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
            task_type, classification_result, token_count, available_protocols
        )

        # Log decision with full context
        self._log_protocol_decision(
            task_type, protocol_choice, classification_result, token_count
        )

        # Create and return response
        return self._create_protocol_response(protocol_choice, candidates_map)
