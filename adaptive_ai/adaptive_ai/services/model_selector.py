# mypy: disable-error-code=import
from typing import Any

from adaptive_ai.config.domain_matrix_generator import validate_matrix_coverage
from adaptive_ai.config.model_catalog import (
    ACTIVE_PROVIDERS,
    domain_fallback_preferences,
    domain_task_model_matrix,
    minion_domain_mappings,
    minion_task_model_mappings,
    provider_model_capabilities,
    task_model_mappings_data,
)
from adaptive_ai.models.llm_classification_models import (
    ClassificationResult,
    DomainClassificationResult,
)
from adaptive_ai.models.llm_core_models import (
    ModelCapability,
    ModelSelectionRequest,
    TaskModelEntry,
)
from adaptive_ai.models.llm_enums import ProviderType, TaskType


class LitLoggerProtocol:
    def log(self, key: str, value: Any) -> None: ...


class ModelSelectionService:
    def __init__(self, lit_logger: LitLoggerProtocol | None = None) -> None:
        self._all_model_capabilities_by_id: dict[
            tuple[ProviderType, str], ModelCapability
        ] = {
            (m_cap.provider, m_cap.model_name): m_cap
            for provider_list in provider_model_capabilities.values()
            for m_cap in provider_list
        }

        # Performance metrics
        self.selection_metrics: dict[str, Any] = {
            "total_selections": 0,
            "domain_matrix_hits": 0,
            "domain_fallback_hits": 0,
            "task_fallback_hits": 0,
            "default_fallback_hits": 0,
            "domain_usage": {},
            "task_usage": {},
            "model_usage": {},
            "performance_times": [],
        }

        # Updated default models to only include active providers
        self._default_task_specific_model_entries: list[TaskModelEntry] = [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
        ]
        self.lit_logger: LitLoggerProtocol | None = lit_logger
        # Validate matrix coverage on startup
        matrix_report = validate_matrix_coverage(domain_task_model_matrix)
        self.log(
            "model_selection_service_init",
            {
                "default_models": [
                    e.model_name for e in self._default_task_specific_model_entries
                ],
                "matrix_coverage": matrix_report["coverage_percentage"],
                "matrix_complete": matrix_report["is_complete"],
                "total_combinations": matrix_report["total_combinations"],
            },
        )

    def log(self, key: str, value: Any) -> None:
        if self.lit_logger:
            self.lit_logger.log(key, value)

    def select_candidate_models(
        self,
        request: ModelSelectionRequest,
        classification_result: ClassificationResult,
        prompt_token_count: int,
        domain_classification: DomainClassificationResult | None = None,
    ) -> list[ModelCapability]:
        self.log(
            "select_candidate_models_called",
            {
                "provider_constraint": request.provider_constraint,
                "prompt_token_count": prompt_token_count,
                "classification_result": classification_result.model_dump(
                    exclude_unset=True
                ),
            },
        )
        if request.provider_constraint:
            all_known_provider_types = {p.value for p in ProviderType}
            # Filter to only include active providers
            eligible_providers: set[ProviderType] = {
                ProviderType(p)
                for p in request.provider_constraint
                if p in all_known_provider_types and ProviderType(p) in ACTIVE_PROVIDERS
            }
            for p in request.provider_constraint:
                if p not in all_known_provider_types:
                    self.log("invalid_provider_constraint", p)
                elif ProviderType(p) not in ACTIVE_PROVIDERS:
                    self.log("inactive_provider_constraint", p)
        else:
            # Only use active providers even when no constraint is specified
            eligible_providers_else: set[ProviderType] = ACTIVE_PROVIDERS
            eligible_providers = eligible_providers_else

        primary_task_type: TaskType = (
            TaskType(classification_result.task_type_1[0])
            if classification_result.task_type_1
            else TaskType.OTHER
        )

        self.log("primary_task_type", primary_task_type.value)

        # NEW 2D MATRIX APPROACH: Check for specific domain-task combination first
        domain_task_specific_models: list[TaskModelEntry] = []
        domain_fallback_models: list[TaskModelEntry] = []

        if domain_classification:
            try:
                # 1. Try to get specific domain-task combination
                domain_task_specific_models = domain_task_model_matrix.get(
                    (domain_classification.domain, primary_task_type), []
                )

                # 2. Get domain fallback preferences
                domain_fallback_models = domain_fallback_preferences.get(
                    domain_classification.domain, []
                )

                # 3. Log missing combinations for future optimization
                if not domain_task_specific_models:
                    self.log(
                        "missing_domain_task_combination",
                        {
                            "domain": domain_classification.domain.value,
                            "task_type": primary_task_type.value,
                            "confidence": domain_classification.confidence,
                            "fallback_available": len(domain_fallback_models) > 0,
                        },
                    )

                self.log(
                    "2d_matrix_selection",
                    {
                        "domain": domain_classification.domain.value,
                        "task_type": primary_task_type.value,
                        "confidence": domain_classification.confidence,
                        "matrix_specific_count": len(domain_task_specific_models),
                        "domain_fallback_count": len(domain_fallback_models),
                        "has_specific_mapping": len(domain_task_specific_models) > 0,
                        "matrix_key": f"{domain_classification.domain.value}_{primary_task_type.value}",
                    },
                )

            except Exception as e:
                self.log(
                    "domain_matrix_error",
                    {
                        "error": str(e),
                        "domain": (
                            domain_classification.domain.value
                            if domain_classification
                            else None
                        ),
                        "task_type": primary_task_type.value,
                    },
                )
                # Reset to empty lists on error
                domain_task_specific_models = []
                domain_fallback_models = []

        # Get task-specific preferences
        task_mapping_data = task_model_mappings_data.get(primary_task_type)
        task_specific_model_entries: list[TaskModelEntry] = (
            task_mapping_data.model_entries
            if task_mapping_data
            else self._default_task_specific_model_entries
        )

        # COMBINE IN PRIORITY ORDER and track metrics:
        # 1. Domain-Task specific models (highest priority)
        # 2. Domain fallback models (if no specific combination)
        # 3. Task-specific models (traditional approach)
        # 4. Default models (fallback)
        combined_model_entries: list[TaskModelEntry] = (
            domain_task_specific_models
            + domain_fallback_models
            + task_specific_model_entries
        )

        # Track metrics
        self.selection_metrics["total_selections"] += 1
        if domain_task_specific_models:
            self.selection_metrics["domain_matrix_hits"] += 1
        elif domain_fallback_models:
            self.selection_metrics["domain_fallback_hits"] += 1
        elif task_specific_model_entries:
            self.selection_metrics["task_fallback_hits"] += 1
        else:
            self.selection_metrics["default_fallback_hits"] += 1

        # Track domain and task usage
        if domain_classification:
            domain_key: str = domain_classification.domain.value
            self.selection_metrics["domain_usage"][domain_key] = (
                self.selection_metrics["domain_usage"].get(domain_key, 0) + 1
            )

        task_key: str = primary_task_type.value
        self.selection_metrics["task_usage"][task_key] = (
            self.selection_metrics["task_usage"].get(task_key, 0) + 1
        )

        seen_model_identifiers: set[tuple[ProviderType, str]] = set()
        candidate_models: list[ModelCapability] = []

        for task_model_entry in combined_model_entries:
            model_identifier: tuple[ProviderType, str] = (
                task_model_entry.provider,
                task_model_entry.model_name,
            )

            if (
                task_model_entry.provider not in eligible_providers
                or model_identifier in seen_model_identifiers
            ):
                self.log(
                    "model_skipped",
                    {
                        "provider": str(task_model_entry.provider),
                        "model": task_model_entry.model_name,
                    },
                )
                continue

            found_model_cap: ModelCapability | None = (
                self._all_model_capabilities_by_id.get(model_identifier)
            )

            if (
                found_model_cap is not None
                and found_model_cap.max_context_tokens >= prompt_token_count
            ):
                candidate_models.append(found_model_cap)
                seen_model_identifiers.add(model_identifier)

                # Track model usage
                model_key: str = (
                    f"{found_model_cap.provider.value}:{found_model_cap.model_name}"
                )
                self.selection_metrics["model_usage"][model_key] = (
                    self.selection_metrics["model_usage"].get(model_key, 0) + 1
                )

                self.log(
                    "model_candidate_added",
                    {
                        "provider": str(found_model_cap.provider),
                        "model": found_model_cap.model_name,
                    },
                )

        self.log("candidate_models_count", len(candidate_models))

        # Log comprehensive metrics periodically
        if self.selection_metrics["total_selections"] % 10 == 0:
            self.log(
                "model_selection_metrics",
                {
                    "total_selections": self.selection_metrics["total_selections"],
                    "domain_matrix_efficiency": (
                        self.selection_metrics["domain_matrix_hits"]
                        / self.selection_metrics["total_selections"]
                    ),
                    "top_domains": sorted(
                        self.selection_metrics["domain_usage"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5],
                    "top_tasks": sorted(
                        self.selection_metrics["task_usage"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5],
                    "top_models": sorted(
                        self.selection_metrics["model_usage"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5],
                },
            )

        return candidate_models

    def get_designated_minion(
        self,
        classification_result: ClassificationResult,
        domain_classification: DomainClassificationResult | None = None,
    ) -> str:
        """Get the designated HuggingFace minion specialist for the task type."""
        primary_task_type: TaskType = (
            TaskType(classification_result.task_type_1[0])
            if classification_result.task_type_1
            else TaskType.OTHER
        )

        # Check for domain-specific minion preference first
        minion_model: str | None = None
        if domain_classification:
            minion_model = minion_domain_mappings.get(domain_classification.domain)
            if minion_model:
                self.log(
                    "domain_specific_minion_selected",
                    {
                        "domain": domain_classification.domain.value,
                        "confidence": domain_classification.confidence,
                        "minion_model": minion_model,
                    },
                )

        # Fallback to task-specific minion if no domain preference
        if not minion_model:
            minion_model = minion_task_model_mappings.get(
                primary_task_type,
                minion_task_model_mappings[TaskType.OTHER],  # Fallback to OTHER
            )
            self.log(
                "task_specific_minion_selected",
                {
                    "task_type": primary_task_type.value,
                    "minion_model": minion_model,
                },
            )

        return minion_model if minion_model is not None else ""

    def get_minion_alternatives(
        self,
        primary_minion: str,
    ) -> list[dict[str, str]]:
        """Generate fallback minion alternatives by using other capable minions."""
        alternatives: list[dict[str, str]] = []

        # Get all other minions from the mapping that could potentially handle the task
        for _task, model in minion_task_model_mappings.items():
            if model != primary_minion:  # Exclude the primary minion
                alternatives.append({"provider": "groq", "model": model})

        # Limit to top 3 alternatives to avoid overwhelming
        return alternatives[:3]

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics for model selection."""
        total: int = self.selection_metrics["total_selections"]

        if total == 0:
            return {
                "total_selections": 0,
                "efficiency_rates": {},
                "usage_stats": {},
                "recommendations": [],
            }

        return {
            "total_selections": total,
            "efficiency_rates": {
                "domain_matrix_hit_rate": self.selection_metrics["domain_matrix_hits"]
                / total,
                "domain_fallback_rate": self.selection_metrics["domain_fallback_hits"]
                / total,
                "task_fallback_rate": self.selection_metrics["task_fallback_hits"]
                / total,
                "default_fallback_rate": self.selection_metrics["default_fallback_hits"]
                / total,
            },
            "usage_stats": {
                "domain_distribution": self.selection_metrics["domain_usage"],
                "task_distribution": self.selection_metrics["task_usage"],
                "model_distribution": self.selection_metrics["model_usage"],
            },
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on usage patterns."""
        recommendations: list[str] = []
        total: int = self.selection_metrics["total_selections"]

        if total == 0:
            return recommendations

        # Check domain matrix efficiency
        domain_efficiency: float = self.selection_metrics["domain_matrix_hits"] / total
        if domain_efficiency < 0.7:
            recommendations.append(
                f"Domain matrix efficiency is {domain_efficiency:.1%}. "
                "Consider adding more domain-task specific combinations."
            )

        # Check for skewed domain usage
        domain_usage: dict[str, int] = self.selection_metrics["domain_usage"]
        if domain_usage:
            max_domain_usage: int = max(domain_usage.values())
            if max_domain_usage > total * 0.6:
                top_domain: str = max(domain_usage, key=lambda k: domain_usage[k])
                recommendations.append(
                    f"Domain '{top_domain}' accounts for {max_domain_usage/total:.1%} of requests. "
                    "Consider optimizing models for this domain."
                )

        # Check for unused models
        model_usage: dict[str, int] = self.selection_metrics["model_usage"]
        if len(model_usage) < 5:
            recommendations.append(
                "Only a few models are being used. Consider reviewing model diversity."
            )

        return recommendations
