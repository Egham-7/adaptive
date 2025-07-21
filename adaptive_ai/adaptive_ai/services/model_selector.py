# mypy: disable-error-code=import
from typing import Any

from adaptive_ai.config import (
    ACTIVE_PROVIDERS,
    minion_domains,
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
        self.log(
            "model_selection_service_init",
            {
                "default_models": [
                    e.model_name for e in self._default_task_specific_model_entries
                ],
                "task_mappings_loaded": len(task_model_mappings_data),
                "minion_domains_loaded": len(minion_domains),
                "total_minion_domain_task_combinations": sum(
                    len(tasks) for tasks in minion_domains.values()
                ),
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

        # Standard protocol: Use task-based selection only (no domain routing)
        # Direct lookup: task_model_mappings_data[task_type] - fail fast if missing
        task_mapping = task_model_mappings_data.get(primary_task_type)
        if not task_mapping:
            raise ValueError(
                f"No task mapping found for task type: {primary_task_type}"
            )

        candidate_model_entries: list[TaskModelEntry] = task_mapping.model_entries

        self.log(
            "task_only_selection",
            {
                "task_type": primary_task_type.value,
                "models_count": len(candidate_model_entries),
                "top_model": (
                    candidate_model_entries[0].model_name
                    if candidate_model_entries
                    else None
                ),
                "domain_info": (
                    domain_classification.domain.value
                    if domain_classification
                    else "not_used"
                ),
            },
        )

        # Track metrics
        self.selection_metrics["total_selections"] += 1

        # Track task usage only (no domain tracking for standard protocol)
        # if domain_classification:  # Commented out - no domain tracking for standard

        task_key: str = primary_task_type.value
        self.selection_metrics["task_usage"][task_key] = (
            self.selection_metrics["task_usage"].get(task_key, 0) + 1
        )

        seen_model_identifiers: set[tuple[ProviderType, str]] = set()
        candidate_models: list[ModelCapability] = []

        for task_model_entry in candidate_model_entries:
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
                    "task_coverage_rate": (
                        len(self.selection_metrics["task_usage"])
                        / max(1, self.selection_metrics["total_selections"])
                    ),
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
        """Get the designated HuggingFace minion specialist for the domain-task combination."""
        primary_task_type: TaskType = (
            TaskType(classification_result.task_type_1[0])
            if classification_result.task_type_1
            else TaskType.OTHER
        )

        # REQUIRE domain classification - strict enforcement
        if not domain_classification:
            raise ValueError("Domain classification is required for minion selection")

        # Direct lookup - require exact domain/task match
        domain = domain_classification.domain
        if domain not in minion_domains:
            raise ValueError(f"Domain {domain.value} not supported in minion domains")
        if primary_task_type not in minion_domains[domain]:
            raise ValueError(
                f"Task {primary_task_type.value} not supported for domain {domain.value}"
            )
        minion_model = minion_domains[domain][primary_task_type]

        self.log(
            "minion_domain_task_selected",
            {
                "domain": domain.value,
                "task_type": primary_task_type.value,
                "confidence": domain_classification.confidence,
                "minion_model": minion_model,
            },
        )
        return minion_model

    def get_available_minions(self) -> list[str]:
        """Get all available minion models from the domain matrix."""
        minions: set[str] = set()
        for domain_tasks in minion_domains.values():
            minions.update(domain_tasks.values())
        return sorted(minions)

    def get_minion_alternatives(self, primary_minion: str) -> list[dict[str, str]]:
        """Get alternative minion models excluding the primary minion."""
        all_minions = self.get_available_minions()
        alternatives = [
            {"provider": "adaptive", "model": minion}
            for minion in all_minions
            if minion != primary_minion
        ]
        self.log(
            "minion_alternatives_selected",
            {
                "primary_minion": primary_minion,
                "alternatives_count": len(alternatives),
                "alternatives": alternatives[:3],  # Log first 3 for debugging
            },
        )
        return alternatives

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
                "task_coverage_rate": len(self.selection_metrics["task_usage"]) / total,
                "model_diversity_rate": len(self.selection_metrics["model_usage"])
                / total,
            },
            "usage_stats": {
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

        # Check task coverage
        task_coverage_rate: float = len(self.selection_metrics["task_usage"]) / total
        if task_coverage_rate < 0.8:
            recommendations.append(
                f"Task coverage rate is {task_coverage_rate:.1%}. "
                "Consider adding more diverse task-specific model mappings."
            )

        # Check for skewed task usage
        task_usage: dict[str, int] = self.selection_metrics["task_usage"]
        if task_usage:
            max_task_usage: int = max(task_usage.values())
            if max_task_usage > total * 0.6:
                top_task: str = max(task_usage, key=lambda k: task_usage[k])
                recommendations.append(
                    f"Task '{top_task}' accounts for {max_task_usage/total:.1%} of requests. "
                    "Consider optimizing models for this task type."
                )

        # Check for unused models
        model_usage: dict[str, int] = self.selection_metrics["model_usage"]
        if len(model_usage) < 5:
            recommendations.append(
                "Only a few models are being used. Consider reviewing model diversity."
            )

        return recommendations
