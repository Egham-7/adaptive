# mypy: disable-error-code=import
from typing import Any

from adaptive_ai.config import (
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
    ModelEntry,
    ModelSelectionRequest,
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

        # Cache for eligible providers per model and token count
        self._eligible_providers_cache: dict[tuple[str, int], list[ProviderType]] = {}

        # Pre-computed mapping of models to their available providers
        self._model_to_providers: dict[str, list[ProviderType]] = {}
        self._build_model_provider_cache()

        # Performance metrics
        self.selection_metrics: dict[str, Any] = {
            "total_selections": 0,
            "domain_usage": {},
            "task_usage": {},
            "model_usage": {},
            "performance_times": [],
        }

        self.lit_logger: LitLoggerProtocol | None = lit_logger
        self.log(
            "model_selection_service_init",
            {
                "task_mappings_loaded": len(task_model_mappings_data),
                "minion_domains_loaded": len(minion_domains),
                "total_minion_domains": len(minion_domains),
            },
        )

    def log(self, key: str, value: Any) -> None:
        if self.lit_logger:
            self.lit_logger.log(key, value)

    def _build_model_provider_cache(self) -> None:
        """Pre-compute which providers are available for each model."""
        for (provider, model_name), _ in self._all_model_capabilities_by_id.items():
            if model_name not in self._model_to_providers:
                self._model_to_providers[model_name] = []
            if provider not in self._model_to_providers[model_name]:
                self._model_to_providers[model_name].append(provider)

    def _get_eligible_providers_for_model(
        self,
        model_entry: ModelEntry,
        eligible_providers: set[ProviderType],
        prompt_token_count: int,
    ) -> list[ProviderType]:
        """Get eligible providers for a model that meet capability requirements."""
        # Create cache key - use rounded token count to improve cache hit rate
        cache_key = (model_entry.model_name, (prompt_token_count // 1000) * 1000)

        # Check cache first
        if cache_key in self._eligible_providers_cache:
            cached_providers = self._eligible_providers_cache[cache_key]
            # Filter cached providers by current eligibility
            return [p for p in cached_providers if p in eligible_providers]

        # Cache miss - compute and cache result
        eligible_providers_for_model = []

        # Use pre-computed available providers to avoid checking non-existent combinations
        available_providers = self._model_to_providers.get(model_entry.model_name, [])

        for provider in model_entry.providers:
            if provider in available_providers and provider in eligible_providers:
                model_identifier = (provider, model_entry.model_name)
                model_cap = self._all_model_capabilities_by_id[model_identifier]
                if model_cap.max_context_tokens >= prompt_token_count:
                    eligible_providers_for_model.append(provider)

        # Cache the result (regardless of current eligible_providers filter)
        all_token_eligible = []
        for provider in available_providers:
            model_identifier = (provider, model_entry.model_name)
            model_cap = self._all_model_capabilities_by_id[model_identifier]
            if model_cap.max_context_tokens >= prompt_token_count:
                all_token_eligible.append(provider)

        self._eligible_providers_cache[cache_key] = all_token_eligible

        return eligible_providers_for_model

    def get_cache_stats(self) -> dict[str, Any]:
        """Get caching performance statistics."""
        return {
            "eligible_providers_cache_size": len(self._eligible_providers_cache),
            "model_to_providers_cache_size": len(self._model_to_providers),
            "total_models_tracked": len(self._model_to_providers),
            "average_providers_per_model": (
                sum(len(providers) for providers in self._model_to_providers.values())
                / len(self._model_to_providers)
                if self._model_to_providers
                else 0
            ),
        }

    def clear_cache(self) -> None:
        """Clear the eligible providers cache (useful for testing or memory management)."""
        self._eligible_providers_cache.clear()
        self.log("cache_cleared", {"cache_type": "eligible_providers"})

    def select_candidate_models(
        self,
        request: ModelSelectionRequest,
        classification_result: ClassificationResult,
        prompt_token_count: int,
        domain_classification: DomainClassificationResult | None = None,
    ) -> list[ModelEntry]:
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
            # Filter to only include known providers with capabilities
            available_providers = set(provider_model_capabilities.keys())
            eligible_providers: set[ProviderType] = {
                ProviderType(p)
                for p in request.provider_constraint
                if p in all_known_provider_types
                and ProviderType(p) in available_providers
            }
            for p in request.provider_constraint:
                if p not in all_known_provider_types:
                    self.log("invalid_provider_constraint", p)
                elif ProviderType(p) not in available_providers:
                    self.log("unavailable_provider_constraint", p)
        else:
            # Use all providers with defined capabilities
            eligible_providers = set(provider_model_capabilities.keys())

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

        candidate_model_entries: list[ModelEntry] = task_mapping.model_entries

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

        seen_model_names: set[str] = set()
        candidate_models: list[ModelEntry] = []

        for task_model_entry in candidate_model_entries:
            if task_model_entry.model_name in seen_model_names:
                continue

            # Filter providers to only eligible ones that have capabilities defined
            eligible_providers_for_model = self._get_eligible_providers_for_model(
                task_model_entry, eligible_providers, prompt_token_count
            )

            if eligible_providers_for_model:
                # Create ModelEntry with only eligible providers
                candidate_entry = ModelEntry(
                    providers=eligible_providers_for_model,
                    model_name=task_model_entry.model_name,
                )
                candidate_models.append(candidate_entry)
                seen_model_names.add(task_model_entry.model_name)

                # Track model usage (use first eligible provider for tracking)
                first_provider = eligible_providers_for_model[0]
                model_key = f"{first_provider.value}:{task_model_entry.model_name}"
                self.selection_metrics["model_usage"][model_key] = (
                    self.selection_metrics["model_usage"].get(model_key, 0) + 1
                )

                self.log(
                    "model_candidate_added",
                    {
                        "model": task_model_entry.model_name,
                        "eligible_providers": [
                            p.value for p in eligible_providers_for_model
                        ],
                    },
                )

        self.log("candidate_models_count", len(candidate_models))

        # Apply cost-based ranking if cost_bias is provided
        if request.cost_bias is not None and candidate_models:
            from adaptive_ai.utils.cost_utils import rank_models_by_cost_performance
            
            original_count = len(candidate_models)
            original_top_model = candidate_models[0].model_name if candidate_models else None
            
            # Apply cost-performance ranking
            candidate_models = rank_models_by_cost_performance(
                model_entries=candidate_models,
                cost_bias=request.cost_bias,
                model_capabilities=self._all_model_capabilities_by_id,
                estimated_tokens=prompt_token_count
            )
            
            new_top_model = candidate_models[0].model_name if candidate_models else None
            
            self.log("cost_bias_routing", {
                "cost_bias": request.cost_bias,
                "original_model_count": original_count,
                "original_top_model": original_top_model,
                "cost_optimized_top_model": new_top_model,
                "reranked_models": [entry.model_name for entry in candidate_models[:5]],
                "ranking_changed": original_top_model != new_top_model
            })

        # Log comprehensive metrics periodically
        if self.selection_metrics["total_selections"] % 10 == 0:
            cache_stats = self.get_cache_stats()
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
                    "cache_performance": cache_stats,
                },
            )

        return candidate_models

    def get_minion_candidates(
        self,
        classification_result: ClassificationResult,
        domain_classification: DomainClassificationResult | None = None,
    ) -> list[ModelEntry]:
        """Get minion candidates with the designated model first, then alternatives."""
        primary_task_type: TaskType = (
            TaskType(classification_result.task_type_1[0])
            if classification_result.task_type_1
            else TaskType.OTHER
        )

        # REQUIRE domain classification - strict enforcement
        if not domain_classification:
            raise ValueError("Domain classification is required for minion selection")

        # Direct lookup - domain maps directly to model entry
        domain = domain_classification.domain
        if domain not in minion_domains:
            raise ValueError(f"Domain {domain.value} not supported in minion domains")

        primary_entry = minion_domains[domain]

        # Get alternatives (excluding the primary)
        alternatives = self.get_minion_alternatives(
            primary_minion=primary_entry.model_name,
            primary_provider=primary_entry.providers[0].value,
        )

        # Return with primary first, then alternatives
        candidates = [primary_entry, *alternatives]

        self.log(
            "minion_candidates_selected",
            {
                "domain": domain.value,
                "task_type": primary_task_type.value,
                "confidence": domain_classification.confidence,
                "primary_model": primary_entry.model_name,
                "primary_providers": [p.value for p in primary_entry.providers],
                "total_candidates": len(candidates),
            },
        )
        return candidates

    def get_available_minions(self) -> list[str]:
        """Get all available minion models from the domain mappings."""
        minions: set[str] = {entry.model_name for entry in minion_domains.values()}
        return sorted(minions)

    def get_minion_alternatives(
        self, primary_minion: str, primary_provider: str
    ) -> list[ModelEntry]:
        """Get alternative minion models with prioritization:
        1. Same model with different providers
        2. Different models
        """
        alternatives = []

        # First priority: Same model with different providers
        for domain_entry in minion_domains.values():
            if domain_entry.model_name == primary_minion:
                other_providers = [
                    p for p in domain_entry.providers if p.value != primary_provider
                ]
                if other_providers:
                    alternatives.append(
                        ModelEntry(
                            providers=other_providers,
                            model_name=domain_entry.model_name,
                        )
                    )
                break  # Only need one entry for the same model

        # Second priority: Different models (all providers for each)
        seen_models = {primary_minion}
        for domain_entry in minion_domains.values():
            if domain_entry.model_name not in seen_models:
                alternatives.append(
                    ModelEntry(
                        providers=domain_entry.providers,
                        model_name=domain_entry.model_name,
                    )
                )
                seen_models.add(domain_entry.model_name)

        self.log(
            "minion_alternatives_selected",
            {
                "primary_minion": primary_minion,
                "primary_provider": primary_provider,
                "alternatives_count": len(alternatives),
                "same_model_different_provider_count": sum(
                    1 for alt in alternatives if alt.model_name == primary_minion
                ),
                "alternatives": [
                    {
                        "model": alt.model_name,
                        "providers": [p.value for p in alt.providers],
                    }
                    for alt in alternatives[:5]
                ],
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
