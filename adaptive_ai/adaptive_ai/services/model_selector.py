# mypy: disable-error-code=import
from typing import Any

from adaptive_ai.config import (
    minion_domains,
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
from adaptive_ai.models.unified_model import Model
from adaptive_ai.services.model_registry import model_registry
from adaptive_ai.services.unified_model_selector import (
    ModelSelector as UnifiedModelSelector,
)


class LitLoggerProtocol:
    def log(self, key: str, value: Any) -> None: ...


class ModelSelectionService:
    def __init__(self, lit_logger: LitLoggerProtocol | None = None) -> None:
        # Build model capabilities from YAML database via model registry
        self._all_model_capabilities_by_id: dict[
            tuple[ProviderType, str], ModelCapability
        ] = {}
        # Cache for eligible providers per model and token count (optimized with token bucketing)
        self._eligible_providers_cache: dict[str, frozenset[ProviderType | str]] = {}

        # Pre-computed mapping of models to their available providers (frozenset for O(1) lookups)
        self._model_to_providers: dict[str, frozenset[ProviderType | str]] = {}

        # Pre-computed reverse mapping: provider -> models for faster filtering
        self._provider_to_models: dict[ProviderType | str, frozenset[str]] = {}

        # Pre-computed context length lookup for faster capability checks
        self._model_context_limits: dict[tuple[ProviderType | str, str], int] = {}

        self._build_capabilities_from_registry()
        self._build_optimized_caches()

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

    def _build_capabilities_from_registry(self) -> None:
        """Build model capabilities from YAML database via model registry."""
        from adaptive_ai.services.yaml_model_loader import yaml_model_db

        # Load YAML models
        yaml_model_db.load_models()

        # Get all models from registry and build capability lookup
        all_models = model_registry.get_all_valid_models()
        for model_name in all_models:
            capability = model_registry.get_model_capability(model_name)
            if capability:
                self._all_model_capabilities_by_id[
                    (capability.provider, capability.model_name)
                ] = capability

    def _build_optimized_caches(self) -> None:
        """Pre-compute optimized lookup structures for faster model selection."""
        # Temporary builders using sets for O(1) operations during construction
        model_providers_builder: dict[str, set[ProviderType | str]] = {}
        provider_models_builder: dict[ProviderType | str, set[str]] = {}

        for (
            provider,
            model_name,
        ), capability in self._all_model_capabilities_by_id.items():
            # Build model -> providers mapping
            if model_name not in model_providers_builder:
                model_providers_builder[model_name] = set()
            model_providers_builder[model_name].add(provider)

            # Build provider -> models mapping
            if provider not in provider_models_builder:
                provider_models_builder[provider] = set()
            provider_models_builder[provider].add(model_name)

            # Pre-compute context limits for O(1) capability checks
            self._model_context_limits[(provider, model_name)] = (
                capability.max_context_tokens or 4096
            )

        # Convert to frozensets for immutable, hashable, and faster lookup
        self._model_to_providers = {
            model: frozenset(providers)
            for model, providers in model_providers_builder.items()
        }
        self._provider_to_models = {
            provider: frozenset(models)
            for provider, models in provider_models_builder.items()
        }

    def _get_eligible_providers_for_model(
        self,
        model_entry: ModelEntry,
        eligible_providers: frozenset[ProviderType | str],
        prompt_token_count: int,
    ) -> list[ProviderType | str]:
        """Get eligible providers for a model that meet capability requirements (optimized)."""
        # Optimize cache key with larger buckets for better hit rate
        token_bucket = (prompt_token_count // 2000) * 2000  # 2K token buckets
        cache_key = str((model_entry.model_name, token_bucket))

        # Check cache first - use frozenset intersection for O(1) filtering
        if cache_key in self._eligible_providers_cache:
            cached_providers = self._eligible_providers_cache[cache_key]
            # Fast set intersection instead of list comprehension
            result_set = cached_providers & eligible_providers
            return list(result_set)

        # Cache miss - compute using optimized lookups
        available_providers = self._model_to_providers.get(
            model_entry.model_name, frozenset()
        )

        # Fast set intersection to get candidate providers
        # Normalize both sets to strings for proper intersection
        model_providers_str = frozenset(
            p.value if hasattr(p, "value") else str(p) for p in model_entry.providers
        )
        available_providers_str = frozenset(
            p.value if hasattr(p, "value") else str(p) for p in available_providers
        )
        candidate_providers_str = model_providers_str & available_providers_str

        # Convert back to original provider objects for consistency
        candidate_providers = frozenset(
            p
            for p in model_entry.providers
            if (p.value if hasattr(p, "value") else str(p)) in candidate_providers_str
        )

        # If no registry providers found, assume custom model and trust user specification
        if not candidate_providers:
            candidate_providers = frozenset(model_entry.providers)

        # Check context limits using pre-computed lookup
        token_eligible_providers = set()
        for provider in candidate_providers:
            context_limit = self._model_context_limits.get(
                (provider, model_entry.model_name)  # type: ignore
            )
            # For custom models without registry data, trust the user's specification
            if context_limit and context_limit >= prompt_token_count:
                token_eligible_providers.add(provider)
            elif not context_limit:
                # No registry data - assume custom model can handle the tokens
                token_eligible_providers.add(provider)

        # Cache as frozenset for future intersections
        token_eligible_frozenset = frozenset(token_eligible_providers)
        self._eligible_providers_cache[cache_key] = token_eligible_frozenset

        # Return intersection with current eligible providers
        result_set = token_eligible_frozenset & frozenset(eligible_providers)
        return list(result_set)  # type: ignore

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

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics in consistent format."""
        return {
            "eligible_providers_cache": {
                "size": len(self._eligible_providers_cache),
                "total_models_tracked": len(self._model_to_providers),
            }
        }

    def clear_cache(self) -> None:
        """Clear the eligible providers cache (useful for testing or memory management)."""
        self._eligible_providers_cache.clear()
        self.log("cache_cleared", {"cache_type": "eligible_providers"})

    def _get_default_models(self) -> list[ModelCapability]:
        """Get default model capabilities from YAML database"""
        return list(self._all_model_capabilities_by_id.values())

    def _use_user_specified_models(
        self,
        user_models: list[ModelCapability],
        classification_result: ClassificationResult,
        prompt_token_count: int,
        request: ModelSelectionRequest,
    ) -> list[ModelEntry]:
        """Convert user-specified ModelCapability objects to ModelEntry objects and apply pipeline."""
        model_entries = []

        self.log("user_specified_models_count", {"count": len(user_models)})

        # Convert all ModelCapability objects to ModelEntry objects
        for model_capability in user_models:
            try:
                # Use the provider directly - no fallback needed
                # Supports both ProviderType enum and custom provider strings
                provider = model_capability.provider

                model_entry = ModelEntry(
                    providers=[provider], model_name=model_capability.model_name
                )
                model_entries.append(model_entry)

                self.log(
                    "user_model_added",
                    {
                        "provider": str(provider),
                        "model": model_capability.model_name,
                        "provider_type": type(provider).__name__,
                    },
                )
            except Exception as e:
                self.log(
                    "model_entry_error",
                    {
                        "provider": model_capability.provider,
                        "model": model_capability.model_name,
                        "error": str(e),
                    },
                )
                continue

        if not model_entries:
            raise ValueError("No valid user-specified models found")

        # Log initial user-specified models
        self.log(
            "user_specified_models_initial",
            {
                "models": [
                    f"{m.providers[0].value if hasattr(m.providers[0], 'value') else m.providers[0]}:{m.model_name}"
                    for m in model_entries
                ],
                "count": len(model_entries),
                "prompt_token_count": prompt_token_count,
            },
        )

        # Apply pipeline steps to user models
        # PIPELINE STEP 1: Apply capability constraints (context length, etc.)
        eligible_providers = frozenset(m.providers[0] for m in model_entries)
        candidate_models = self._apply_capability_constraints(
            model_entries, eligible_providers, prompt_token_count  # type: ignore
        )

        # PIPELINE STEP 2: Apply cost optimization
        candidate_models = self._apply_cost_optimization(
            candidate_models, request, prompt_token_count
        )

        # Check if any models survived the pipeline
        if not candidate_models:
            # Fallback to first user-specified model
            candidate_models = [model_entries[0]]
            provider_str = (
                model_entries[0].providers[0].value
                if hasattr(model_entries[0].providers[0], "value")
                else model_entries[0].providers[0]
            )
            self.log(
                "user_models_pipeline_fallback",
                {
                    "fallback_model": f"{provider_str}:{model_entries[0].model_name}",
                    "reason": "No user models passed pipeline constraints",
                    "original_count": len(model_entries),
                    "prompt_token_count": prompt_token_count,
                },
            )

        # Update metrics
        self._update_metrics(classification_result, candidate_models)

        # Log final user-specified models
        self.log(
            "user_specified_models_final",
            {
                "models": [
                    f"{m.providers[0].value if hasattr(m.providers[0], 'value') else m.providers[0]}:{m.model_name}"
                    for m in candidate_models
                ],
                "count": len(candidate_models),
                "filtered_count": len(model_entries) - len(candidate_models),
                "used_fallback": len(candidate_models) == 1 and len(model_entries) > 1,
            },
        )

        return candidate_models

    def _use_system_model_selection(
        self,
        request: ModelSelectionRequest,
        classification_result: ClassificationResult,
        prompt_token_count: int,
        domain_classification: DomainClassificationResult | None = None,
    ) -> list[ModelEntry]:
        """Use system's default model selection logic."""
        self.log(
            "system_model_selection_called",
            {
                "prompt_token_count": prompt_token_count,
                "classification_result": classification_result.model_dump(
                    exclude_unset=True
                ),
            },
        )

        # PIPELINE: Start with task-appropriate models
        candidate_models = self._get_initial_candidates(
            classification_result, domain_classification
        )

        # PIPELINE STEP 1: Apply capability constraints (context length, etc.)
        eligible_providers = frozenset(
            capability.provider
            for capability in self._all_model_capabilities_by_id.values()
        )
        candidate_models = self._apply_capability_constraints(
            candidate_models, eligible_providers, prompt_token_count
        )

        # PIPELINE STEP 2: Apply cost optimization
        candidate_models = self._apply_cost_optimization(
            candidate_models, request, prompt_token_count
        )

        # Update metrics
        self._update_metrics(classification_result, candidate_models)

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

    def _apply_cost_optimization(
        self,
        candidate_models: list[ModelEntry],
        request: ModelSelectionRequest,
        prompt_token_count: int,
    ) -> list[ModelEntry]:
        """Apply cost-based ranking if cost_bias is provided."""
        cost_bias = (
            request.protocol_manager_config.cost_bias
            if request.protocol_manager_config
            else None
        )

        if cost_bias is None or not candidate_models:
            return candidate_models

        from adaptive_ai.services.cost_optimizer import rank_models_by_cost_performance

        original_count = len(candidate_models)
        original_top_model = (
            candidate_models[0].model_name if candidate_models else None
        )

        # Apply cost-performance ranking
        optimized_models = rank_models_by_cost_performance(
            model_entries=candidate_models,
            cost_bias=cost_bias,
            model_capabilities=self._all_model_capabilities_by_id,  # type: ignore
            estimated_tokens=prompt_token_count,
        )

        new_top_model = optimized_models[0].model_name if optimized_models else None

        self.log(
            "cost_bias_routing",
            {
                "cost_bias": cost_bias,
                "original_model_count": original_count,
                "original_top_model": original_top_model,
                "cost_optimized_top_model": new_top_model,
                "reranked_models": [entry.model_name for entry in optimized_models[:5]],
                "ranking_changed": original_top_model != new_top_model,
            },
        )

        return optimized_models

    def _get_initial_candidates(
        self,
        classification_result: ClassificationResult,
        domain_classification: DomainClassificationResult | None = None,
    ) -> list[ModelEntry]:
        """Get initial candidate models based on task type."""
        primary_task_type: TaskType = (
            TaskType(classification_result.task_type_1[0])
            if classification_result.task_type_1
            else TaskType.OTHER
        )

        self.log("primary_task_type", primary_task_type.value)

        # Get task-appropriate models
        task_mapping = task_model_mappings_data.get(primary_task_type)
        if not task_mapping:
            raise ValueError(
                f"No task mapping found for task type: {primary_task_type}"
            )

        candidate_models = task_mapping.model_entries

        self.log(
            "initial_candidates",
            {
                "task_type": primary_task_type.value,
                "models_count": len(candidate_models),
                "top_model": (
                    candidate_models[0].model_name if candidate_models else None
                ),
                "domain_info": (
                    domain_classification.domain.value
                    if domain_classification
                    else "not_used"
                ),
            },
        )

        return candidate_models

    def _apply_capability_constraints(
        self,
        candidate_models: list[ModelEntry],
        eligible_providers: frozenset[ProviderType | str],
        prompt_token_count: int,
    ) -> list[ModelEntry]:
        """Filter models based on capability constraints like context length."""
        filtered_models = []
        seen_model_names: set[str] = set()

        for model_entry in candidate_models:
            if model_entry.model_name in seen_model_names:
                continue

            # Get providers that meet capability requirements
            eligible_providers_for_model = self._get_eligible_providers_for_model(
                model_entry, eligible_providers, prompt_token_count
            )

            if eligible_providers_for_model:
                # Create new ModelEntry with only eligible providers
                filtered_entry = ModelEntry(
                    providers=list(eligible_providers_for_model),
                    model_name=model_entry.model_name,
                )
                filtered_models.append(filtered_entry)
                seen_model_names.add(model_entry.model_name)

                self.log(
                    "model_capability_approved",
                    {
                        "model": model_entry.model_name,
                        "eligible_providers": [
                            p.value if hasattr(p, "value") else str(p)
                            for p in eligible_providers_for_model
                        ],
                    },
                )

        self.log(
            "capability_constraints_applied",
            {
                "original_count": len(candidate_models),
                "filtered_count": len(filtered_models),
                "token_limit": prompt_token_count,
            },
        )

        return filtered_models

    def _update_metrics(
        self,
        classification_result: ClassificationResult,
        candidate_models: list[ModelEntry],
    ) -> None:
        """Update selection metrics."""
        self.selection_metrics["total_selections"] += 1

        # Track task usage
        primary_task_type = (
            TaskType(classification_result.task_type_1[0])
            if classification_result.task_type_1
            else TaskType.OTHER
        )
        task_key = primary_task_type.value
        self.selection_metrics["task_usage"][task_key] = (
            self.selection_metrics["task_usage"].get(task_key, 0) + 1
        )

        # Track model usage
        for model_entry in candidate_models:
            if model_entry.providers:
                first_provider = model_entry.providers[0]
                provider_str = (
                    first_provider.value
                    if hasattr(first_provider, "value")
                    else str(first_provider)
                )
                model_key = f"{provider_str}:{model_entry.model_name}"
                self.selection_metrics["model_usage"][model_key] = (
                    self.selection_metrics["model_usage"].get(model_key, 0) + 1
                )

        self.log("final_candidate_count", len(candidate_models))

    def select_candidate_models(
        self,
        request: ModelSelectionRequest,
        classification_result: ClassificationResult,
        prompt_token_count: int,
        domain_classification: DomainClassificationResult | None = None,
    ) -> list[ModelEntry]:
        # Check if user provided their own models
        if request.protocol_manager_config and request.protocol_manager_config.models:
            return self._use_user_specified_models(
                request.protocol_manager_config.models,
                classification_result,
                prompt_token_count,
                request,
            )
        else:
            return self._use_system_model_selection(
                request,
                classification_result,
                prompt_token_count,
                domain_classification,
            )

    def get_minion_candidates(
        self,
        domain_classification: DomainClassificationResult,
    ) -> list[ModelEntry]:

        # Direct lookup - domain maps directly to model entry
        domain = domain_classification.domain
        if domain not in minion_domains:
            raise ValueError(f"Domain {domain.value} not supported in minion domains")

        primary_entry = minion_domains[domain]

        # Get alternatives (excluding the primary)
        alternatives = self.get_minion_alternatives(
            primary_minion=primary_entry.model_name,
            primary_provider=(
                primary_entry.providers[0].value
                if hasattr(primary_entry.providers[0], "value")
                else str(primary_entry.providers[0])
            ),
        )

        # Return with primary first, then alternatives
        candidates = [primary_entry, *alternatives]

        self.log(
            "minion_candidates_selected",
            {
                "domain": domain.value,
                "confidence": domain_classification.confidence,
                "primary_model": primary_entry.model_name,
                "primary_providers": [
                    p.value if hasattr(p, "value") else str(p)
                    for p in primary_entry.providers
                ],
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
                    p
                    for p in domain_entry.providers
                    if (p.value if hasattr(p, "value") else str(p)) != primary_provider
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
                        "providers": [
                            p.value if hasattr(p, "value") else str(p)
                            for p in alt.providers
                        ],
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

    def enrich_partial_models(
        self, partial_models: list[ModelCapability]
    ) -> list[ModelCapability]:
        """
        Enrich partial ModelCapability objects with complete capability information.
        Now supports custom models with task_type and complexity specifications.

        Args:
            partial_models: List of ModelCapability objects that may be partial
                          (only provider/model_name) or fully specified with task info

        Returns:
            List of fully enriched ModelCapability objects

        Raises:
            ValueError: If any models cannot be enriched or are invalid
        """
        if not partial_models:
            return []

        enriched_capabilities = []
        invalid_models = []

        for model_cap in partial_models:
            try:
                # Check if this is a fully specified custom model
                required_fields = [
                    "cost_per_1m_input_tokens",
                    "cost_per_1m_output_tokens",
                    "max_context_tokens",
                    "supports_function_calling",
                ]

                is_fully_specified = all(
                    hasattr(model_cap, field) and getattr(model_cap, field) is not None
                    for field in required_fields
                )

                if is_fully_specified:
                    # Custom model with full specification - pass through
                    enriched_capabilities.append(model_cap)
                    self.log(
                        "custom_model_accepted",
                        {
                            "model": f"{model_cap.provider}/{model_cap.model_name}",
                            "task_type": getattr(model_cap, "task_type", "general"),
                            "complexity": getattr(model_cap, "complexity", "medium"),
                            "reason": "fully_specified_custom_model",
                        },
                    )
                    continue

                # Handle partial models that need registry lookup
                if model_cap.provider and model_cap.model_name:
                    # Try registry lookup for partial models
                    model_key = (model_cap.provider, model_cap.model_name)
                    full_capability = self._all_model_capabilities_by_id.get(model_key)
                    if full_capability:
                        enriched_capabilities.append(full_capability)
                        self.log(
                            "registry_model_found",
                            {"model": f"{model_cap.provider}/{model_cap.model_name}"},
                        )
                    else:
                        # Not in registry - might be a custom model missing required fields
                        invalid_models.append(
                            f"{model_cap.provider}/{model_cap.model_name} (not in registry and missing required fields)"
                        )

                elif model_cap.model_name and not model_cap.provider:
                    # Search registry by model name only - try hardcoded first
                    found_capability = None
                    for (
                        _,
                        model_name,
                    ), capability in self._all_model_capabilities_by_id.items():
                        if model_name == model_cap.model_name:
                            found_capability = capability
                            break

                    # If not found in hardcoded registry, try YAML database
                    if not found_capability:
                        found_capability = model_registry.get_model_capability(
                            model_cap.model_name
                        )

                    if found_capability:
                        enriched_capabilities.append(found_capability)
                        self.log(
                            "registry_model_found_by_name",
                            {
                                "model": model_cap.model_name,
                                "provider": found_capability.provider,
                                "source": (
                                    "yaml_database"
                                    if model_cap.model_name
                                    not in [
                                        mc[1]
                                        for mc in self._all_model_capabilities_by_id.keys()
                                    ]
                                    else "hardcoded_registry"
                                ),
                            },
                        )
                    else:
                        invalid_models.append(f"unknown/{model_cap.model_name}")

                else:
                    invalid_models.append("missing_provider_and_model_name")

            except Exception as e:
                self.log(
                    "model_enrichment_error",
                    {"model": str(model_cap), "error": str(e)},
                )
                invalid_models.append(f"error_processing_{model_cap}: {e!s}")

        # Log results but don't fail on invalid models - let the system continue
        if invalid_models:
            self.log(
                "some_models_invalid",
                {
                    "invalid_models": invalid_models,
                    "valid_models": len(enriched_capabilities),
                },
            )

        self.log(
            "model_enrichment_complete",
            {
                "enriched_count": len(enriched_capabilities),
                "total_requested": len(partial_models),
                "custom_models": len(
                    [
                        m
                        for m in enriched_capabilities
                        if hasattr(m, "task_type") and m.task_type is not None
                    ]
                ),
            },
        )

        # Return what we have - even if some models were invalid
        return enriched_capabilities

    def select_models_unified(
        self,
        request: ModelSelectionRequest,
        classification_result: ClassificationResult,
        prompt_token_count: int,
        domain_classification: DomainClassificationResult | None = None,
    ) -> list[Model]:
        """
        Use the new unified model selector for clean, readable model selection.
        Supports both registry and custom models with task-aware selection.
        """

        # Extract task type from classification
        task_type = (
            classification_result.task_type_1[0]
            if classification_result.task_type_1
            else "OTHER"
        )

        # Get prompt complexity score
        prompt_complexity = (
            classification_result.prompt_complexity_score[0]
            if classification_result.prompt_complexity_score
            else 0.5
        )

        # Determine cost preference
        cost_bias = (
            request.protocol_manager_config.cost_bias
            if request.protocol_manager_config
            and request.protocol_manager_config.cost_bias is not None
            else 0.5
        )
        prefer_cost = cost_bias < 0.5

        # Prepare custom models if provided
        custom_models_dict = None
        if request.protocol_manager_config and request.protocol_manager_config.models:

            # Convert ModelCapability objects to dict format for unified selector
            custom_models_dict = []
            for model_cap in request.protocol_manager_config.models:
                if self._is_fully_specified_model(model_cap):
                    model_dict = {
                        "provider": str(model_cap.provider),
                        "model_name": model_cap.model_name,
                        "cost_per_1m_input_tokens": model_cap.cost_per_1m_input_tokens,
                        "cost_per_1m_output_tokens": model_cap.cost_per_1m_output_tokens,
                        "max_context_tokens": model_cap.max_context_tokens,
                        "supports_function_calling": model_cap.supports_function_calling,
                        "task_type": getattr(model_cap, "task_type", None),
                        "complexity": getattr(model_cap, "complexity", "medium"),
                    }
                    custom_models_dict.append(model_dict)

        # Get registry models as ModelCapability objects from YAML database
        registry_capabilities = list(self._all_model_capabilities_by_id.values())

        # Initialize unified selector
        unified_selector = UnifiedModelSelector(registry_capabilities)

        # Find best models using unified selector
        selected_models = unified_selector.find_best_models(
            task_type=task_type,
            prompt_complexity=prompt_complexity,
            prompt_token_count=prompt_token_count,
            custom_models=custom_models_dict,
            prefer_cost_over_performance=prefer_cost,
            max_models=10,
        )

        # Log the selection
        self.log(
            "unified_model_selection",
            {
                "task_type": task_type,
                "prompt_complexity": prompt_complexity,
                "prompt_tokens": prompt_token_count,
                "prefer_cost": prefer_cost,
                "selected_models": [m.unique_id for m in selected_models],
                "custom_models_provided": (
                    len(custom_models_dict) if custom_models_dict else 0
                ),
            },
        )

        return selected_models

    def _is_fully_specified_model(self, model_cap: ModelCapability) -> bool:
        """Check if a ModelCapability has all required fields for custom models."""
        required_fields = [
            "cost_per_1m_input_tokens",
            "cost_per_1m_output_tokens",
            "max_context_tokens",
            "supports_function_calling",
        ]

        return all(
            hasattr(model_cap, field) and getattr(model_cap, field) is not None
            for field in required_fields
        )
