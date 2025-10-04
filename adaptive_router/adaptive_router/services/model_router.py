"""Model routing service with complexity-aware intelligent selection."""

from __future__ import annotations

import logging

from adaptive_router.models.llm_core_models import (
    ModelCapability,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.services.model_registry import ModelRegistry
from adaptive_router.services.yaml_model_loader import YAMLModelDatabase
from adaptive_router.services.prompt_task_complexity_classifier import (
    PromptClassifier,
)

logger = logging.getLogger(__name__)


class ModelRouter:
    """Intelligent model routing with complexity-aware selection.

    Selects optimal LLM models based on task complexity, cost optimization,
    and model capability matching.
    """

    _DEFAULT_COST = 1.0

    def __init__(
        self,
        model_registry: ModelRegistry | None = None,
        yaml_db: YAMLModelDatabase | None = None,
        prompt_classifier: PromptClassifier | None = None,
    ) -> None:
        """Initialize router with model registry and classifier.

        Args:
            model_registry: Optional ModelRegistry instance. If not provided, creates one internally.
            yaml_db: Optional YAMLModelDatabase instance. Used only if model_registry is not provided.
            prompt_classifier: Optional PromptClassifier instance. If not provided, creates one internally.
        """
        if model_registry is None:
            if yaml_db is None:
                yaml_db = YAMLModelDatabase()
            model_registry = ModelRegistry(yaml_db)
        self._model_registry = model_registry

        if prompt_classifier is None:

            prompt_classifier = PromptClassifier()
        self._prompt_classifier = prompt_classifier

    def select_model(self, request: ModelSelectionRequest) -> ModelSelectionResponse:
        """Select optimal model based on prompt analysis.

        This is the main public API method. It handles classification and selection internally.

        Args:
            request: ModelSelectionRequest with prompt and optional models/cost_bias

        Returns:
            ModelSelectionResponse with selected provider, model, and alternatives

        Raises:
            ValueError: If no eligible models found or validation fails
            RuntimeError: If classification or routing fails
        """
        classification_dict = self._prompt_classifier.classify_prompt(request.prompt)

        from adaptive_router.models.llm_classification_models import (
            ClassificationResult,
        )

        classification = ClassificationResult(**classification_dict)

        task_type = (
            classification.task_type_1 if classification.task_type_1 else "Other"
        )
        task_complexity = (
            classification.prompt_complexity_score
            if classification.prompt_complexity_score is not None
            else 0.5
        )

        selected_models = self._select_models(
            task_complexity,
            task_type,
            request.models,
            request.cost_bias if request.cost_bias is not None else 0.5,
        )

        if not selected_models:
            raise ValueError("No eligible models found")

        best_model = selected_models[0]
        if not best_model.provider or not best_model.model_name:
            raise ValueError("Selected model missing provider or model_name")

        from adaptive_router.models.llm_core_models import Alternative

        alternatives = [
            Alternative(provider=alt.provider, model=alt.model_name)
            for alt in selected_models[1:]
            if alt.provider and alt.model_name
        ]

        return ModelSelectionResponse(
            provider=best_model.provider,
            model=best_model.model_name,
            alternatives=alternatives,
        )

    def _select_models(
        self,
        task_complexity: float,
        task_type: str,
        models_input: list[ModelCapability] | None = None,
        cost_bias: float = 0.5,
    ) -> list[ModelCapability]:
        """Select best models with complexity-aware routing."""
        if not 0.0 <= task_complexity <= 1.0:
            raise ValueError(f"task_complexity must be 0.0-1.0, got {task_complexity}")

        candidates = self._get_candidate_models(models_input, task_type)
        logger.debug(
            "Found candidate models",
            extra={"candidates_count": len(candidates), "task_type": task_type},
        )

        return self._rank_models(candidates, task_complexity, cost_bias)

    def _get_candidate_models(
        self, models_input: list[ModelCapability] | None, task_type: str
    ) -> list[ModelCapability]:
        """Get candidate models from input, handling full vs partial specs."""
        if not models_input:
            return self._get_all_models_for_task(task_type)

        candidates = []
        for model in models_input:
            if model.is_partial:
                matching = self._model_registry.find_models_matching_criteria(model)
                self._validate_resolved_models(matching)
                candidates.extend(matching)
            else:
                registry_model = self._model_registry.get_model_capability(
                    model.unique_id
                )
                candidates.append(registry_model or model)

        # Remove duplicates and filter by task type
        unique_models = self._deduplicate_models(candidates)
        return self._filter_by_task_type(unique_models, task_type)

    def _get_all_models_for_task(self, task_type: str) -> list[ModelCapability]:
        """Get all available models filtered by task type."""
        all_names = self._model_registry.get_all_model_names()
        models = []
        for name in all_names:
            model = self._model_registry.get_model_capability(name)
            if model:
                models.append(model)
        return self._filter_by_task_type(models, task_type)

    def _validate_resolved_models(self, models: list[ModelCapability]) -> None:
        """Validate that registry returned complete models."""
        for model in models:
            if model.is_partial:
                raise ValueError(
                    f"Registry returned partial model: {model}. "
                    "Registry data incomplete or resolution logic failed."
                )

    def _deduplicate_models(
        self, models: list[ModelCapability]
    ) -> list[ModelCapability]:
        """Remove duplicate models while preserving order."""
        seen = set()
        result = []
        for model in models:
            if model.unique_id not in seen:
                seen.add(model.unique_id)
                result.append(model)
        return result

    def _filter_by_task_type(
        self, models: list[ModelCapability], task_type: str
    ) -> list[ModelCapability]:
        """Filter models based on task type compatibility."""
        if not task_type:
            return models

        filtered = [m for m in models if self._supports_task_type(m, task_type)]
        logger.debug(
            "Task filtering completed",
            extra={
                "task_type": task_type,
                "input_models": len(models),
                "filtered_models": len(filtered),
            },
        )
        return filtered

    def _supports_task_type(self, model: ModelCapability, task_type: str) -> bool:
        """Check if a model supports a specific task type."""
        if not model.task_type:
            return True

        model_task = str(model.task_type).lower().strip()
        target_task = str(task_type).lower().strip()

        # Special compatibility rules
        if target_task == "other":
            return True
        if model_task == "text generation":
            return True  # Most general category
        if model_task == "code generation" and target_task == "text generation":
            return True

        return model_task == target_task

    def _rank_models(
        self,
        models: list[ModelCapability],
        task_complexity: float,
        cost_bias: float,
    ) -> list[ModelCapability]:
        """Rank models by complexity suitability and cost bias."""
        if not models:
            return models

        # Filter unsuitable models for complex tasks
        filtered = self._filter_by_complexity_threshold(models, task_complexity)
        if not filtered:
            filtered = models  # Fallback if filtering removes everything

        # Calculate scores and sort
        scored = [
            (self._calculate_model_score(m, filtered, task_complexity, cost_bias), m)
            for m in filtered
        ]

        return [model for _, model in sorted(scored, key=self._sort_key, reverse=True)]

    def _filter_by_complexity_threshold(
        self, models: list[ModelCapability], task_complexity: float
    ) -> list[ModelCapability]:
        """Filter out models that can't handle the required complexity."""
        if task_complexity <= 0.5:
            return models

        threshold_context = 8000 if task_complexity > 0.7 else 4000
        min_complexity = 0.2 if task_complexity > 0.7 else 0.0

        filtered = []
        for model in models:
            if task_complexity > 0.7:
                model_complexity = self._get_model_complexity(model, models)
                if (model.max_context_tokens or 0) < threshold_context:
                    continue
                if model_complexity < min_complexity:
                    continue
            elif (model.max_context_tokens or 0) < threshold_context:
                continue

            filtered.append(model)

        return filtered

    def _calculate_model_score(
        self,
        model: ModelCapability,
        models: list[ModelCapability],
        task_complexity: float,
        cost_bias: float,
    ) -> float:
        """Calculate overall model score combining complexity and cost factors."""
        complexity_score = self._calculate_complexity_score(
            model, models, task_complexity
        )
        cost_score = self._calculate_cost_score(model, models)

        # Blend scores based on cost bias
        return (1 - cost_bias) * cost_score + cost_bias * complexity_score

    def _calculate_complexity_score(
        self,
        model: ModelCapability,
        models: list[ModelCapability],
        task_complexity: float,
    ) -> float:
        """Calculate how well model complexity matches task requirements."""
        model_complexity = self._get_model_complexity(model, models)

        # Alignment score (closer to task complexity is better)
        alignment = 1.0 - abs(task_complexity - model_complexity)

        # Capability bonus (can handle required complexity)
        if model_complexity >= task_complexity:
            capability_bonus = 0.2 * (1.0 - abs(task_complexity - model_complexity))
            alignment += capability_bonus

        return alignment

    def _calculate_cost_score(
        self, model: ModelCapability, models: list[ModelCapability]
    ) -> float:
        """Calculate cost efficiency score (higher = more cost efficient)."""
        model_cost = (model.cost_per_1m_input_tokens or 0) + (
            model.cost_per_1m_output_tokens or 0
        )
        model_capability = model.max_context_tokens or 0

        # Normalize against other models
        costs = [
            (m.cost_per_1m_input_tokens or 0) + (m.cost_per_1m_output_tokens or 0)
            for m in models
        ]
        capabilities = [m.max_context_tokens or 0 for m in models]

        max_cost = max(costs) if costs else 1
        max_capability = max(capabilities) if capabilities else 1

        # Cost score (lower cost = higher score)
        cost_efficiency = 1 - (model_cost / max_cost if max_cost > 0 else 0)

        # Capability score
        capability_score = (
            model_capability / max_capability if max_capability > 0 else 0
        )

        # Balanced score
        return 0.6 * cost_efficiency + 0.4 * capability_score

    def _get_model_complexity(
        self, model: ModelCapability, models: list[ModelCapability]
    ) -> float:
        """Get model complexity score, using cost as proxy if not specified."""
        if model.complexity is not None:
            return model.complexity_score

        # Use cost as complexity proxy
        valid_costs = [
            m.cost_per_1m_input_tokens
            for m in models
            if m.cost_per_1m_input_tokens is not None
        ]
        max_cost = max(valid_costs) if valid_costs else self._DEFAULT_COST

        return (model.cost_per_1m_input_tokens or 0) / max_cost if max_cost > 0 else 0.5

    @staticmethod
    def _sort_key(scored_model: tuple[float, ModelCapability]) -> tuple[float, str]:
        """Generate sort key for stable sorting."""
        score, model = scored_model
        return (score, model.model_name or "")
