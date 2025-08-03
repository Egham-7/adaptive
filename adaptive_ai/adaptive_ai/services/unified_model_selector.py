"""
Clean, readable model selector that handles both registry and custom models.
Simple, efficient, and scalable implementation.
"""

from collections import defaultdict
import logging
from typing import Any

from adaptive_ai.models.llm_core_models import ModelCapability
from adaptive_ai.models.unified_model import (
    Model,
    create_model_from_capability,
    create_model_from_dict,
)

logger = logging.getLogger(__name__)


class ModelSelector:
    """Selects the best models for a given task and complexity."""

    def __init__(self, registry_models: list[ModelCapability] | None = None):
        """Initialize with optional registry models."""
        self.registry_models = self._convert_registry_models(registry_models or [])
        self._task_index = self._build_task_index()

    def find_best_models(
        self,
        task_type: str,
        prompt_complexity: float,
        prompt_token_count: int,
        custom_models: list[dict[str, Any]] | None = None,
        prefer_cost_over_performance: bool = True,
        max_models: int = 5,
    ) -> list[Model]:
        """
        Find the best models for the given requirements.
        """
        all_models = self._gather_candidate_models(task_type, custom_models)

        capable_models = self._filter_capable_models(
            models=all_models,
            prompt_complexity=prompt_complexity,
            prompt_token_count=prompt_token_count,
        )

        sorted_models = self._sort_by_preference(
            capable_models, prefer_cost_over_performance
        )

        return sorted_models[:max_models]

    def _gather_candidate_models(
        self, task_type: str, custom_models: list[dict[str, Any]] | None
    ) -> list[Model]:
        """Collect all models that could potentially handle this task."""
        candidates = []

        candidates.extend(self._get_registry_models_for_task(task_type))

        if custom_models:
            candidates.extend(self._convert_custom_models(custom_models))

        seen = set()
        unique_candidates = []
        for model in candidates:
            if model.unique_id not in seen:
                seen.add(model.unique_id)
                unique_candidates.append(model)

        logger.info(
            f"Gathered {len(unique_candidates)} candidate models for task {task_type}"
        )
        return unique_candidates

    def _filter_capable_models(
        self, models: list[Model], prompt_complexity: float, prompt_token_count: int
    ) -> list[Model]:
        """Keep only models that can handle the complexity and token count."""
        capable = []

        for model in models:
            if not model.can_handle_prompt_size(prompt_token_count):
                logger.debug(
                    f"Model {model.unique_id} rejected: insufficient context length "
                    f"({model.max_context_tokens} < {prompt_token_count})"
                )
                continue

            min_required_complexity = prompt_complexity * 0.8
            if model.complexity_level < min_required_complexity:
                logger.debug(
                    f"Model {model.unique_id} rejected: insufficient complexity "
                    f"({model.complexity_level} < {min_required_complexity})"
                )
                continue

            capable.append(model)

        logger.info(
            f"Filtered to {len(capable)} capable models from {len(models)} candidates"
        )
        return capable

    def _sort_by_preference(
        self, models: list[Model], prefer_cost: bool
    ) -> list[Model]:
        """Sort models by user preference: cost-first or performance-first."""
        if prefer_cost:
            sorted_models = sorted(
                models,
                key=lambda m: (m.cost_per_million_input_tokens, -m.complexity_level),
            )
            logger.debug("Sorted models by cost preference (cheapest first)")
        else:
            sorted_models = sorted(
                models,
                key=lambda m: (-m.complexity_level, m.cost_per_million_input_tokens),
            )
            logger.debug("Sorted models by performance preference (most capable first)")

        return sorted_models

    def _get_registry_models_for_task(self, task_type: str) -> list[Model]:
        """Get registry models that are good for the specified task."""
        task_models = self._task_index.get(task_type, [])
        general_models = self._task_index.get(None, [])

        all_models = task_models + general_models
        logger.debug(
            f"Found {len(task_models)} task-specific + {len(general_models)} general models"
        )

        return all_models

    def _convert_custom_models(
        self, custom_model_specs: list[dict[str, Any]]
    ) -> list[Model]:
        """Convert user-provided model specifications to Model objects."""
        models = []

        for spec in custom_model_specs:
            try:
                model = create_model_from_dict(spec)
                models.append(model)
                logger.info(f"Successfully converted custom model: {model.unique_id}")
            except KeyError as e:
                logger.warning(
                    f"Skipping invalid custom model spec, missing field: {e}"
                )
                continue
            except Exception as e:
                logger.warning(f"Skipping invalid custom model spec: {e}")
                continue

        return models

    def _convert_registry_models(
        self, registry_models: list[ModelCapability]
    ) -> list[Model]:
        """Convert registry ModelCapability objects to unified Model objects."""
        models = []

        for capability in registry_models:
            try:
                model = create_model_from_capability(capability)
                models.append(model)
            except Exception as e:
                logger.warning(
                    f"Failed to convert registry model {capability.model_name}: {e}"
                )
                continue

        logger.info(f"Converted {len(models)} registry models")
        return models

    def _build_task_index(self) -> dict[str | None, list[Model]]:
        """Build an index of models by their best task type for fast lookup."""
        task_index = defaultdict(list)

        for model in self.registry_models:
            task_index[model.best_for_task].append(model)

        logger.debug(f"Built task index with {len(task_index)} task categories")
        return dict(task_index)

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the current model selection setup."""
        total_models = len(self.registry_models)
        task_distribution = {}

        for task, models in self._task_index.items():
            task_name = task or "general_purpose"
            task_distribution[task_name] = len(models)

        cost_distribution = {
            "budget_friendly": len(
                [m for m in self.registry_models if m.is_budget_friendly]
            ),
            "premium": len(
                [m for m in self.registry_models if not m.is_budget_friendly]
            ),
        }

        return {
            "total_registry_models": total_models,
            "task_distribution": task_distribution,
            "cost_distribution": cost_distribution,
        }
