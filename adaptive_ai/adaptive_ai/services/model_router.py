"""Model routing service with complexity-aware intelligent selection."""

from __future__ import annotations

import logging

from adaptive_ai.models.llm_core_models import (
    ModelCapability,
)
from adaptive_ai.models.llm_enums import TaskType
from adaptive_ai.services.model_registry import model_registry

# yaml_model_loader removed - using model_registry directly

logger = logging.getLogger(__name__)


class ModelRouter:
    """Intelligent model routing with complexity-aware selection.

    The ModelRouter selects optimal LLM models based on task complexity analysis,
    cost optimization preferences, and model capability matching.

    Key Features:
    - Handles full ModelCapability objects (route to specific models)
    - Handles partial ModelCapability objects (use as filter criteria)
    - Complexity-aware model selection using task classification
    - Cost optimization with configurable bias
    - Simple filtering using ModelCapability fields directly

    Architecture:
    1. Process input models:
       - Full models: Use directly for routing
       - Partial models: Find matching models using criteria
       - No models: Use all available models
    2. Apply complexity-based scoring and ranking
    3. Apply cost bias to balance price vs. performance
    4. Return ranked list of suitable models

    Example:
        >>> router = ModelRouter()

        >>> # Route to specific models
        >>> full_models = [
        ...     ModelCapability(provider="openai", model_name="gpt-4", ...),
        ...     ModelCapability(provider="anthropic", model_name="claude-3", ...)
        ... ]
        >>> models = router.select_models(0.7, TaskType.CODE_GENERATION, full_models)

        >>> # Use partial models as filter criteria
        >>> criteria = [
        ...     ModelCapability(provider="openai", max_context_tokens=8000),  # OpenAI with 8K+ context
        ...     ModelCapability(supports_function_calling=True, cost_per_1m_input_tokens=10.0)  # Functions + budget
        ... ]
        >>> models = router.select_models(0.5, TaskType.TEXT_GENERATION, criteria)
    """

    # Constants for scoring and thresholds
    _DEFAULT_COST = 1.0  # Default cost for normalization

    def __init__(self) -> None:
        """Initialize router."""
        pass

    def select_models(
        self,
        task_complexity: float,
        task_type: TaskType,
        models_input: list[ModelCapability] | None = None,
        cost_bias: float = 0.5,
    ) -> list[ModelCapability]:
        """Select best models with complexity-aware routing.

        This is the main entry point for model selection. It processes the input models
        and applies intelligent routing based on whether they are full or partial specs.

        Args:
            task_complexity: Task complexity score (0.0-1.0) from ML classification
            task_type: Type of task for filtering compatibility
            models_input: Optional list of ModelCapability objects:
                         - Full models: Route directly to these specific models
                         - Partial models: Use as filter criteria to find matching models
                         - Mix allowed: Can have both full and partial in same list
                         - If None: Use all available models
            cost_bias: Cost vs. capability preference (0.0-1.0)

        Returns:
            List of ModelCapability objects ranked by suitability (best first)

        Examples:
            >>> # Use all models
            >>> models = router.select_models(0.5, TaskType.TEXT_GENERATION)

            >>> # Route to specific models
            >>> specific = [
            ...     ModelCapability(provider="openai", model_name="gpt-4", cost_per_1m_input_tokens=30.0, ...),
            ...     ModelCapability(provider="anthropic", model_name="claude-3", cost_per_1m_input_tokens=15.0, ...)
            ... ]
            >>> models = router.select_models(0.7, TaskType.CODE_GENERATION, specific)

            >>> # Use partial models as criteria
            >>> criteria = [
            ...     ModelCapability(provider="openai", max_context_tokens=8000),  # Find OpenAI models with 8K+ context
            ...     ModelCapability(supports_function_calling=True)              # Find any models with functions
            ... ]
            >>> models = router.select_models(0.6, TaskType.CODE_GENERATION, criteria)
        """
        if not 0.0 <= task_complexity <= 1.0:
            raise ValueError(
                f"task_complexity must be between 0.0 and 1.0, got {task_complexity}"
            )

        # Get candidate models based on input
        candidate_models = self._get_candidate_models(models_input, task_type)

        # Log registry usage for metrics
        logger.debug(f"Registry lookup found {len(candidate_models)} candidate models")

        # Apply integrated complexity and cost routing
        final_models = self._apply_integrated_routing(
            candidate_models, task_complexity, cost_bias
        )

        return final_models

    def _get_candidate_models(
        self, models_input: list[ModelCapability] | None, task_type: TaskType
    ) -> list[ModelCapability]:
        """Get candidate models from input, handling full vs partial specs."""
        if models_input is None or len(models_input) == 0:
            # No models specified or empty array - use all available models from global registry
            all_model_names = model_registry.get_all_model_names()
            all_models = []
            for model_name in all_model_names:
                model_capability = model_registry.get_model_capability(model_name)
                if model_capability:
                    all_models.append(model_capability)
            return self._filter_by_task_type(all_models, task_type)

        candidate_models = []
        for model in models_input:
            if model.is_partial:
                # Partial model - use as filter criteria
                matching_models = model_registry.find_models_matching_criteria(model)
                # Validate that registry returned complete models
                for resolved_model in matching_models:
                    if resolved_model.is_partial:
                        raise ValueError(
                            f"Registry returned partial model after resolution: {resolved_model}. "
                            "This indicates incomplete registry data or resolution logic failure."
                        )
                candidate_models.extend(matching_models)
            else:
                # Full model - use directly (but verify it exists in global registry)
                registry_model = model_registry.get_model_capability(model.unique_id)
                if registry_model:
                    candidate_models.append(registry_model)
                else:
                    # Model not in registry, use the provided model as-is
                    candidate_models.append(model)

        # Remove duplicates using dict comprehension for order preservation
        unique_models = list(
            {model.unique_id: model for model in candidate_models}.values()
        )

        # Filter by task type
        return self._filter_by_task_type(unique_models, task_type)

    def _filter_by_task_type(
        self, models: list[ModelCapability], task_type: TaskType
    ) -> list[ModelCapability]:
        """Filter models based on task type compatibility."""
        if not task_type:
            return models

        filtered_models = [
            model
            for model in models
            if self._model_supports_task_type(model, task_type)
        ]

        logger.debug(
            f"Task type filtering: {task_type.value} - {len(models)} total models, {len(filtered_models)} filtered models"
        )

        return filtered_models

    def _model_supports_task_type(
        self, model: ModelCapability, task_type: TaskType
    ) -> bool:
        """Check if a model supports a specific task type."""
        # If model has no task type specified, assume it supports all tasks
        if model.task_type is None:
            return True

        # Normalize both sides to comparable strings for enum vs string comparison
        model_task_str = None
        if model.task_type is not None:
            # Handle both string and enum types
            if hasattr(model.task_type, "value"):
                model_task_str = str(model.task_type.value).lower().strip()
            elif hasattr(model.task_type, "name"):
                model_task_str = str(model.task_type.name).lower().strip()
            else:
                model_task_str = str(model.task_type).lower().strip()

        task_type_str = None
        if task_type is not None:
            # Handle both string and enum types
            if hasattr(task_type, "value"):
                task_type_str = str(task_type.value).lower().strip()
            elif hasattr(task_type, "name"):
                task_type_str = str(task_type.name).lower().strip()
            else:
                task_type_str = str(task_type).lower().strip()

        # Handle special cases for task type matching
        # TaskType.OTHER should match all models (it's a generic/fallback category)
        if task_type_str == "other":
            return True

        # TEXT_GENERATION models can handle ALL specific task types (it's the most general category)
        if model_task_str == "text generation":
            return True

        # CODE_GENERATION models can handle TEXT_GENERATION tasks
        if model_task_str == "code generation" and task_type_str == "text generation":
            return True

        # Compare normalized strings for exact matches
        return model_task_str == task_type_str

    def _calculate_complexity_score(
        self, model: ModelCapability, models: list[ModelCapability]
    ) -> float:
        """Calculate complexity score for a model."""
        if model.complexity is not None:
            return model.complexity_score

        # Use cost as complexity proxy
        # Precompute valid costs to avoid ValueError when no valid costs exist
        valid_costs = [
            m.cost_per_1m_input_tokens or 0
            for m in models
            if m.cost_per_1m_input_tokens
        ]

        max_cost = max(valid_costs) if valid_costs else self._DEFAULT_COST

        return (model.cost_per_1m_input_tokens or 0) / max_cost if max_cost > 0 else 0.5

    def _apply_complexity_routing(
        self, models: list[ModelCapability], task_complexity: float
    ) -> list[ModelCapability]:
        """Apply complexity-based routing to rank models by task suitability.

        This method ranks models based on how well they match the required task complexity,
        without considering cost factors. Models that can handle the required complexity
        receive higher scores. Basic models are filtered out for complex tasks.

        Args:
            models: List of available models to rank
            task_complexity: Required task complexity (0.0-1.0)

        Returns:
            List of models ranked by complexity suitability (best first)
        """
        if not models:
            return models

        # Filter out basic models for complex tasks
        filtered_models = []
        for model in models:
            model_complexity = self._calculate_complexity_score(model, models)

            # For complex tasks (complexity > 0.7), exclude very basic models
            if task_complexity > 0.7:
                # Exclude models with very low context or very low complexity
                if (model.max_context_tokens or 0) < 8000 or model_complexity < 0.2:
                    continue

            # For moderate tasks (complexity > 0.5), exclude extremely basic models
            elif task_complexity > 0.5:
                # Exclude models with very low context
                if (model.max_context_tokens or 0) < 4000:
                    continue

            filtered_models.append(model)

        # If filtering removed all models, use original list
        if not filtered_models:
            filtered_models = models

        def calculate_complexity_score(model: ModelCapability) -> float:
            # Calculate how well this model's complexity aligns with task requirements
            model_complexity = self._calculate_complexity_score(model, filtered_models)
            complexity_diff = abs(task_complexity - model_complexity)
            alignment_score = 1.0 - complexity_diff

            # Boost score if model can handle the required complexity
            if model_complexity >= task_complexity:
                capability_bonus = 0.2 * (1.0 - complexity_diff)
                alignment_score += capability_bonus

            return alignment_score

        # Calculate complexity scores and sort
        scored_models = [
            (calculate_complexity_score(model), model) for model in filtered_models
        ]

        # Sort by complexity score (highest first), then by model name for stability
        return [
            model
            for _, model in sorted(
                scored_models, key=lambda x: (x[0], x[1].model_name), reverse=True
            )
        ]

    def _apply_cost_bias(
        self, models: list[ModelCapability], cost_bias: float
    ) -> list[ModelCapability]:
        """Apply cost bias to rerank models based on cost vs capability preferences.

        This method takes models that have already been ranked by complexity suitability
        and reranks them based on cost optimization preferences.

        Args:
            models: List of models already ranked by complexity
            cost_bias: Cost vs capability preference (0.0-1.0)
                      0.0 = prefer cheapest models
                      1.0 = prefer most capable models

        Returns:
            List of models reranked by cost bias (best first)
        """
        if not models:
            return models

        # Pre-calculate max values for normalization
        total_costs = [
            (m.cost_per_1m_input_tokens or 0) + (m.cost_per_1m_output_tokens or 0)
            for m in models
        ]
        max_cost = max(total_costs) if total_costs else 0
        max_capability = max((m.max_context_tokens or 0) for m in models)

        def calculate_cost_bias_score(model: ModelCapability) -> float:
            # Cost and capability scores
            cost = (model.cost_per_1m_input_tokens or 0) + (
                model.cost_per_1m_output_tokens or 0
            )
            capability = model.max_context_tokens or 0

            cost_score = 1 - (cost / max_cost if max_cost > 0 else 0)
            capability_score = capability / max_capability if max_capability > 0 else 0

            # EXTREME COST BIAS OVERRIDE: For very low/high cost_bias, prioritize cost over complexity
            if cost_bias <= 0.1:
                # Ultra-low cost bias: prioritize cheapest models (100% cost, 0% capability)
                return cost_score
            elif cost_bias >= 0.9:
                # Ultra-high cost bias: prioritize most expensive models
                # For cost_bias >= 0.9, directly use the normalized cost as the score
                # Higher cost = higher score
                return cost / max_cost if max_cost > 0 else 0.5
            else:
                # Standard balanced routing for moderate cost_bias values
                cost_capability_score = (
                    1 - cost_bias
                ) * cost_score + cost_bias * capability_score

                # For moderate cost_bias, we combine cost and capability in a balanced way
                return cost_capability_score

        # Calculate cost bias scores and sort
        scored_models = [(calculate_cost_bias_score(model), model) for model in models]

        # Sort by cost bias score (highest first), then by model name for stability
        return [
            model
            for _, model in sorted(
                scored_models, key=lambda x: (x[0], x[1].model_name), reverse=True
            )
        ]

    def _apply_integrated_routing(
        self, models: list[ModelCapability], task_complexity: float, cost_bias: float
    ) -> list[ModelCapability]:
        """Apply intelligent routing that adapts based on cost bias extremes.

        For extreme cost bias values (≤0.1 or ≥0.9), cost preference dominates.
        For moderate cost bias values, complexity analysis is primary with cost as secondary.

        Args:
            models: List of available models to rank
            task_complexity: Required task complexity (0.0-1.0)
            cost_bias: Cost vs capability preference (0.0-1.0)

        Returns:
            List of models ranked by the integrated approach (best first)
        """
        if not models:
            return models

        # For extreme cost bias, prioritize cost over complexity
        if cost_bias <= 0.1 or cost_bias >= 0.9:
            # For extreme cost bias, only apply cost bias ranking without complexity filtering
            # This ensures the cost preference is respected
            return self._apply_cost_bias(models, cost_bias)
        else:
            # For moderate cost bias, use traditional complexity-first approach
            # Step 1: Apply complexity-based routing
            complexity_ranked = self._apply_complexity_routing(models, task_complexity)

            # Step 2: Apply cost bias to rerank
            final_ranked = self._apply_cost_bias(complexity_ranked, cost_bias)

            return final_ranked
