"""Model routing service using UniRouter cluster-based intelligent selection.

This service provides intelligent model routing based on cluster-specific error rates
and cost optimization using the UniRouter algorithm.
"""

from __future__ import annotations

import logging
from typing import Optional

from adaptive_router.models.llm_core_models import (
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.services.unirouter_service import UniRouterService

logger = logging.getLogger(__name__)


class ModelRouter:
    """Intelligent model routing using UniRouter cluster-based selection.

    Selects optimal LLM models based on cluster-specific error rates,
    cost optimization, and model capability matching.
    """

    def __init__(
        self,
        unirouter_service: Optional[UniRouterService] = None,
        use_modal_gpu: bool = False,
    ) -> None:
        """Initialize router with UniRouter service.

        Args:
            unirouter_service: Optional UniRouterService instance. If not provided, creates one internally.
            use_modal_gpu: If True, attempt to use Modal GPU for feature extraction (default: False).
        """
        if unirouter_service is None:
            unirouter_service = UniRouterService(use_modal_gpu=use_modal_gpu)

        self._unirouter = unirouter_service
        logger.info("ModelRouter initialized with UniRouter (cluster-based routing)")

    def select_model(self, request: ModelSelectionRequest) -> ModelSelectionResponse:
        """Select optimal model based on prompt analysis using UniRouter.

        This is the main public API method. It delegates to UniRouter for
        cluster-based routing decisions.

        Args:
            request: ModelSelectionRequest with prompt and optional models/cost_bias

        Returns:
            ModelSelectionResponse with selected provider, model, and alternatives

        Raises:
            ValueError: If no eligible models found or validation fails
            RuntimeError: If classification or routing fails
        """
        return self._unirouter.select_model(request)
