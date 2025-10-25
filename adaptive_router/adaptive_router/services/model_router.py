"""Model routing service using cluster-based intelligent selection.

This service provides intelligent model routing based on cluster-specific error rates
and cost optimization.
"""

from __future__ import annotations

import logging
from typing import Optional

from adaptive_router.models.llm_core_models import (
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.services.router_service import RouterService

logger = logging.getLogger(__name__)


class ModelRouter:
    """Intelligent model routing using cluster-based selection.

    Selects optimal LLM models based on cluster-specific error rates,
    cost optimization, and model capability matching.
    """

    def __init__(
        self,
        router_service: Optional[RouterService] = None,
    ) -> None:
        """Initialize router with RouterService.

        Args:
            router_service: Optional RouterService instance. If not provided, creates one internally.
        """
        if router_service is None:
            router_service = RouterService()

        self._router = router_service
        logger.info("ModelRouter initialized with Router (cluster-based routing)")

    def select_model(self, request: ModelSelectionRequest) -> ModelSelectionResponse:
        """Select optimal model based on prompt analysis using Router.

        This is the main public API method. It delegates to Router for
        cluster-based routing decisions.

        Args:
            request: ModelSelectionRequest with prompt and optional models/cost_bias

        Returns:
            ModelSelectionResponse with selected provider, model, and alternatives

        Raises:
            ValueError: If no eligible models found or validation fails
            RuntimeError: If classification or routing fails
        """
        return self._router.select_model(request)
