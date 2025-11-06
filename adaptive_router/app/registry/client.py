from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from adaptive_router.models.registry import (
    RegistryClientConfig,
    RegistryConnectionError,
    RegistryModel,
    RegistryResponseError,
)

logger = logging.getLogger(__name__)


class RegistryClient:
    """HTTP client for the Adaptive model registry service.

    Uses dependency injection for httpx.Client to enable connection pooling
    and better testability.
    """

    def __init__(self, config: RegistryClientConfig, client: httpx.Client) -> None:
        """Initialize registry client.

        Args:
            config: Registry client configuration
            client: HTTP client for making requests (injected dependency)
        """
        self._config = config
        self._client = client

    def health_check(self) -> None:
        """Raise if the registry health check fails."""
        self._request("GET", "/healthz")

    def list_models(
        self,
        *,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> List[RegistryModel]:
        """List models from registry with optional filtering.

        Args:
            provider: Filter by provider name
            model_name: Filter by model name

        Returns:
            List of registry models matching filters

        Raises:
            RegistryConnectionError: If registry cannot be reached
            RegistryResponseError: If registry returns invalid response
        """
        params = {
            "provider": provider,
            "model_name": model_name,
        }
        resp = self._request("GET", "/models", params=params)
        payload = resp.json()

        if not isinstance(payload, list):
            raise RegistryResponseError(
                f"unexpected response payload type: {type(payload).__name__}"
            )

        models: List[RegistryModel] = []
        for item in payload:
            try:
                models.append(RegistryModel.model_validate(item))
            except (ValueError, TypeError) as err:
                logger.warning("invalid model entry from registry: %s", err)
        return models

    def get_by_provider_and_name(
        self, provider: str, name: str
    ) -> RegistryModel | None:
        """Get model by provider and model name.

        Args:
            provider: Provider name
            name: Model name

        Returns:
            RegistryModel if found, None otherwise

        Raises:
            ValueError: If provider or name is empty
            RegistryConnectionError: If registry cannot be reached
            RegistryResponseError: If registry returns invalid response
        """
        # Strip whitespace from inputs
        provider = provider.strip() if provider else ""
        name = name.strip() if name else ""

        # Validate after stripping
        if not provider:
            raise ValueError("provider must be provided")
        if not name:
            raise ValueError("name must be provided")

        return self._get_model(f"/models/{provider}/{name}")

    def _get_model(self, path: str) -> RegistryModel | None:
        """Get a single model from the registry.

        Args:
            path: API path to fetch

        Returns:
            RegistryModel if found, None if not found

        Raises:
            RegistryConnectionError: If registry cannot be reached
            RegistryResponseError: If registry returns invalid response
        """
        resp = self._request("GET", path)
        if resp.status_code == httpx.codes.NOT_FOUND:
            return None

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RegistryResponseError(str(exc)) from exc

        return RegistryModel.model_validate(resp.json())

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Make HTTP request to registry.

        Args:
            method: HTTP method
            path: API path
            params: Optional query parameters

        Returns:
            HTTP response

        Raises:
            RegistryConnectionError: If request fails or times out
            RegistryResponseError: If registry returns error status
        """
        url = self._build_url(path)
        headers = self._config.normalized_headers()

        try:
            response = self._client.request(
                method,
                url,
                params=params,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise RegistryConnectionError("registry request timed out") from exc
        except httpx.RequestError as exc:
            raise RegistryConnectionError(f"registry request failed: {exc!s}") from exc

        if response.status_code >= 400:
            if response.status_code == httpx.codes.NOT_FOUND:
                return response
            raise RegistryResponseError(
                f"registry returned {response.status_code}: {response.text}"
            )

        return response

    def _build_url(self, path: str) -> str:
        base = self._config.base_url.rstrip("/")
        suffix = path if path.startswith("/") else f"/{path}"
        return f"{base}{suffix}"
