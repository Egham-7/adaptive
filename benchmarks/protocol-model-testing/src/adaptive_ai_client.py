"""Client for making requests to the adaptive_ai service."""

import time
from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass
class ModelSelectionRequest:
    """Request model for adaptive_ai service."""

    prompt: str
    provider_preferences: Optional[list[str]] = None
    cost_threshold: Optional[float] = None
    performance_threshold: Optional[float] = None


@dataclass
class AdaptiveAIResponse:
    """Response from adaptive_ai service."""

    selected_model: str
    protocol: str
    classification_result: dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class AdaptiveAIClient:
    """Client for testing the adaptive_ai service."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Initialize the client.

        Args:
            base_url: Base URL of the adaptive_ai service
        """
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Check if the service is running.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def make_request(self, prompt: str, **kwargs: Any) -> AdaptiveAIResponse:
        """Make a request to the adaptive_ai service.

        Args:
            prompt: The prompt to send
            **kwargs: Additional parameters for the request

        Returns:
            Response from the service
        """
        request_data = {
            "prompt": prompt,
            "provider_preferences": kwargs.get("provider_preferences"),
            "cost_threshold": kwargs.get("cost_threshold"),
            "performance_threshold": kwargs.get("performance_threshold"),
        }

        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}

        start_time = time.time()

        try:
            response = self.session.post(
                f"{self.base_url}/predict", json=request_data, timeout=30
            )

            execution_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                return AdaptiveAIResponse(
                    selected_model=result.get("selected_model", "unknown"),
                    protocol=result.get("protocol", "unknown"),
                    classification_result=result.get("classification_result", {}),
                    execution_time=execution_time,
                    success=True,
                )
            else:
                return AdaptiveAIResponse(
                    selected_model="unknown",
                    protocol="unknown",
                    classification_result={},
                    execution_time=execution_time,
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.text}",
                )

        except Exception as e:
            execution_time = time.time() - start_time
            return AdaptiveAIResponse(
                selected_model="unknown",
                protocol="unknown",
                classification_result={},
                execution_time=execution_time,
                success=False,
                error_message=str(e),
            )

    def batch_request(
        self, prompts: list[str], **kwargs: Any
    ) -> list[AdaptiveAIResponse]:
        """Make batch requests to the adaptive_ai service.

        Args:
            prompts: List of prompts to send
            **kwargs: Additional parameters for each request

        Returns:
            List of responses
        """
        responses = []

        for i, prompt in enumerate(prompts):
            response = self.make_request(prompt, **kwargs)
            responses.append(response)

            # Progress logging
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(prompts)} requests")

        return responses

    def get_service_info(self) -> dict[str, Any]:
        """Get information about the running service.

        Returns:
            Service information
        """
        try:
            response = self.session.get(f"{self.base_url}/info", timeout=5)
            if response.status_code == 200:
                return response.json()  # type: ignore[no-any-return]
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def test_connection(self) -> dict[str, Any]:
        """Test the connection to the service.

        Returns:
            Connection test results
        """
        results = {
            "health_check": self.health_check(),
            "service_info": self.get_service_info(),
            "base_url": self.base_url,
        }

        # Test a simple request
        try:
            test_response = self.make_request("Hello, world!")
            results["test_request"] = {
                "success": test_response.success,
                "execution_time": test_response.execution_time,
                "selected_model": test_response.selected_model,
                "protocol": test_response.protocol,
                "error": test_response.error_message,
            }
        except Exception as e:
            results["test_request"] = {"success": False, "error": str(e)}

        return results
