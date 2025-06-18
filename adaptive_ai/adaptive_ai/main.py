import litserve as ls

from adaptive_ai.core.config import get_settings
from adaptive_ai.models.orchestrator import OrchestratorResponse
from adaptive_ai.models.requests import PromptRequest
from adaptive_ai.services.model_selector import get_model_selector
from adaptive_ai.services.prompt_classifier import get_prompt_classifier


class AdaptiveModelSelectionAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        self.settings = get_settings()
        self.model_selector = get_model_selector(get_prompt_classifier())

    def decode_request(self, request: PromptRequest) -> PromptRequest:
        return request

    def predict(self, requests: list[PromptRequest]) -> list[OrchestratorResponse]:
        prompts = [req.prompt for req in requests]
        responses = []
        for prompt in prompts:
            response = self.model_selector.select_orchestrator_route(prompt)
            responses.append(response)
        return responses

    def encode_response(self, output: OrchestratorResponse) -> OrchestratorResponse:
        return output


def create_app() -> ls.LitServer:
    """Factory function to create the LitServer app."""
    settings = get_settings()
    api = AdaptiveModelSelectionAPI(
        max_batch_size=settings.litserve.max_batch_size,
        batch_timeout=settings.litserve.batch_timeout,
    )

    return ls.LitServer(
        api,
        accelerator=settings.litserve.accelerator,
        devices=settings.litserve.devices,
    )


def main() -> None:
    """Main entry point for the adaptive-ai CLI command."""
    settings = get_settings()
    app = create_app()
    app.run(
        generate_client_file=False, host=settings.server.host, port=settings.server.port
    )


if __name__ == "__main__":
    main()
