import litserve as ls  # type: ignore
from typing import List

from adaptive_ai.core.config import get_settings
from adaptive_ai.services.model_selector import get_model_selector
from adaptive_ai.models.requests import PromptRequest, ModelSelectionResponse
from adaptive_ai.services.prompt_classifier import get_prompt_classifier


class AdaptiveModelSelectionAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        self.settings = get_settings()
        self.model_selector = get_model_selector(get_prompt_classifier())

    def decode_request(self, request: PromptRequest) -> str:
        return request.prompt

    def predict(self, prompts: List[str]) -> List[ModelSelectionResponse]:
        return self.model_selector.select_model(prompts)

    def encode_response(self, output: ModelSelectionResponse) -> ModelSelectionResponse:
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
        generate_client_file=False,
        host=settings.server.host,
        port=settings.server.port
    )


if __name__ == "__main__":
    main()
