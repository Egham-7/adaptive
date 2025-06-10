import litserve as ls  # type: ignore
from typing import List

from core.config import get_settings
from services.model_selector import get_model_selector
from models.requests import PromptRequest, ModelSelectionResponse
from services.prompt_classifier import get_prompt_classifier


class AdaptiveModelSelectionAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        self.settings = get_settings()
        self.model_selector = get_model_selector(get_prompt_classifier())

    def decode_request(self, request: PromptRequest) -> str:
        return request.prompt

    def predict(self, prompts: List[str]) -> ModelSelectionResponse:
        return self.model_selector.select_model(prompts)

    def encode_response(self, output: ModelSelectionResponse) -> ModelSelectionResponse:
        return output


def create_app() -> ls.LitServer:
    """Factory function to create the LitServer app."""
    settings = get_settings()
    api = AdaptiveModelSelectionAPI()

    return ls.LitServer(
        api,
        accelerator=settings.accelerator,
        devices=settings.devices,
        max_batch_size=settings.max_batch_size,
        batch_timeout=settings.batch_timeout,
    )


if __name__ == "__main__":
    app = create_app()
    app.run(port=8000)
