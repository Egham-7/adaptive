from typing import Any

import litserve as ls

from core.config import get_settings
from models.requests import PromptRequest
from services.prompt_classifier import get_prompt_classifier


class PromptClassificationAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        self.settings = get_settings()
        self.prompt_classifier = get_prompt_classifier()

    def decode_request(self, request: PromptRequest) -> PromptRequest:
        return request

    def predict(self, requests: list[PromptRequest]) -> list[dict[str, Any]]:
        prompts = [req.prompt for req in requests]
        classification_results = self.prompt_classifier.classify_prompts(prompts)
        return classification_results

    def encode_response(self, output: dict[str, Any]) -> dict[str, Any]:
        return output


def create_app() -> ls.LitServer:
    """Factory function to create the LitServer app."""
    settings = get_settings()
    api = PromptClassificationAPI(
        max_batch_size=settings.litserve.max_batch_size,
        batch_timeout=settings.litserve.batch_timeout,
    )

    return ls.LitServer(
        api,
        accelerator=settings.litserve.accelerator,
        devices=settings.litserve.devices,
    )


def main() -> None:
    """Main entry point for the prompt classification service."""
    settings = get_settings()
    app = create_app()
    app.run(
        generate_client_file=False, host=settings.server.host, port=settings.server.port
    )


if __name__ == "__main__":
    main()
