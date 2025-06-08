import litserve as ls  # type: ignore
from pydantic import BaseModel, ValidationError
from typing import Any, Dict


class PromptRequest(BaseModel):
    prompt: str


class AdaptiveModelSelectionAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        from services.model_selector import ModelSelector
        from services.prompt_classifier import get_prompt_classifier

        self.model_selector = ModelSelector(get_prompt_classifier())

    def decode_request(self, request: Dict[str, Any]) -> str:
        try:
            req = PromptRequest.model_validate(request)
            return req.prompt
        except ValidationError as e:
            raise ValueError(f"Invalid request: {e}") from e

    def predict(self, prompt: str) -> Dict[str, Any]:
        return self.model_selector.select_model(prompt)

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return output


if __name__ == "__main__":
    api = AdaptiveModelSelectionAPI()
    server = ls.LitServer(api, accelerator="auto", devices="auto")
    server.run(port=8000)
