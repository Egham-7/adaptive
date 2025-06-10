import litserve as ls  # type: ignore
from pydantic import BaseModel
from typing import Any, Dict, List


class PromptRequest(BaseModel):
    prompt: List[str]


class AdaptiveModelSelectionAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        from services.model_selector import ModelSelector
        from services.prompt_classifier import get_prompt_classifier

        self.model_selector = ModelSelector(get_prompt_classifier())

    def decode_request(self, request: PromptRequest) -> List[str]:
        return request.prompt

    def predict(self, prompt: List[str]) -> List[Dict[str, Any]]:
        # Flatten the list if it's double-wrapped
        if len(prompt) == 1 and isinstance(prompt[0], list):
            prompt = prompt[0]
        return self.model_selector.select_model(prompt)

    def encode_response(self, output: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return output


if __name__ == "__main__":
    api = AdaptiveModelSelectionAPI()
    server = ls.LitServer(api, accelerator="auto", devices="auto")
    server.run(port=8000)
