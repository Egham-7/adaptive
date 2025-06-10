import litserve as ls  # type: ignore
from pydantic import BaseModel
from typing import Any, Dict, List


class PromptRequest(BaseModel):
    prompt: str


class AdaptiveModelSelectionAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        from services.model_selector import ModelSelector
        from services.prompt_classifier import get_prompt_classifier

        self.model_selector = ModelSelector(get_prompt_classifier())

    def decode_request(self, request: PromptRequest) -> str:
        return request.prompt

    def predict(self, prompt: List[str]) -> Dict[str, Any]:
        return self.model_selector.select_model(prompt)

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return output


if __name__ == "__main__":
    api = AdaptiveModelSelectionAPI()
    server = ls.LitServer(
        api, accelerator="auto", devices="auto", max_batch_size=8, batch_timeout=0.05
    )
    server.run(port=8000)
