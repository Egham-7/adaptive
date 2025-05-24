import litserve as ls  # type: ignore
from pydantic import BaseModel, ValidationError

class PromptRequest(BaseModel):
    prompt: str

class AdaptiveModelSelectionAPI(ls.LitAPI):
    def setup(self, device):
        from services.model_selector import ModelSelector
        from services.prompt_classifier import get_prompt_classifier
        from services.domain_classifier import get_domain_classifier

        self.model_selector = ModelSelector(
            get_prompt_classifier(), get_domain_classifier()
        )

    def decode_request(self, request : PromptRequest):
        # Use Pydantic to validate input
        try:
            req = PromptRequest.model_validate(request)
            return req.prompt
        except ValidationError as e:
            # You can customize this error as you wish
            raise ValueError(f"Invalid request: {e}")

    def predict(self, prompt):
    
        return self.model_selector.select_model(prompt)

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    api = AdaptiveModelSelectionAPI()
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8000)
