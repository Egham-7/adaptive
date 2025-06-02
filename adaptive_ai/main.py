import litserve as ls  # type: ignore[import-untyped]
from pydantic import BaseModel, ValidationError, Field
from typing import Optional, Dict, Any, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptRequest(BaseModel):
    prompt: str = Field(
        ..., min_length=1, max_length=4096, description="The input prompt to analyze"
    )
    domain: str = Field(
        default="Computers_and_Electronics",
        description="The domain context for the prompt",
    )


class ModelSelectionResponse(BaseModel):
    selected_model: str
    provider: str
    match_score: float
    task_type: str
    difficulty: str
    prompt_scores: Dict[str, Any]
    complexity_score: float
    thresholds: Dict[str, float]


class ErrorResponse(BaseModel):
    error: str
    details: Optional[Dict[str, Any]] = None


class AdaptiveModelSelectionAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        from services.model_selector import ModelSelector
        from services.prompt_classifier import get_prompt_classifier

        self.model_selector = ModelSelector(get_prompt_classifier())
        logger.info("API initialized successfully")

    def decode_request(self, request: Dict[str, Any]) -> PromptRequest:
        try:
            req = PromptRequest.model_validate(request)
            return req
        except ValidationError as e:
            logger.error(f"Invalid request: {e}")
            raise ValueError(f"Invalid request: {e}")

    def predict(self, request: PromptRequest) -> Dict[str, Any]:
        try:
            logger.info(f"Processing prompt for domain: {request.domain}")
            return self.model_selector.select_model(request.prompt, request.domain)
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            raise

    def encode_response(self, output: Dict[str, Any]) -> Union[ModelSelectionResponse, ErrorResponse]:
        try:
            return ModelSelectionResponse.model_validate(output)
        except ValidationError as e:
            logger.error(f"Error encoding response: {e}")
            return ErrorResponse(
                error="Failed to encode response", details={"validation_error": str(e)}
            )


if __name__ == "__main__":
    api = AdaptiveModelSelectionAPI()
    server = ls.LitServer(api, accelerator="auto", devices="auto")
    logger.info("Starting server on port 8000")
    server.run(port=8000)
