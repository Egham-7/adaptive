from fastapi import APIRouter, Depends, HTTPException
from models.domain import PromptRequest
from services.model_selector import ModelSelector
from services.prompt_classifier import get_prompt_classifier
from services.domain_classifier import get_domain_classifier
from prometheus_client import Counter, Histogram, Summary
import time

router = APIRouter()

# Define Prometheus metrics
MODEL_SELECTION_COUNTER = Counter(
    "model_selection_total", 
    "Total number of model selection requests",
    ["model_type", "status"]
)

MODEL_SELECTION_LATENCY = Histogram(
    "model_selection_latency_seconds",
    "Latency of model selection requests",
    ["endpoint"]
)

MODEL_PARAMETERS_SUMMARY = Summary(
    "model_parameters_processing_seconds",
    "Time spent processing model parameters"
)


def get_model_selection_service(
    prompt_classifier=Depends(get_prompt_classifier),
    domain_classifier=Depends(get_domain_classifier),
):
    return ModelSelector(prompt_classifier, domain_classifier)


@router.post("/select-model")
@MODEL_SELECTION_LATENCY.labels(endpoint="select-model").time()
async def chat_bot(
    request: PromptRequest,
    model_selection_service: ModelSelector = Depends(get_model_selection_service),
):
    start_time = time.time()
    try:
        result = model_selection_service.select_model(request.prompt)
        # Record successful request
        MODEL_SELECTION_COUNTER.labels(
            model_type=result.get("model_type", "unknown"),
            status="success"
        ).inc()
        return result
    except ValueError as e:
        # Record failed request
        MODEL_SELECTION_COUNTER.labels(
            model_type="unknown",
            status="error"
        ).inc()
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/parameters")
async def model_parameters(
    request: PromptRequest,
    model_selection_service: ModelSelector = Depends(get_model_selection_service),
):
    # Use Summary metric to time the processing
    with MODEL_PARAMETERS_SUMMARY.time():
        try:
            result = model_selection_service.get_model_parameters(request.prompt)
            # Record successful request
            MODEL_SELECTION_COUNTER.labels(
                model_type=result.get("model_type", "unknown"),
                status="success"
            ).inc()
            return result
        except ValueError as e:
            # Record failed request
            MODEL_SELECTION_COUNTER.labels(
                model_type="unknown",
                status="error"
            ).inc()
            raise HTTPException(status_code=400, detail=str(e))
