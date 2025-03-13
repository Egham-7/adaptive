from fastapi import APIRouter, Depends, HTTPException
from models.domain import PromptRequest
from services.model_selector import ModelSelector
from services.prompt_classifier import get_prompt_classifier
from services.domain_classifier import get_domain_classifier

router = APIRouter()


def get_model_selection_service(
    prompt_classifier=Depends(get_prompt_classifier),
    domain_classifier=Depends(get_domain_classifier),
):
    return ModelSelector(prompt_classifier, domain_classifier)


@router.post("/select-model")
async def chat_bot(
    request: PromptRequest,
    model_selection_service: ModelSelector = Depends(get_model_selection_service),
):
    try:
        return model_selection_service.select_model(request.prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/parameters")
async def model_parameters(
    request: PromptRequest,
    model_selection_service: ModelSelector = Depends(get_model_selection_service),
):
    try:
        return model_selection_service.get_model_parameters(request.prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
