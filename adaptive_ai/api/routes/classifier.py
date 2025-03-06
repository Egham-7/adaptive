from fastapi import APIRouter, Depends
from models.domain import PromptRequest
from services.prompt_classifier import get_prompt_classifier
from services.domain_classifier import get_domain_classifier

router = APIRouter()


@router.post("/classify/prompt")
async def classify_prompt(
    request: PromptRequest, prompt_classifier=Depends(get_prompt_classifier)
):
    result = prompt_classifier.classify_prompt(request.prompt)
    return result


@router.post("/classify/domain")
async def classify_domain(
    request: PromptRequest, domain_classifier=Depends(get_domain_classifier)
):
    result = domain_classifier.classify(request.prompt)
    return result
