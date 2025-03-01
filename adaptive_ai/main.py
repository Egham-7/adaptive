from fastapi import FastAPI
from prompt_classifier import PromptClassifier
from domain_classifier import DomainClassifier
from pydantic import BaseModel
from llms import domain_model_mapping, model_capabilities

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


prompt_classifier = PromptClassifier()
domain_classifier = DomainClassifier()

class PromptRequest(BaseModel):
    prompt: str


@app.post("/classify/prompt")
async def classify_prompt(request: PromptRequest):
    result = prompt_classifier.classify_prompt(request.prompt)
    return result


@app.post("/classify/domain")
async def classify_prompt(request: PromptRequest):
    result = domain_classifier.classify(request.prompt)
    return result



@app.post("/bot")
async def chat_bot(request: PromptRequest):
    complexity = prompt_classifier.classify_prompt(request.prompt)
    complexity = complexity["prompt_complexity_score"][0]
    domain = domain_classifier.classify(request.prompt)[0]
    
    if domain not in domain_model_mapping:
        raise ValueError(f"Domain '{domain}' is not recognized.")
    
    # Filter models suitable for the given domain
    suitable_models = domain_model_mapping[domain]
    
    # Find a model within the suitable models that matches the complexity score
    for model_name in suitable_models:
        complexity_range = model_capabilities[model_name]["complexity_range"]
        if complexity_range[0] <= complexity <= complexity_range[1]:
            return {"selected_model": model_name}
    
    # If no model matches the complexity score, return a default model
    return {"selected_model": suitable_models[0]}

