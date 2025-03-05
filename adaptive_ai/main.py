from fastapi import FastAPI
from prompt_classifier import PromptClassifier
from domain_classifier import DomainClassifier
from pydantic import BaseModel
from llms import domain_model_mapping, model_capabilities
from parameters import adjust_parameters

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
async def classify_domain(request: PromptRequest):
    result = domain_classifier.classify(request.prompt)
    return result


@app.post("/select-model")
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
        provider = model_capabilities[model_name]["provider"]
        if complexity_range[0] <= complexity <= complexity_range[1]:
            return {"selected_model": model_name, "provider": provider}

    # If no model matches the complexity score, return a default model
    return {
        "selected_model": suitable_models[0],
        "provider": model_capabilities[suitable_models[0]]["provider"],
    }

@app.post("/parameters")
async def chat_bot(request: PromptRequest):
    complexity = prompt_classifier.classify_prompt(request.prompt)
    
    complexity_score = complexity["prompt_complexity_score"][0]
    domain = domain_classifier.classify(request.prompt)[0]

    if domain not in domain_model_mapping:
        raise ValueError(f"Domain '{domain}' is not recognized.")

    # Filter models suitable for the given domain
    suitable_models = domain_model_mapping[domain]

    # Find a model within the suitable models that matches the complexity score
    for model_name in suitable_models:
        complexity_range = model_capabilities[model_name]["complexity_range"]
        provider = model_capabilities[model_name]["provider"]
        if complexity_range[0] <= complexity_score <= complexity_range[1]:
            return {"selected_model": model_name, "provider": provider}

    # Find best parameters combinations
    prompt_scores = {
    "creativity_scope": complexity["creativity_scope"],
    "reasoning": complexity["reasoning"],
    "contextual_knowledge": complexity["contextual_knowledge"],
    "prompt_complexity_score": complexity["prompt_complexity_score"],
    "domain_knowledge": complexity["domain_knowledge"]
}
    parameters = adjust_parameters(domain, prompt_scores)
    """
    return {
        "Temperature": round(temperature, 2),
        "TopP": round(top_p, 2),
        "PresencePenalty": round(presence_penalty, 2),
        "FrequencyPenalty": round(frequency_penalty, 2),
        "MaxCompletionTokens": int(max_tokens),
        "N": n
    }
    """
    print(parameters)
    # If no model matches the complexity score, return a default model
    return {
        "selected_model": suitable_models[0],
        "provider": model_capabilities[suitable_models[0]]["provider"],
        "parameters": parameters
    }
