from fastapi import FastAPI
from prompt_classifier import PromptClassifier
from domain_classifier import DomainClassifier
from pydantic import BaseModel
from llms import domain_model_mapping, model_capabilities
from parameters import adjust_parameters
from abc import ABC, abstractmethod
from dataclasses import dataclass

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


prompt_classifier = PromptClassifier()
domain_classifier = DomainClassifier()


class LLMProvider(ABC):
    @abstractmethod
    def get_parameters(self) -> dict:
        pass

@dataclass
class OpenGroqAILLMProvider(LLMProvider):
    model: str
    temperature: float
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    max_tokens: int
    n: int

    def __post_init__(self):
        self.temperature = round(self.temperature, 2)
        self.top_p = round(self.top_p, 2)
        self.presence_penalty = round(self.presence_penalty, 2)
        self.frequency_penalty = round(self.frequency_penalty, 2)
        self.max_tokens = int(self.max_tokens)

    def get_parameters(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_tokens": self.max_tokens,
            "n": self.n
        }

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

    suitable_models = domain_model_mapping[domain]

    for model_name in suitable_models:
        complexity_range = model_capabilities[model_name]["complexity_range"]
        provider = model_capabilities[model_name]["provider"]
        if complexity_range[0] <= complexity_score <= complexity_range[1]:
            selected_model = {"selected_model": model_name, "provider": provider}
            break
    else:
        selected_model = {"selected_model": suitable_models[0], "provider": model_capabilities[suitable_models[0]]["provider"]}

    prompt_scores = {
        "creativity_scope": complexity["creativity_scope"],
        "reasoning": complexity["reasoning"],
        "contextual_knowledge": complexity["contextual_knowledge"],
        "prompt_complexity_score": complexity["prompt_complexity_score"],
        "domain_knowledge": complexity["domain_knowledge"]
    }
    
    parameters = adjust_parameters(domain, prompt_scores)
    provider = selected_model["provider"]

    if provider in ["OpenAI", "GROQ"]:
        llm_provider = OpenGroqAILLMProvider(
            model=selected_model["selected_model"],
            **parameters
        )
        parameters_model = llm_provider.get_parameters()
    else:
        parameters_model = {}

    return {
        "parameters": parameters_model,
        "provider": provider
    }