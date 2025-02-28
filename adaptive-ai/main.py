from fastapi import FastAPI
from ai.prompt_classifier import PromptClassifier
from ai.domain_classifier import DomainClassifier
from ai.adaptive_bot import ChatBot
from pydantic import BaseModel

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
    domain = domain_classifier.classify(request.prompt)

    bot = ChatBot(domain=domain[0], prompt_complexity_score=complexity)
    result = bot.send(request.prompt)

    return result
