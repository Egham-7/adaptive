from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel

# Represents a chat message role
MessageRole = str  # "user" | "assistant" | "system" | "tool"

# Represents a single chat message
class Message(BaseModel):
    role: MessageRole
    content: str
    tool_call_id: Optional[str] = None
    function_call: Optional[Dict[str, str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    messages: List[Message]


class OpenAIChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str  # "stop" | "length" | "function_call" | "content_filter" | "tool_calls"

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage

class AnthropicContentItem(BaseModel):
    type: str
    text: str

class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int

class AnthropicResponse(BaseModel):
    id: str
    type: str
    role: str
    content: List[AnthropicContentItem]
    model: str
    stop_reason: str
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage

class GroqChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class GroqUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class GroqResponse(BaseModel):
    id: str
    model: str
    created: int
    choices: List[GroqChoice]
    usage: GroqUsage

class DeepSeekLogProbs(BaseModel):
    token_logprobs: List[float]
    tokens: List[str]
    top_logprobs: List[Dict[str, float]]

class DeepSeekChoice(BaseModel):
    index: int
    message: Message
    logprobs: Optional[DeepSeekLogProbs] = None
    finish_reason: str  # "stop" | "length" | "tool_calls" | "content_filter"

class DeepSeekUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class DeepSeekResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[DeepSeekChoice]
    usage: DeepSeekUsage
ProviderResponse = Union[OpenAIResponse, AnthropicResponse, GroqResponse, DeepSeekResponse]

class ChatCompletionResponse(BaseModel):
    provider: str
    response: ProviderResponse
    error: Optional[str] = None



class BaseStreamingResponse(BaseModel):
    id: str
    model: str
    provider: str
    choices: List[Dict[str, Any]]

class OpenAIStreamingResponse(BaseStreamingResponse):
    object: str
    created: int
    choices: List[Dict[str, Any]]

class GroqStreamingResponse(BaseStreamingResponse):
    object: str
    created: int
    choices: List[Dict[str, Any]]

class DeepSeekStreamingResponse(BaseStreamingResponse):
    created: int
    choices: List[Dict[str, Any]]

StreamingResponse = Union[OpenAIStreamingResponse, GroqStreamingResponse, DeepSeekStreamingResponse]



def extract_content_from_streaming_response(chunk: StreamingResponse) -> str:
    if isinstance(chunk, OpenAIStreamingResponse) or isinstance(chunk, GroqStreamingResponse):
        return chunk.choices[0].get("delta", {}).get("content", "")
    elif isinstance(chunk, DeepSeekStreamingResponse):
        choice = chunk.choices[0]
        return choice.get("text", "") or choice.get("message", {}).get("content", "")
    return ""
