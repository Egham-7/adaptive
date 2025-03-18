from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field
import json


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


# Base Streaming Response
class BaseStreamingResponse(BaseModel):
    id: str
    model: str
    provider: str
    choices: List[Dict[str, Any]]
    
    def extract_content(self) -> str:
        """Extracts the content from a streaming response."""
        if not self.choices or len(self.choices) == 0:
            return ""
        
        # Get the first choice
        choice = self.choices[0]
        
        # Different providers might structure their response differently
        if "delta" in choice and "content" in choice["delta"]:
            return choice["delta"]["content"] or ""
        elif "text" in choice:
            return choice["text"] or ""
        elif "content" in choice:
            return choice["content"] or ""
        elif "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"] or ""
        
        # If we can't find content in any of the expected places, return empty string
        return ""


# OpenAI Streaming Response
class OpenAIStreamingResponse(BaseStreamingResponse):
    object: str = "chat.completion.chunk"
    created: int
    
    def extract_content(self) -> str:
        return self.choices[0].get("delta", {}).get("content", "")


# Groq Streaming Response
class GroqStreamingResponse(BaseStreamingResponse):
    object: str = "chat.completion.chunk"
    created: int
    
    def extract_content(self) -> str:
        return self.choices[0].get("delta", {}).get("content", "")


# DeepSeek Streaming Response
class DeepSeekStreamingResponse(BaseStreamingResponse):
    created: int
    
    def extract_content(self) -> str:
        choice = self.choices[0]
        return choice.get("text", "") or choice.get("message", {}).get("content", "")


# Discriminated union for StreamingResponse
class StreamingResponse(BaseModel):
    id: str
    model: str
    provider: Literal["openai", "groq", "deepseek"]
    choices: List[Dict[str, Any]]
    object: Optional[str] = None
    created: Optional[int] = None
    
    def extract_content(self) -> str:
        """Extracts the content from a streaming response."""
        if not self.choices or len(self.choices) == 0:
            return ""
        
        # Get the first choice
        choice = self.choices[0]
        
        # Handle based on provider
        if self.provider == "openai" or self.provider == "groq":
            return choice.get("delta", {}).get("content", "")
        elif self.provider == "deepseek":
            return choice.get("text", "") or choice.get("message", {}).get("content", "")
        
        # Fallback extraction logic
        if "delta" in choice and "content" in choice["delta"]:
            return choice["delta"]["content"] or ""
        elif "text" in choice:
            return choice["text"] or ""
        elif "content" in choice:
            return choice["content"] or ""
        elif "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"] or ""
        
        return ""


# Function to Parse Streaming Response
def parse_streaming_response(data: str) -> StreamingResponse:
    json_data = json.loads(data)
    
    # Add provider field if not present but can be inferred
    if "provider" not in json_data:
        if json_data.get("object") == "chat.completion.chunk":
            # Could be OpenAI or Groq, need more info to distinguish
            # For now, default to OpenAI
            json_data["provider"] = "openai"
        # Add more provider detection logic as needed
    
    return StreamingResponse(**json_data)