import requests
import httpx
import asyncio
from typing import Callable
from .provider import ChatCompletionRequest, ChatCompletionResponse, StreamingResponse
from .utils import parse_provider_response
from .exceptions import APIError

API_BASE_URL = "http://localhost:8080/chat/completions"

class ChatCompletionsClient:
    def __init__(self):
        self.session = requests.Session()

    def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Sends a chat request and returns the response."""
        response = self.session.post(API_BASE_URL, json=request.model_dump())  # Updated from .dict() to .model_dump()

        if response.status_code != 200:
            raise APIError(f"API Error: {response.status_code} - {response.text}")

        data = response.json()
        parsed_response = parse_provider_response(data["provider"], data["response"])

        return ChatCompletionResponse(provider=data["provider"], response=parsed_response)

    async def create_streaming_chat_completion(
        self, 
        request: ChatCompletionRequest, 
        on_chunk: Callable[[StreamingResponse], None], 
        on_complete: Callable[[], None] = None, 
        on_error: Callable[[Exception], None] = None
    ):
        """Handles streaming chat completion."""
        url = f"{API_BASE_URL}/stream"
        headers = {"Accept": "text/event-stream"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=request.model_dump(), headers=headers, timeout=None)  # Updated from .dict() to .model_dump()
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip() or not line.startswith("data: "):
                        continue

                    data_content = line[6:].strip()  # Remove 'data: ' prefix
                    if data_content == "[DONE]":
                        if on_complete:
                            on_complete()
                        break

                    try:
                        json_data = StreamingResponse.parse_raw(data_content)
                        on_chunk(json_data)
                    except Exception as e:
                        print(f"Error parsing stream: {e}")
                        if on_error:
                            on_error(e)

            except httpx.HTTPStatusError as e:
                if on_error:
                    on_error(e)
            except Exception as e:
                if on_error:
                    on_error(e)
