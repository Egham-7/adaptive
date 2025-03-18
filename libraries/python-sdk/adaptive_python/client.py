import requests
import httpx
import asyncio
from typing import Callable, List, Optional
from pydantic import BaseModel
from provider import ChatCompletionRequest, ChatCompletionResponse, StreamingResponse, Message, parse_streaming_response
from utils import parse_provider_response
from exceptions import APIError
import json

API_BASE_URL = "http://localhost:8080/api/chat/completions"


class ChatCompletionsClient:
    def __init__(self):
        self.session = requests.Session()


    def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Sends a chat request and returns the response."""
        response = self.session.post(API_BASE_URL, json=request.model_dump())  

        if response.status_code != 200:
            raise APIError(f"API Error: {response.status_code} - {response.text}")

        data = response.json()
        parsed_response = parse_provider_response(data["provider"], data["response"])

        return ChatCompletionResponse(provider=data["provider"], response=parsed_response)




    async def create_streaming_chat_completion(
        self,
        request: ChatCompletionRequest,
        on_chunk: Callable[[StreamingResponse], None],
        on_complete: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> Callable[[], None]:
        """
        Creates a streaming chat completion request.
        
        Args:
            request: The chat completion request
            on_chunk: Callback function for each chunk received
            on_complete: Optional callback function when stream completes
            on_error: Optional callback function when an error occurs
            
        Returns:
            A function that can be called to abort the stream
        """
        # Create client and task for cancellation
        client = httpx.AsyncClient()
        task = None

        async def stream_processor():
            try:
                # Make request with appropriate headers for SSE
                async with client.stream(
                    "POST",
                    f"{API_BASE_URL}/stream",
                    json=request.model_dump(),
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                    },
                    timeout=60.0,
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise APIError(f"API Error: {response.status_code} - {error_text.decode('utf-8')}")

                    # Process the stream
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk

                        # Process complete SSE messages
                        while "\n\n" in buffer:
                            line, buffer = buffer.split("\n\n", 1)

                            if not line.strip() or not line.startswith("data: "):
                                continue

                            data_content = line[6:]  # Remove 'data: ' prefix

                            # Check for the special [DONE] message
                            if data_content.strip() == "[DONE]":
                                if on_complete:
                                    on_complete()
                                continue

                            try:
                                # Parse the JSON content and pass to callback
                                streaming_response = parse_streaming_response(data_content)
                                on_chunk(streaming_response)
                            except Exception as e:
                                print(f"Error parsing SSE message: {e}")

                # Call on_complete if we finished without errors
                if on_complete:
                    on_complete()

            except Exception as e:
                # Only call on_error if it's not a cancellation
                if isinstance(e, asyncio.CancelledError):
                    return
                if on_error:
                    on_error(e)
            finally:
                await client.aclose()

        # Start the streaming task
        task = asyncio.create_task(stream_processor())

        # Return a function that can abort the stream
        def abort_stream():
            if task and not task.done():
                task.cancel()

        return abort_stream

class Adaptive:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.chatcompletion = ChatCompletionsClient()


class Adaptive:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._chatcompletion = ChatCompletionsClient()

    def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return self._chatcompletion.create_chat_completion(request)

    async def create_streaming_chat_completion(
        self,
        request: ChatCompletionRequest,
        on_chunk: Callable[[StreamingResponse], None],
        on_complete: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> Callable[[], None]:
        return await self._chatcompletion.create_streaming_chat_completion(request, on_chunk, on_complete, on_error)


# Example usage:
adaptive = Adaptive(api_key="test")
request = ChatCompletionRequest(messages=[Message(role="user", content="What is the capital of France?")])
async def test_streaming():
    print("\nTesting streaming functionality...")

    def on_chunk(chunk):
        print(f"Received chunk: {chunk}")

    def on_complete():
        print("Streaming completed!")

    def on_error(error):
        print(f"Streaming error: {error}")

    # Start streaming and get the abort function
    abort_stream = await adaptive.create_streaming_chat_completion(
        request=request,
        on_chunk=on_chunk,
        on_complete=on_complete,
        on_error=on_error
    )

    # Wait for a while to receive some chunks
   
asyncio.run(test_streaming())
