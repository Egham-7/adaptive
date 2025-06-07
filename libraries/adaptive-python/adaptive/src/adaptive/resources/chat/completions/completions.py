from typing import Iterator, List, Dict, Union
import requests

# mypy: disable-error-code="import"

from adaptive.models.completions import (
    ChatCompletionResponse,  # type: ignore
    ChatCompletionStreamingResponse,  # type: ignore
)
from adaptive.exceptions.api import APIError  # type: ignore
from adaptive.resources.base import BaseAPIClient  # type: ignore

# type: ignore # type: ignore # type: ignore # type: ignore
from adaptive.utils.streaming import SSEStreamHandler


class Completions(BaseAPIClient):
    """
    Client for chat completions API that inherits from BaseAPIClient.
    """

    def __init__(self, api_key: str, base_url: str):
        """
        Initialize the Completions client.
        Args:
            api_key: API key for authentication
            base_url: Base URL for the API
        """
        super().__init__(base_url, api_key)
        self.stream_handler = SSEStreamHandler()

    def create(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletionResponse, Iterator[ChatCompletionStreamingResponse]]:
        """
        Creates a chat completion request.
        Args:
            messages: A list of message dictionaries with 'role' and 'content'
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API
        Returns:
            Either a ChatCompletionResponse or an Iterator of ChatCompletionStreamingResponse objects
        """
        if stream:
            return self._create_streaming_chat_completion(messages, **kwargs)
        else:
            return self._create_chat_completion(messages, **kwargs)

    def _create_chat_completion(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> ChatCompletionResponse:
        """
        Creates a non-streaming chat completion request.
        Args:
            messages: A list of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters to pass to the API
        Returns:
            A ChatCompletionResponse object
        """
        # Build request payload
        payload = {
            "messages": messages,
        }
        # Add any additional kwargs
        payload.update(kwargs)
        # Send request using the base client's post method
        data = self.post("api/chat/completions", json_data=payload)

        if not isinstance(data, dict):
            raise APIError("Invalid response format")

        provider = str(data.get("provider", ""))
        response_data = data.get("response", {})

        if not isinstance(response_data, dict):
            raise APIError("Invalid response format")

        parsed_response = ChatCompletionResponse.parse_provider_response(
            provider, response_data
        )
        return ChatCompletionResponse(
            provider=data.get("provider"), response=parsed_response
        )

    def _handle_streaming_response(
        self, response: requests.Response
    ) -> Iterator[ChatCompletionStreamingResponse]:
        """
        Process a streaming response using the provided stream handler.
        Args:
            response: The HTTP response object
            stream_handler: The handler for processing stream chunks
        Returns:
            An iterator of ChatCompletionStreamingResponse objects
        """
        if response.status_code != 200:
            raise APIError(f"API Error: {response.status_code} - {response.text}")

        for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
            if not chunk:
                continue
            result = self.stream_handler.process_chunk(chunk)
            if result:
                yield result
            if self.stream_handler.is_done(chunk):
                break

    def _create_streaming_chat_completion(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Iterator[ChatCompletionStreamingResponse]:
        """
        Creates a streaming chat completion request.
        Args:
            messages: A list of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters to pass to the API
        Returns:
            An iterator that yields streaming responses
        """
        # Build request payload
        payload = {
            "messages": messages,
        }
        # Add any additional kwargs
        payload.update(kwargs)
        # Set up streaming headers
        streaming_headers = {
            "Accept": "text/event-stream",
        }
        # Make the streaming request using the base client's post method with stream=True
        response = self.request(
            "POST",
            "api/chat/completions/stream",
            json_data=payload,
            headers=streaming_headers,
            stream=True,
        )

        if not isinstance(response, Iterator):
            raise APIError("Invalid response format")

        # Transform each dictionary in the response to a ChatCompletionStreamingResponse object
        for chunk in response:
            # Convert each dictionary chunk to a ChatCompletionStreamingResponse object
            yield ChatCompletionStreamingResponse(**chunk.__dict__)
