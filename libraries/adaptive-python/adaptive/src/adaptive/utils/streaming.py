from adaptive.models.completions import ChatCompletionStreamingResponse  # type: ignore
from typing import Generic, Optional, TypeVar
import json

T = TypeVar("T")


class StreamHandler(Generic[T]):
    """Base class for handling different types of streaming responses."""

    def process_chunk(self, _chunk: str) -> Optional[T]:
        """
        Process a chunk of data from the stream.

        Args:
            chunk: A chunk of data from the stream

        Returns:
            Optional processed result of type T
        """
        raise NotImplementedError("Subclasses must implement process_chunk")

    def is_done(self, _chunk: str) -> bool:
        """
        Check if the chunk indicates the stream is done.

        Args:
            chunk: A chunk of data from the stream

        Returns:
            True if the stream is done, False otherwise
        """
        return False


class SSEStreamHandler(StreamHandler[ChatCompletionStreamingResponse]):
    """Handler for Server-Sent Events (SSE) streams."""

    def __init__(self):
        self.buffer = ""

    def process_chunk(self, chunk: str) -> Optional[ChatCompletionStreamingResponse]:
        """
        Process a chunk from an SSE stream.

        Args:
            chunk: A chunk of data from the SSE stream

        Returns:
            Optional ChatCompletionStreamingResponse if a complete message was parsed
        """
        self.buffer += chunk

        # Check if we have a complete SSE message
        if "\n\n" not in self.buffer:
            return None

        # Extract the complete message
        line, self.buffer = self.buffer.split("\n\n", 1)
        if not line.strip() or not line.startswith("data: "):
            return None

        data_content = line[6:]  # Remove 'data: ' prefix

        # Check for the special [DONE] message
        if data_content.strip() == "[DONE]":
            return None

        try:
            # Parse the JSON content
            data = json.loads(data_content)
            # The provider is determined by the API, no need to add it manually
            return ChatCompletionStreamingResponse(**data)
        except Exception as e:
            print(f"Error parsing SSE message: {e}")
            return None

    def is_done(self, chunk: str) -> bool:
        """
        Check if the chunk contains a [DONE] message.

        Args:
            chunk: A chunk of data from the SSE stream

        Returns:
            True if the chunk contains a [DONE] message, False otherwise
        """
        return "data: [DONE]" in chunk
