import requests
from typing import Iterator, List, Dict, Any, Optional, Callable
from provider.provider import ChatCompletionResponse, StreamingResponse, parse_streaming_response
from utils.utils import parse_provider_response
from exceptions.api_error import APIError

API_BASE_URL = "https://backend-go.salmonwave-ec8d1f2a.eastus.azurecontainerapps.io/"

class StreamHandler:
    """Base class for handling different types of streaming responses."""
    
    def process_chunk(self, chunk: str) -> Optional[StreamingResponse]:
        """Process a chunk of data from the stream."""
        raise NotImplementedError("Subclasses must implement process_chunk")
    
    def is_done(self, chunk: str) -> bool:
        """Check if the chunk indicates the stream is done."""
        return False

class SSEStreamHandler(StreamHandler):
    """Handler for Server-Sent Events (SSE) streams."""
    
    def __init__(self, parser_func: Callable[[str], StreamingResponse]):
        self.buffer = ""
        self.parser_func = parser_func
    
    def process_chunk(self, chunk: str) -> Optional[StreamingResponse]:
        """Process a chunk from an SSE stream."""
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
            return self.parser_func(data_content)
        except Exception as e:
            print(f"Error parsing SSE message: {e}")
            return None
    
    def is_done(self, chunk: str) -> bool:
        """Check if the chunk contains a [DONE] message."""
        return "data: [DONE]" in chunk

class ChatCompletionsClient:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def create_chat_completion(
        self,
        messages,
        **kwargs
    ) -> ChatCompletionResponse:
        """
        Creates a chat completion request similar to OpenAI's interface.
        
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
        
        # Send request
        response = self.session.post(
            f"{API_BASE_URL}/api/chat/completions", 
            json=payload,
            headers=self.headers
        )  

        if response.status_code != 200:
            # For development/debugging, you might want to extract the response even on error
            try:
                data = response.json()
                if "provider" in data and "response" in data:
                    print(f"Warning: Got status code {response.status_code} but response looks valid")
                    parsed_response = parse_provider_response(data["provider"], data["response"])
                    return ChatCompletionResponse(provider=data["provider"], response=parsed_response)
            except:
                pass
            raise APIError(f"API Error: {response.status_code} - {response.text}")

        data = response.json()
        parsed_response = parse_provider_response(data["provider"], data["response"])

        return ChatCompletionResponse(provider=data["provider"], response=parsed_response)

    def _handle_streaming_response(
        self,
        response: requests.Response,
        stream_handler: StreamHandler
    ) -> Iterator[StreamingResponse]:
        """
        Process a streaming response using the provided stream handler.
        
        Args:
            response: The HTTP response object
            stream_handler: The handler for processing stream chunks
            
        Returns:
            An iterator of StreamingResponse objects
        """
        if response.status_code != 200:
            raise APIError(f"API Error: {response.status_code} - {response.text}")
        
        for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
            if not chunk:
                continue
                
            result = stream_handler.process_chunk(chunk)
            if result:
                yield result
                
            if stream_handler.is_done(chunk):
                break

    def create_streaming_chat_completion(
        self,
        messages: List[Dict[str, str]] = None,
        provider: str = None,
        **kwargs
    ) -> Iterator[StreamingResponse]:
        """
        Creates a streaming chat completion request.
        
        Args:
            messages: A list of message dictionaries with 'role' and 'content'
            provider: Optional provider to use (if not specified, the API will choose)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            An iterator that yields streaming responses
        """
        # Build request payload
        payload = {
            "messages": messages,
        }
        
        if provider:
            payload["provider"] = provider
            
        # Add any additional kwargs
        payload.update(kwargs)
        
        # Set up streaming headers
        streaming_headers = {
            **self.headers,
            "Accept": "text/event-stream",
        }
        
        # Make the streaming request
        response = self.session.post(
            f"{API_BASE_URL}/api/chat/completions/stream",
            json=payload,
            headers=streaming_headers,
            stream=True
        )
        
        # Create appropriate stream handler based on provider
        # For now, we only have one handler type, but this can be extended
        stream_handler = SSEStreamHandler(parse_streaming_response)
        
        # Process the stream
        yield from self._handle_streaming_response(response, stream_handler)

    def stream_complete(
        self,
        messages: List[Dict[str, str]] = None,
        provider: str = None,
        **kwargs
    ) -> str:
        """
        Wrapper for streaming chat completion that returns the full response as a string.
        
        Args:
            messages: A list of message dictionaries with 'role' and 'content'
            provider: Optional provider to use
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The complete response as a string
        """
        full_response = ""
        for chunk in self.create_streaming_chat_completion(
            messages=messages, 
            provider=provider,
            **kwargs
        ):
            if chunk.delta and hasattr(chunk.delta, 'content') and chunk.delta.content:
                full_response += chunk.delta.content
        
        return full_response