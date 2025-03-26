from chat.chat_completions_client import ChatCompletionsClient
from chat.chat import Chat
from exceptions.api_error import APIError


class Adaptive:
    def __init__(self, api_key: str):
        """
        Initialize the Adaptive client.
        """
        self._chat_client = ChatCompletionsClient(api_key=api_key)
        self.chat = Chat(self._chat_client)

# For testing during development
if __name__ == "__main__":
    client = Adaptive(api_key="test")

    # Use a test API key for development
    adaptive = Adaptive(api_key="test")
    
    # Example messages in OpenAI format
    messages = [
        {
            "role": "user",
            "content": "Say 'double bubble bath' two times fast.",
        },
    ]
    
    print("\nTesting functionality...")
    
    # For development, you might want to handle the 401 error differently
    try:
        # Non-streaming example
        response = adaptive.chat.completions.create(
            messages=messages,
            stream=False
        )
        print(f"Response: {response}")
    except APIError as e:
        print(f"API Error occurred: {e}")
        # During development, you might want to continue despite errors
        print("Continuing with streaming test...")
    
    # Test streaming
    try:
        # Streaming example
        stream = adaptive.chat.completions.create(
            messages=messages,
            stream=True
        )
        for chunk in stream:
            print(f"Chunk: {chunk}")
    except APIError as e:
        print(f"Streaming API Error occurred: {e}")