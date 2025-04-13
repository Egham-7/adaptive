from adaptive.adaptive import Adaptive  # type: ignore
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    # Initialize the Adaptive client
    adaptive = Adaptive()

    # Define a list of messages
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thanks!"},
        {"role": "user", "content": "What's the weather like today?"},
    ]

    try:
        # Create a non-streaming chat completion request
        response = adaptive.chat.completions.create(messages)

        print("Non-Streaming Response:")
        print(response)

        # Create a streaming chat completion request
        streaming_response = adaptive.chat.completions.create(messages, stream=True)

        print("\nStreaming Response:")
        for chunk in streaming_response:
            print(chunk)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
