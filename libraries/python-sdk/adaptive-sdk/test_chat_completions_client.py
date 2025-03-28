from adaptive import Adaptive

# For testing during development
if __name__ == "__main__":
    adaptive = Adaptive(api_key="test")

    # Example messages
    messages = [
        {"role": "user", "content": "Say 'double bubble bath' two times fast."}
    ]

    print("\nTesting functionality...")

    # Test non-streaming
    try:
        response = adaptive.chat.completions.create(messages=messages, stream=False)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Continuing with streaming test...")

    # Test streaming
    try:
        stream = adaptive.chat.completions.create(messages=messages, stream=True)
        for chunk in stream:
            print(f"Chunk: {chunk}")
    except Exception as e:
        print(f"Streaming error occurred: {e}")