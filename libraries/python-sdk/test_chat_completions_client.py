
# If this file is run directly, execute the test
if __name__ == "__main__":
    try:
        # Create a client instance
        client = ChatCompletionsClient()

        # Create a test message - properly instantiate a Message object
        messages = [
            Message(role="user", content="What is the capital of France?")
        ]

        # Create a request with the messages
        request = ChatCompletionRequest(messages=messages)

        print("Sending chat completion request...")
        # Send the request and get the response
        response = client.create_chat_completion(request)

        print(f"Provider: {response.provider}")
        print(f"Response: {response.response}")

        # Test streaming functionality
        async def test_streaming():
            print("\nTesting streaming functionality...")

            def on_chunk(chunk):
                print(f"Received chunk: {chunk}")

            def on_complete():
                print("Streaming completed!")

            def on_error(error):
                print(f"Streaming error: {error}")

            # Start streaming and get the abort function
            abort_stream = await client.create_streaming_chat_completion(
                request=request,
                on_chunk=on_chunk,
                on_complete=on_complete,
                on_error=on_error
            )

            # Wait for a while to receive some chunks
            await asyncio.sleep(10)

            # Uncomment to test cancellation
            # print("Cancelling stream...")
            # abort_stream()

        # Run the streaming test if asyncio is available
        print("\nWould you like to test streaming? (y/n)")
        choice = input().lower()
        if choice == 'y':
            asyncio.run(test_streaming())

        print("\nTests completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")