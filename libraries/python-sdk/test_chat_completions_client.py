import unittest
from unittest.mock import patch, MagicMock
import requests
import httpx
from adaptive_python import ChatCompletionsClient, ChatCompletionRequest, ChatCompletionResponse, StreamingResponse, APIError

class TestChatCompletionsClient(unittest.TestCase):
    
    @patch('requests.Session.post')  # Mocking the `requests.post` method for the synchronous request
    def test_create_chat_completion_success(self, mock_post):
        # Prepare mock response data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "provider": "some-provider",
            "response": "some-response"
        }
        mock_post.return_value = mock_response

        # Create a mock ChatCompletionRequest
        mock_request = MagicMock(spec=ChatCompletionRequest)
        mock_request.model_dump.return_value = {"some_field": "some_value"}

        # Instantiate the client
        client = ChatCompletionsClient()

        # Call the method
        response = client.create_chat_completion(mock_request)

        # Assertions
        mock_post.assert_called_once_with("http://localhost:8080/chat/completions", json={"some_field": "some_value"})
        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertEqual(response.provider, "some-provider")
        self.assertEqual(response.response, "some-response")

    @patch('requests.Session.post')  # Mocking the `requests.post` method for the synchronous request
    def test_create_chat_completion_failure(self, mock_post):
        # Simulate a failed request
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        # Create a mock ChatCompletionRequest
        mock_request = MagicMock(spec=ChatCompletionRequest)
        mock_request.model_dump.return_value = {"some_field": "some_value"}

        # Instantiate the client
        client = ChatCompletionsClient()

        # Call the method and expect an APIError
        with self.assertRaises(APIError) as context:
            client.create_chat_completion(mock_request)
        
        self.assertTrue("API Error" in str(context.exception))

    @patch('httpx.AsyncClient.post')  # Mocking the `httpx.AsyncClient.post` method for the asynchronous streaming request
    @patch('httpx.AsyncClient.aiter_lines', return_value=["data: {\"field\": \"value\"}", "data: [DONE]"])  # Mocking streaming lines
    def test_create_streaming_chat_completion(self, mock_post, mock_aiter_lines):
        # Prepare mock response data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Create a mock ChatCompletionRequest
        mock_request = MagicMock(spec=ChatCompletionRequest)
        mock_request.model_dump.return_value = {"some_field": "some_value"}

        # Create a mock callback function for on_chunk
        def on_chunk(data):
            self.assertEqual(data.field, "value")

        # Instantiate the client
        client = ChatCompletionsClient()

        # Call the async method
        async def test_method():
            await client.create_streaming_chat_completion(
                mock_request,
                on_chunk=on_chunk,
                on_complete=lambda: print("Done"),
                on_error=lambda e: print(f"Error: {e}")
            )

        # Run the test method
        asyncio.run(test_method())

        # Check that `aiter_lines` was called
        mock_aiter_lines.assert_called_once()

    @patch('httpx.AsyncClient.post')  # Mocking the `httpx.AsyncClient.post` method for the asynchronous streaming request
    @patch('httpx.AsyncClient.aiter_lines', return_value=["data: invalid_json", "data: [DONE]"])  # Simulate invalid JSON
    def test_create_streaming_chat_completion_error(self, mock_post, mock_aiter_lines):
        # Prepare mock response data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Create a mock ChatCompletionRequest
        mock_request = MagicMock(spec=ChatCompletionRequest)
        mock_request.model_dump.return_value = {"some_field": "some_value"}

        # Create a mock callback function for on_error
        def on_error(e):
            self.assertIsInstance(e, Exception)

        # Instantiate the client
        client = ChatCompletionsClient()

        # Call the async method
        async def test_method():
            await client.create_streaming_chat_completion(
                mock_request,
                on_chunk=lambda data: None,
                on_complete=lambda: None,
                on_error=on_error
            )

        # Run the test method
        asyncio.run(test_method())

        # Check that `aiter_lines` was called
        mock_aiter_lines.assert_called_once()


if __name__ == '__main__':
    unittest.main()
