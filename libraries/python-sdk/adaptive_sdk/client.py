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