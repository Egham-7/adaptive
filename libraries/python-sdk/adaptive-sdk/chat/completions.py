
from provider.provider import ChatCompletionResponse, StreamingResponse
from typing import Iterator, Union
from chat.chat_completions_client import ChatCompletionsClient

class Completions:
    def __init__(self, chat_client: ChatCompletionsClient):
        self._chat_client = chat_client

    def create(
        self,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletionResponse, Iterator[StreamingResponse]]:
        """
        Creates a chat completion request.
        """
        if not stream:
            return self._chat_client.create_chat_completion(
                **kwargs
            )
        return self._chat_client.create_streaming_chat_completion(
      
            **kwargs
        )