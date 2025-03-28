from chat.chat_completions_client import ChatCompletionsClient as chatCompletion
from chat.completions import Completions as completions
class Chat:
    def __init__(self, chat_client: chatCompletion):
        self.completions = completions(chat_client)