from completionschat import ChatCompletionsClient as chatCompletion
from completions import Completions as completions
class Chat:
    def __init__(self, chat_client: chatCompletion):
        self.completions = completions(chat_client)