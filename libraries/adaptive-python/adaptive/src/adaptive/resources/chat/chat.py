from resources.chat.completions.completions import Completions  # type: ignore
from adaptive.resources.base import BaseAPIClient  # type: ignore


class Chat(BaseAPIClient):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.api_key = api_key
        self.base_url = base_url
        self.completions = Completions(self.api_key, self.base_url)
