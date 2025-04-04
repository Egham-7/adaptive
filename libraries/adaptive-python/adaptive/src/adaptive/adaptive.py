from typing import Optional
from resources.chat import Chat  # type: ignore
import os


class Adaptive:
    def __init__(self, api_key: Optional[str], base_url: Optional[str]):
        """
        Initialize the Adaptive client.
        """

        # Read the API key from the environment variable if not provided

        if not api_key:
            api_key = os.getenv("ADAPTIVE_API_KEY")

        if not api_key:
            raise ValueError("API key is required")

        if not base_url:
            base_url = (
                "https://backend-go.salmonwave-ec8d1f2a.eastus.azurecontainerapps.io/"
            )
        self.chat = Chat(api_key=api_key, base_url=base_url)
