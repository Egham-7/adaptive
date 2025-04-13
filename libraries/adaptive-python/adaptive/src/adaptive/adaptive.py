from typing import Optional
from adaptive.resources.chat import Chat  # type: ignore
import os


class Adaptive:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the Adaptive client.
        """

        # Read the API key from the environment variable if not provided

        if not api_key:
            api_key = os.getenv("ADAPTIVE_API_KEY")

        if not base_url:
            base_url = (
                "https://backend-go.salmonwave-ec8d1f2a.eastus.azurecontainerapps.io/"
            )
        self.chat = Chat(api_key=api_key, base_url=base_url)
