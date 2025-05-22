from typing import Optional
from adaptive.resources.chat import Chat  # type: ignore
import os


class Adaptive:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the Adaptive client.

        Args:
            api_key (Optional[str]): The API key to authenticate with the Adaptive backend.
            base_url (Optional[str]): The base URL of the Adaptive backend.

        Raises:
            ValueError: If the API key is not provided or found in the environment.
        """

        # Load the API key from environment if not provided
        self.api_key = api_key or os.getenv("ADAPTIVE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Adaptive SDK: Missing API key. You must provide it explicitly or set it as an environment variable 'ADAPTIVE_API_KEY'."
            )

        # Set a default base URL if none is provided
        self.base_url = base_url or os.getenv(
            "ADAPTIVE_BASE_URL",
            "https://backend-go.salmonwave-ec8d1f2a.eastus.azurecontainerapps.io/",
        )

        if (
            self.base_url
            and not self.base_url.startswith("https://")
            and not self.base_url.startswith("http://")
        ):
            raise ValueError(
                f"Adaptive SDK: Invalid base_url '{self.base_url}'. It must start with http:// or https://"
            )

        try:
            self.chat = Chat(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            raise RuntimeError(
                f"Adaptive SDK: Failed to initialize Chat client. Reason: {str(e)}"
            ) from e
