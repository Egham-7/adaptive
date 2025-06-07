from typing import Optional
import requests


class APIError(Exception):
    """Exception raised for API errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[requests.Response] = None,
        original_error: Optional[Exception] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response = response
        self.original_error = original_error
        super().__init__(self.message)
