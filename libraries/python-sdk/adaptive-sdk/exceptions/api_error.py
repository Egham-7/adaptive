# libraries/exceptions/api_error.py

class APIError(Exception):
    """
    Exception raised for errors that occur during API calls.

    Attributes:
        status_code (int): HTTP status code returned by the API, if applicable.
        message (str): Explanation of the error.
        payload (dict): Optional additional data provided with the error.
    """
    def __init__(self, message: str, status_code: int = None, payload: dict = None):
        self.message = message
        self.status_code = status_code
        self.payload = payload or {}
        super().__init__(self.message)

    def __str__(self):
        error_info = f"APIError: {self.message}"
        if self.status_code is not None:
            error_info += f" (Status Code: {self.status_code})"
        if self.payload:
            error_info += f" | Payload: {self.payload}"
        return error_info
