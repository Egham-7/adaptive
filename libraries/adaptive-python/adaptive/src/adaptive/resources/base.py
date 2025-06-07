import requests
from typing import Dict, Any, Optional, Union, Iterator
import logging
from adaptive.exceptions.api import APIError  # type: ignore
import json


class BaseAPIClient:
    """Base API client for making HTTP requests to external services."""

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 60):
        """
        Initialize the base API client.
        Args:
            base_url: The base URL for the API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _build_url(self, endpoint: str) -> str:
        """Build the full URL for the API request."""
        endpoint = endpoint.lstrip("/")
        return f"{self.base_url}/{endpoint}"

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle the API response and raise appropriate exceptions."""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error: {e}"
            try:
                error_data = response.json()
                error_msg = f"{error_msg}, Details: {error_data}"
            except ValueError:
                error_msg = f"{error_msg}, Response: {response.text}"
            self.logger.error(error_msg)
            raise APIError(
                error_msg, status_code=response.status_code, response=response
            )
        except ValueError:
            error_msg = f"Invalid JSON response: {response.text}"
            self.logger.error(error_msg)
            raise APIError(
                error_msg, status_code=response.status_code, response=response
            )

    def _handle_streaming_response(
        self, response: requests.Response
    ) -> Iterator[Dict[str, Any]]:
        """Handle streaming API response and yield parsed JSON chunks."""
        try:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        yield json.loads(line)
                    except ValueError as e:
                        self.logger.warning(
                            f"Failed to parse JSON from stream with error {e} at line: {line.decode('utf-8')}"
                        )
                        # Continue processing other lines even if one fails
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error in stream: {e}"
            try:
                error_data = response.json()
                error_msg = f"{error_msg}, Details: {error_data}"
            except ValueError:
                error_msg = f"{error_msg}, Response: {response.text}"
            self.logger.error(error_msg)
            raise APIError(
                error_msg, status_code=response.status_code, response=response
            )

    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Make an HTTP request to the API.
        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint
            params: URL parameters
            data: Form data
            json_data: JSON data
            headers: Additional headers
            stream: Whether to stream the response
        Returns:
            Parsed JSON response or an iterator of parsed JSON chunks if streaming
        """
        url = self._build_url(endpoint)
        request_headers = {}
        if headers:
            request_headers.update(headers)
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json_data,
                headers=request_headers,
                timeout=self.timeout,
                stream=stream,
            )

            if stream:
                return self._handle_streaming_response(response)
            else:
                return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            self.logger.error(error_msg)
            raise APIError(error_msg, original_error=e)

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Make a GET request."""
        return self.request(
            "GET", endpoint, params=params, headers=headers, stream=stream
        )

    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Make a POST request."""
        return self.request(
            "POST",
            endpoint,
            data=data,
            json_data=json_data,
            headers=headers,
            stream=stream,
        )

    def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Make a PUT request."""
        return self.request(
            "PUT",
            endpoint,
            data=data,
            json_data=json_data,
            headers=headers,
            stream=stream,
        )

    def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Make a DELETE request."""
        return self.request(
            "DELETE", endpoint, params=params, headers=headers, stream=stream
        )
