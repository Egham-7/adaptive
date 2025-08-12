"""Base classes for scalable performance testing."""

import json
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import requests
import aiohttp
import asyncio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax

from .config import Config, EndpointConfig


@dataclass
class TestResult:
    """Single test result."""

    endpoint_name: str
    method: str
    url: str
    status_code: int
    response_time: float  # in milliseconds
    success: bool
    error_message: Optional[str] = None
    request_size: int = 0
    response_size: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestSession:
    """Test session results."""

    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    results: List[TestResult] = field(default_factory=list)
    config: Optional[Dict[str, Any]] = None

    @property
    def duration(self) -> float:
        """Session duration in seconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    @property
    def total_requests(self) -> int:
        """Total number of requests made."""
        return len(self.results)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.success)
        return (successful / len(self.results)) * 100

    @property
    def average_response_time(self) -> float:
        """Average response time in milliseconds."""
        if not self.results:
            return 0.0
        return sum(r.response_time for r in self.results) / len(self.results)

    def get_percentiles(self) -> Dict[str, float]:
        """Get response time percentiles."""
        if not self.results:
            return {"p50": 0, "p95": 0, "p99": 0}

        response_times = sorted(r.response_time for r in self.results)
        n = len(response_times)

        return {
            "p50": response_times[int(n * 0.5)],
            "p95": response_times[int(n * 0.95)],
            "p99": response_times[int(n * 0.99)],
        }


class BaseEndpointTester(ABC):
    """Base class for endpoint testing."""

    def __init__(self, config: Config, console: Optional[Console] = None):
        self.config = config
        self.console = console or Console()
        self.session = requests.Session()
        self._setup_session()

    def _setup_session(self):
        """Setup requests session with global configuration."""
        self.session.headers.update(self.config.global_settings.headers)

        # Add authentication header if configured
        auth_header = self.config.get_auth_header()
        if auth_header:
            self.session.headers.update(auth_header)

        # Configure timeouts and retries
        adapter = requests.adapters.HTTPAdapter(
            max_retries=self.config.global_settings.max_retries
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_full_url(self, endpoint: EndpointConfig) -> str:
        """Get full URL for endpoint."""
        return f"{self.config.base_url.rstrip('/')}{endpoint.path}"

    def prepare_request_data(
        self, endpoint: EndpointConfig
    ) -> Optional[Dict[str, Any]]:
        """Prepare request data for endpoint."""
        if endpoint.test_data is None:
            return None

        if isinstance(endpoint.test_data, list):
            return random.choice(endpoint.test_data)
        return endpoint.test_data

    def make_request(self, endpoint_name: str, endpoint: EndpointConfig) -> TestResult:
        """Make a single request to endpoint."""
        url = self.get_full_url(endpoint)
        data = self.prepare_request_data(endpoint)

        # Prepare request arguments
        headers = dict(self.session.headers)
        if endpoint.headers:
            headers.update(endpoint.headers)

        # Prepare request data
        json_data = None
        if data is not None and endpoint.method.upper() in ["POST", "PUT", "PATCH"]:
            json_data = data

        # Verbose logging - request
        if self.config.global_settings.verbose:
            # Convert headers to string-only dict for logging
            str_headers = {k: str(v) for k, v in headers.items()}
            self._log_request(
                endpoint_name, endpoint.method, url, str_headers, json_data
            )

        start_time = time.time()
        success = False
        status_code = 0
        error_message = None
        response_size = 0
        response_content = None

        try:
            response = self.session.request(
                endpoint.method,
                url,
                headers=headers,
                json=json_data,
                timeout=self.config.global_settings.timeout,
                verify=self.config.global_settings.verify_ssl,
            )
            status_code = response.status_code
            response_content = response.content if response.content else b""
            response_size = len(response_content)

            # Consider 2xx and 3xx as success
            success = 200 <= status_code < 400

            if not success:
                error_message = f"HTTP {status_code}: {response.reason}"

        except Exception as e:
            error_message = str(e)

        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Verbose logging - response
        if self.config.global_settings.verbose:
            # Ensure response_content is bytes for logging
            log_content = response_content if response_content is not None else b""
            self._log_response(
                endpoint_name, status_code, response_time, log_content, error_message
            )

        return TestResult(
            endpoint_name=endpoint_name,
            method=endpoint.method,
            url=url,
            status_code=status_code,
            response_time=response_time,
            success=success,
            error_message=error_message,
            request_size=len(json.dumps(data).encode()) if data else 0,
            response_size=response_size,
        )

    def _log_request(
        self,
        endpoint_name: str,
        method: str,
        url: str,
        headers: Dict[str, str],
        json_data: Optional[Dict[str, Any]],
    ):
        """Log request details in verbose mode."""
        self.console.print(f"\n[bold blue]🔄 Request to {endpoint_name}[/bold blue]")
        self.console.print(f"[cyan]Method:[/cyan] {method}")
        self.console.print(f"[cyan]URL:[/cyan] {url}")

        # Log headers (mask authentication keys)
        masked_headers = dict(headers)
        auth_headers = ["Authorization", "X-API-Key", "api-key", "X-Stainless-API-Key"]
        for auth_header in auth_headers:
            if auth_header in masked_headers:
                masked_headers[auth_header] = "***"

        self.console.print(
            Panel(
                Syntax(json.dumps(masked_headers, indent=2), "json"),
                title="Headers",
                border_style="blue",
            )
        )

        if json_data:
            self.console.print(
                Panel(
                    Syntax(json.dumps(json_data, indent=2), "json"),
                    title="Request Body",
                    border_style="blue",
                )
            )

    def _log_response(
        self,
        endpoint_name: str,
        status_code: int,
        response_time: float,
        content: bytes,
        error_message: Optional[str],
    ):
        """Log response details in verbose mode."""
        status_color = "green" if 200 <= status_code < 400 else "red"
        self.console.print(
            f"[bold {status_color}]📥 Response from {endpoint_name}[/bold {status_color}]"
        )
        self.console.print(
            f"[cyan]Status:[/cyan] [{status_color}]{status_code}[/{status_color}]"
        )
        self.console.print(f"[cyan]Response Time:[/cyan] {response_time:.2f}ms")

        if error_message:
            self.console.print(f"[red]Error:[/red] {error_message}")

        if content:
            try:
                # Try to parse as JSON for better formatting
                response_json = json.loads(content.decode("utf-8"))
                # Truncate long responses for readability
                if len(str(response_json)) > 1000:
                    response_json = {
                        "...": "Response truncated for verbose mode",
                        "size": f"{len(content)} bytes",
                    }

                self.console.print(
                    Panel(
                        Syntax(json.dumps(response_json, indent=2), "json"),
                        title=f"Response Body ({len(content)} bytes)",
                        border_style=status_color,
                    )
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If not JSON, show truncated text
                text_content = content.decode("utf-8", errors="ignore")
                if len(text_content) > 500:
                    text_content = text_content[:500] + "..."

                self.console.print(
                    Panel(
                        text_content,
                        title=f"Response Body ({len(content)} bytes)",
                        border_style=status_color,
                    )
                )

    @abstractmethod
    def run_test(self, scenario_name: str) -> TestSession:
        """Run performance test for given scenario."""
        pass


class SyncEndpointTester(BaseEndpointTester):
    """Synchronous endpoint tester for simple load testing."""

    def run_test(self, scenario_name: str) -> TestSession:
        """Run synchronous load test."""
        if scenario_name not in self.config.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found in configuration")

        scenario = self.config.scenarios[scenario_name]
        enabled_endpoints = self.config.get_weighted_endpoints()

        if not enabled_endpoints:
            raise ValueError("No enabled endpoints found for testing")

        session = TestSession(
            session_id=f"sync_{scenario_name}_{int(time.time())}",
            start_time=datetime.now(),
            config={"scenario": scenario_name, "tester": "SyncEndpointTester"},
        )

        # Create weighted endpoint list for selection
        endpoint_choices = []
        for name, endpoint, weight in enabled_endpoints:
            endpoint_choices.extend([(name, endpoint)] * weight)

        total_requests = scenario.users * scenario.duration

        self.console.print(f"Starting sync test: {scenario_name}")
        self.console.print(f"Endpoints: {[name for name, _, _ in enabled_endpoints]}")
        self.console.print(f"Users: {scenario.users}, Duration: {scenario.duration}s")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task(
                f"Running {scenario_name}...", total=total_requests
            )

            # Simple sequential testing (for demonstration)
            # In production, you'd use threads or processes for concurrency
            for i in range(total_requests):
                endpoint_name, endpoint = random.choice(endpoint_choices)
                result = self.make_request(endpoint_name, endpoint)
                session.results.append(result)

                progress.update(task, advance=1)

                # Simple rate limiting
                if i > 0 and i % scenario.spawn_rate == 0:
                    time.sleep(1.0)

        session.end_time = datetime.now()
        return session


class AsyncEndpointTester(BaseEndpointTester):
    """Async endpoint tester for high concurrency testing."""

    def __init__(self, config: Config, console: Optional[Console] = None):
        super().__init__(config, console)
        self.connector = None

    async def make_async_request(
        self,
        session: aiohttp.ClientSession,
        endpoint_name: str,
        endpoint: EndpointConfig,
    ) -> TestResult:
        """Make an async request to endpoint."""
        url = self.get_full_url(endpoint)
        data = self.prepare_request_data(endpoint)

        # Prepare request arguments
        headers = dict(self.config.global_settings.headers)

        # Add authentication header if configured
        auth_header = self.config.get_auth_header()
        if auth_header:
            headers.update(auth_header)

        if endpoint.headers:
            headers.update(endpoint.headers)

        # Prepare request data
        json_data = None
        if data is not None and endpoint.method.upper() in ["POST", "PUT", "PATCH"]:
            json_data = data

        # Verbose logging - request
        if self.config.global_settings.verbose:
            self._log_request(endpoint_name, endpoint.method, url, headers, json_data)

        timeout = aiohttp.ClientTimeout(total=self.config.global_settings.timeout)

        start_time = time.time()
        success = False
        status_code = 0
        error_message = None
        response_size = 0
        response_content = b""

        try:
            async with session.request(
                endpoint.method,
                url,
                headers=headers,
                json=json_data,
                timeout=timeout,
                ssl=self.config.global_settings.verify_ssl,
            ) as response:
                status_code = response.status
                response_content = await response.read()
                response_size = len(response_content)

                success = 200 <= status_code < 400

                if not success:
                    error_message = f"HTTP {status_code}: {response.reason}"

        except Exception as e:
            error_message = str(e)

        response_time = (time.time() - start_time) * 1000

        # Verbose logging - response
        if self.config.global_settings.verbose:
            self._log_response(
                endpoint_name,
                status_code,
                response_time,
                response_content,
                error_message,
            )

        return TestResult(
            endpoint_name=endpoint_name,
            method=endpoint.method,
            url=url,
            status_code=status_code,
            response_time=response_time,
            success=success,
            error_message=error_message,
            request_size=len(json.dumps(data).encode()) if data else 0,
            response_size=response_size,
        )

    async def run_async_test(self, scenario_name: str) -> TestSession:
        """Run async load test with high concurrency."""
        if scenario_name not in self.config.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found in configuration")

        scenario = self.config.scenarios[scenario_name]
        enabled_endpoints = self.config.get_weighted_endpoints()

        if not enabled_endpoints:
            raise ValueError("No enabled endpoints found for testing")

        session = TestSession(
            session_id=f"async_{scenario_name}_{int(time.time())}",
            start_time=datetime.now(),
            config={"scenario": scenario_name, "tester": "AsyncEndpointTester"},
        )

        # Create weighted endpoint list
        endpoint_choices = []
        for name, endpoint, weight in enabled_endpoints:
            endpoint_choices.extend([(name, endpoint)] * weight)

        self.console.print(f"Starting async test: {scenario_name}")
        self.console.print(f"Concurrent users: {scenario.users}")
        self.console.print(f"Duration: {scenario.duration}s")

        # Setup connector for connection pooling
        connector = aiohttp.TCPConnector(
            limit=scenario.users * 2,  # Connection pool size
            limit_per_host=scenario.users,
        )

        async with aiohttp.ClientSession(connector=connector) as http_session:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(scenario.users)

            async def make_requests_for_duration():
                """Make requests for specified duration."""
                end_time = time.time() + scenario.duration
                tasks = []

                while time.time() < end_time:
                    # Create batch of concurrent requests
                    batch_size = min(scenario.spawn_rate, scenario.users)

                    for _ in range(batch_size):
                        if time.time() >= end_time:
                            break

                        endpoint_name, endpoint = random.choice(endpoint_choices)

                        async def limited_request(sem, ep_name, ep):
                            async with sem:
                                return await self.make_async_request(
                                    http_session, ep_name, ep
                                )

                        task = asyncio.create_task(
                            limited_request(semaphore, endpoint_name, endpoint)
                        )
                        tasks.append(task)

                    # Wait a bit before next batch
                    await asyncio.sleep(1.0)

                # Wait for all remaining tasks
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, TestResult):
                            session.results.append(result)

            await make_requests_for_duration()

        session.end_time = datetime.now()
        return session

    def run_test(self, scenario_name: str) -> TestSession:
        """Run async test (wrapper for sync interface)."""
        return asyncio.run(self.run_async_test(scenario_name))
