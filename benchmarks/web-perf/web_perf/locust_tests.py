"""Locust-based load testing with multi-endpoint support."""

import random
from typing import Dict, List, Tuple, Optional, Any
from locust import HttpUser, task
from locust.env import Environment
from locust.stats import stats_printer, stats_history

from web_perf.config import Config, EndpointConfig, load_config


class AdaptiveEndpointUser(HttpUser):
    """Locust user class that tests multiple endpoints based on configuration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config: Optional[Config] = None
        self.weighted_endpoints: List[Tuple[str, EndpointConfig, int]] = []
        self.endpoint_choices: List[Tuple[str, EndpointConfig]] = []

    def on_start(self):
        """Initialize user with configuration."""
        # Load config from environment or use default
        config_path = getattr(self.environment, "config_path", "config/endpoints.yaml")
        self.config = load_config(config_path)

        # Setup weighted endpoints
        self.weighted_endpoints = self.config.get_weighted_endpoints()

        # Create weighted endpoint choices for random selection
        self.endpoint_choices = []
        for name, endpoint, weight in self.weighted_endpoints:
            self.endpoint_choices.extend([(name, endpoint)] * weight)

        # Set host from config if not already set
        if not self.host:
            self.host = self.config.base_url

        # Update client settings - Note: timeout handled by requests session
        self.client.verify = self.config.global_settings.verify_ssl

        # Set global headers
        self.client.headers.update(self.config.global_settings.headers)

    def get_request_data(self, endpoint: EndpointConfig) -> Optional[Dict[str, Any]]:
        """Get request data for endpoint."""
        if endpoint.test_data is None:
            return None

        if isinstance(endpoint.test_data, list):
            return random.choice(endpoint.test_data)
        return endpoint.test_data

    @task
    def test_weighted_endpoints(self):
        """Test endpoints based on their configured weights."""
        if not self.endpoint_choices:
            return

        # Select random endpoint based on weights
        endpoint_name, endpoint = random.choice(self.endpoint_choices)

        # Prepare request data
        data = self.get_request_data(endpoint)

        # Prepare request kwargs
        kwargs: Dict[str, Any] = {}
        if endpoint.headers:
            kwargs["headers"] = {**dict(self.client.headers), **endpoint.headers}

        # Make request based on method
        method = endpoint.method.upper()
        url = endpoint.path

        # Add name for better Locust reporting
        name = f"{method} {endpoint_name}"

        try:
            if method == "GET":
                self.client.get(url, name=name, **kwargs)
            elif method == "POST":
                if data:
                    kwargs["json"] = data
                self.client.post(url, name=name, **kwargs)
            elif method == "PUT":
                if data:
                    kwargs["json"] = data
                self.client.put(url, name=name, **kwargs)
            elif method == "DELETE":
                self.client.delete(url, name=name, **kwargs)
            elif method == "PATCH":
                if data:
                    kwargs["json"] = data
                self.client.patch(url, name=name, **kwargs)
            else:
                # Generic request
                if data:
                    kwargs["json"] = data
                self.client.request(method, url, name=name, **kwargs)

        except Exception:
            # Locust will handle the exception reporting
            raise


class ConfigurableLocustRunner:
    """Runner for Locust tests with configuration support."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_config(config_path)

    def run_scenario(
        self,
        scenario_name: str,
        headless: bool = True,
        html_report: Optional[str] = None,
    ):
        """Run a specific scenario using Locust."""
        if scenario_name not in self.config.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        scenario = self.config.scenarios[scenario_name]

        # Setup Locust environment
        env = Environment(
            user_classes=[AdaptiveEndpointUser], host=self.config.base_url
        )

        # Store config path in environment for users to access
        setattr(env, "config_path", self.config_path)

        if headless:
            # Run headless (programmatic)
            env.create_local_runner()

            # Start users
            if env.runner:
                env.runner.start(scenario.users, spawn_rate=scenario.spawn_rate)

            # Setup stats printing
            import gevent

            gevent.spawn(stats_printer(env.stats))
            gevent.spawn(stats_history, env.runner)

            # Run for specified duration
            gevent.sleep(scenario.duration)

            # Stop users
            if env.runner:
                env.runner.stop()

            # Generate HTML report if requested
            if html_report:
                from locust.html import get_html_report

                with open(html_report, "w") as f:
                    f.write(get_html_report(env))

            return env.stats
        else:
            # Run with web UI
            env.create_web_ui(host="127.0.0.1", port=8091)
            if env.web_ui:
                env.web_ui.start()

            print("Locust web UI available at http://127.0.0.1:8091")
            print(f"Scenario: {scenario_name}")
            print(
                f"Recommended settings: {scenario.users} users, {scenario.spawn_rate} spawn rate"
            )

            # Keep running until interrupted
            try:
                if env.runner and hasattr(env.runner, "greenlet"):
                    env.runner.greenlet.join()
            except KeyboardInterrupt:
                pass

    def validate_config(self) -> bool:
        """Validate configuration before running tests."""
        issues = self.config.validate_configuration()
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False

        print("Configuration validation passed")
        return True

    def list_scenarios(self):
        """List available scenarios."""
        print("Available scenarios:")
        for name, scenario in self.config.scenarios.items():
            print(
                f"  {name}: {scenario.users} users, {scenario.spawn_rate} spawn rate, {scenario.duration}s duration"
            )

    def list_endpoints(self):
        """List configured endpoints."""
        print("Configured endpoints:")
        for name, endpoint in self.config.endpoints.items():
            status = "✓ enabled" if endpoint.enabled else "✗ disabled"
            print(
                f"  {name} ({status}): {endpoint.method} {endpoint.path} (weight: {endpoint.weight})"
            )


def create_locust_file(config_path: str, output_path: str = "locustfile.py"):
    """Create a standalone locustfile.py for use with locust command."""

    # Read the current module content

    locustfile_content = f'''"""
Auto-generated Locustfile for web performance testing.
Generated from config: {config_path}

Run with: locust -f {output_path}
"""

import json
import random
from typing import Dict, List, Tuple
from locust import HttpUser, task, between
from pathlib import Path

# Embedded configuration loading (simplified)
import yaml
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union


class EndpointConfig(BaseModel):
    path: str
    method: str = "GET"
    description: str = ""
    test_data: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    weight: int = Field(default=1, ge=1, le=100)
    enabled: bool = True
    headers: Optional[Dict[str, str]] = None


class GlobalSettings(BaseModel):
    timeout: int = 30
    max_retries: int = 3
    verify_ssl: bool = True
    headers: Dict[str, str] = Field(default_factory=dict)


class Config(BaseModel):
    base_url: str
    global_settings: GlobalSettings = Field(default_factory=GlobalSettings)
    endpoints: Dict[str, EndpointConfig]
    
    @classmethod
    def from_file(cls, config_path):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def get_enabled_endpoints(self):
        return {{
            name: endpoint 
            for name, endpoint in self.endpoints.items() 
            if endpoint.enabled
        }}
    
    def get_weighted_endpoints(self):
        return [
            (name, endpoint, endpoint.weight)
            for name, endpoint in self.get_enabled_endpoints().items()
        ]


# Load configuration
CONFIG_PATH = "{config_path}"
config = Config.from_file(CONFIG_PATH)


class AdaptiveEndpointUser(HttpUser):
    """Locust user for testing configured endpoints."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    host = config.base_url
    
    def on_start(self):
        # Setup weighted endpoints
        self.weighted_endpoints = config.get_weighted_endpoints()
        
        # Create weighted endpoint choices
        self.endpoint_choices = []
        for name, endpoint, weight in self.weighted_endpoints:
            self.endpoint_choices.extend([(name, endpoint)] * weight)
        
        # Set client settings
        self.client.timeout = config.global_settings.timeout
        self.client.verify = config.global_settings.verify_ssl
        self.client.headers.update(config.global_settings.headers)
    
    def get_request_data(self, endpoint):
        if endpoint.test_data is None:
            return None
        
        if isinstance(endpoint.test_data, list):
            return random.choice(endpoint.test_data)
        return endpoint.test_data
    
    @task
    def test_endpoints(self):
        if not self.endpoint_choices:
            return
        
        endpoint_name, endpoint = random.choice(self.endpoint_choices)
        data = self.get_request_data(endpoint)
        
        kwargs = {{}}
        if endpoint.headers:
            kwargs['headers'] = {{**self.client.headers, **endpoint.headers}}
        
        method = endpoint.method.upper()
        url = endpoint.path
        name = f"{{method}} {{endpoint_name}}"
        
        if method == "GET":
            response = self.client.get(url, name=name, **kwargs)
        elif method == "POST":
            if data:
                kwargs['json'] = data
            response = self.client.post(url, name=name, **kwargs)
        elif method in ["PUT", "PATCH", "DELETE"]:
            if data and method != "DELETE":
                kwargs['json'] = data
            response = self.client.request(method, url, name=name, **kwargs)
'''

    with open(output_path, "w") as f:
        f.write(locustfile_content)

    print(f"Locustfile created: {output_path}")
    print(f"Run with: locust -f {output_path}")
