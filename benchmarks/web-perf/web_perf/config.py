"""Configuration management for web performance testing."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field


class TestData(BaseModel):
    """Test data for endpoint requests."""

    pass  # Will be dynamically typed based on endpoint needs


class EndpointConfig(BaseModel):
    """Configuration for a single endpoint."""

    path: str
    method: str = "GET"
    description: str = ""
    test_data: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    weight: int = Field(default=1, ge=1, le=100)
    enabled: bool = True
    headers: Optional[Dict[str, str]] = None
    auth: Optional[Dict[str, str]] = None


class ScenarioConfig(BaseModel):
    """Load testing scenario configuration."""

    users: int = Field(ge=1)
    spawn_rate: int = Field(ge=1)
    duration: int = Field(ge=1)  # seconds


class GlobalSettings(BaseModel):
    """Global test settings."""

    timeout: int = 30
    max_retries: int = 3
    verify_ssl: bool = True
    headers: Dict[str, str] = Field(default_factory=dict)
    verbose: bool = False


class ThresholdConfig(BaseModel):
    """Performance threshold configuration."""

    class ResponseTime(BaseModel):
        p50: int = 500  # milliseconds
        p95: int = 2000
        p99: int = 5000

    class RPS(BaseModel):
        minimum: int = 1
        target: int = 100

    response_time: ResponseTime = Field(default_factory=ResponseTime)
    success_rate: float = Field(default=95.0, ge=0, le=100)
    rps: RPS = Field(default_factory=RPS)


class ReportingConfig(BaseModel):
    """Reporting configuration."""

    output_dir: str = "results"
    generate_charts: bool = True
    export_formats: List[str] = Field(default_factory=lambda: ["html", "csv"])
    chart_formats: List[str] = Field(default_factory=lambda: ["png"])


class AuthConfig(BaseModel):
    """Authentication configuration."""

    type: str = "bearer"  # bearer, api_key, basic
    header: str = "Authorization"
    prefix: str = "Bearer"


class AdaptiveAuth(BaseModel):
    """Adaptive API authentication."""

    api_key: str = Field(default_factory=lambda: os.getenv("ADAPTIVE_API_KEY", ""))


class Config(BaseModel):
    """Main configuration class."""

    base_url: str
    global_settings: GlobalSettings = Field(default_factory=GlobalSettings)
    scenarios: Dict[str, ScenarioConfig]
    endpoints: Dict[str, EndpointConfig]
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    adaptive: Optional[AdaptiveAuth] = None
    default_auth: Optional[AuthConfig] = None

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        # Try to load auth configuration
        auth_path = config_path.parent / "auth.yaml"
        if auth_path.exists():
            with open(auth_path, "r") as f:
                auth_data = yaml.safe_load(f)

            # Merge auth data into main config
            if "adaptive" in auth_data:
                adaptive_config = auth_data["adaptive"]
                # If api_key is empty, try environment variable
                if not adaptive_config.get("api_key"):
                    adaptive_config["api_key"] = os.getenv("ADAPTIVE_API_KEY", "")
                data["adaptive"] = adaptive_config
            if "default_auth" in auth_data:
                data["default_auth"] = auth_data["default_auth"]

        return cls(**data)

    def get_enabled_endpoints(self) -> Dict[str, EndpointConfig]:
        """Get only enabled endpoints."""
        return {
            name: endpoint
            for name, endpoint in self.endpoints.items()
            if endpoint.enabled
        }

    def get_weighted_endpoints(self) -> List[tuple[str, EndpointConfig, int]]:
        """Get enabled endpoints with their weights."""
        return [
            (name, endpoint, endpoint.weight)
            for name, endpoint in self.get_enabled_endpoints().items()
        ]

    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check if at least one endpoint is enabled
        enabled_endpoints = self.get_enabled_endpoints()
        if not enabled_endpoints:
            issues.append("No endpoints are enabled for testing")

        # Check weight totals
        total_weight = sum(ep.weight for ep in enabled_endpoints.values())
        if total_weight == 0:
            issues.append("Total endpoint weights cannot be zero")

        # Validate scenarios
        for name, scenario in self.scenarios.items():
            if scenario.spawn_rate > scenario.users:
                issues.append(f"Scenario '{name}': spawn_rate cannot exceed users")

        return issues

    def get_auth_header(self) -> Optional[Dict[str, str]]:
        """Get authorization header if auth is configured."""
        if not self.adaptive or not self.adaptive.api_key:
            return None

        auth_config = self.default_auth or AuthConfig()

        if auth_config.type == "bearer":
            return {auth_config.header: f"{auth_config.prefix} {self.adaptive.api_key}"}
        elif auth_config.type == "api_key":
            if auth_config.prefix:
                return {
                    auth_config.header: f"{auth_config.prefix} {self.adaptive.api_key}"
                }
            else:
                return {auth_config.header: self.adaptive.api_key}

        return None


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from file."""
    return Config.from_file(config_path)
