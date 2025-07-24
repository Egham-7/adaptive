# Standard library imports
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Callable


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, don't attempt
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class LoadAttempt:
    count: int = 0
    last_attempt: Optional[datetime] = None
    consecutive_failures: int = 0
    circuit_state: CircuitState = CircuitState.CLOSED
    last_failure: Optional[datetime] = None

    def __post_init__(self):
        if self.last_attempt is None:
            self.last_attempt = datetime.now()


class CircuitBreaker:
    """Circuit breaker pattern for model loading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        logger_callback: Optional[Callable] = None,
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self._logger_callback = logger_callback
        self.attempts: Dict[str, LoadAttempt] = {}

    def _log(self, key: str, value):
        """Log metrics if callback is available."""
        if self._logger_callback:
            self._logger_callback(key, value)

    def should_attempt_load(self, model_name: str) -> bool:
        """Check if we should attempt to load a model based on circuit breaker state."""
        if model_name not in self.attempts:
            return True

        attempt = self.attempts[model_name]

        if attempt.circuit_state == CircuitState.CLOSED:
            return True
        elif attempt.circuit_state == CircuitState.OPEN:
            # Check if timeout has passed
            if (
                attempt.last_failure
                and (datetime.now() - attempt.last_failure).total_seconds()
                > self.timeout_seconds
            ):
                attempt.circuit_state = CircuitState.HALF_OPEN
                self._log("circuit_breaker_half_open", model_name)
                return True
            return False
        elif attempt.circuit_state == CircuitState.HALF_OPEN:
            return True

        return False

    def record_success(self, model_name: str):
        """Record a successful model load."""
        if model_name not in self.attempts:
            self.attempts[model_name] = LoadAttempt()

        attempt = self.attempts[model_name]
        attempt.count += 1
        attempt.last_attempt = datetime.now()
        attempt.consecutive_failures = 0

        if attempt.circuit_state == CircuitState.HALF_OPEN:
            attempt.circuit_state = CircuitState.CLOSED
            self._log("circuit_breaker_closed", model_name)

    def record_failure(self, model_name: str):
        """Record a failed model load."""
        if model_name not in self.attempts:
            self.attempts[model_name] = LoadAttempt()

        attempt = self.attempts[model_name]
        attempt.count += 1
        attempt.last_attempt = datetime.now()
        attempt.consecutive_failures += 1
        attempt.last_failure = datetime.now()

        if (
            attempt.consecutive_failures >= self.failure_threshold
            and attempt.circuit_state == CircuitState.CLOSED
        ):
            attempt.circuit_state = CircuitState.OPEN
            self._log("circuit_breaker_opened", model_name)

    def get_failure_count(self, model_name: str) -> int:
        """Get the consecutive failure count for a model."""
        if model_name not in self.attempts:
            return 0
        return self.attempts[model_name].consecutive_failures

    def get_state(self, model_name: str) -> CircuitState:
        """Get the current circuit breaker state for a model."""
        if model_name not in self.attempts:
            return CircuitState.CLOSED
        return self.attempts[model_name].circuit_state
