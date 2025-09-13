#!/usr/bin/env python3
"""Test the /classify endpoint of the deployed Modal app"""

import jwt
import httpx
import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, List

# Configuration
MODAL_URL = "https://egham-7--nvidia-prompt-classifier-serve.modal.run"
# Note: This JWT secret must match what's configured in Modal secrets
JWT_SECRET = (
    "ByzOO6hHOfrHSF21mACgfswC8Qqm7yeNtkjf3Liwgok"  # Replace with your actual secret
)


@pytest.fixture
def auth_headers() -> Dict[str, str]:
    """Generate JWT token and return headers"""
    payload = {
        "sub": "test_user",
        "user": "claude_test",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    token_bytes = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    token_str = (
        token_bytes.decode("utf-8") if isinstance(token_bytes, bytes) else token_bytes
    )
    return {"Authorization": f"Bearer {token_str}", "Content-Type": "application/json"}


@pytest.fixture
def test_prompts() -> List[str]:
    """Test prompts for classification"""
    return [
        "Write a Python function to sort a list",
        "Summarize this article about climate change",
        "What is the capital of France?",
        "Generate a creative story about a robot",
        "Extract key information from this document",
    ]


def test_classify_endpoint(
    auth_headers: Dict[str, str], test_prompts: List[str]
) -> None:
    """Test the /classify endpoint with various prompts"""
    data = {"prompts": test_prompts}

    with httpx.Client(timeout=120) as client:  # 2 minute timeout for model loading
        response = client.post(f"{MODAL_URL}/classify", headers=auth_headers, json=data)

        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"

        result = response.json()

        # Verify all expected fields are present
        expected_fields = [
            "task_type_1",
            "task_type_2",
            "task_type_prob",
            "creativity_scope",
            "reasoning",
            "contextual_knowledge",
            "prompt_complexity_score",
            "domain_knowledge",
            "number_of_few_shots",
            "no_label_reason",
            "constraint_ct",
        ]

        for field in expected_fields:
            assert field in result, f"Missing field: {field}"
            assert len(result[field]) == len(
                test_prompts
            ), f"Field {field} has wrong length"

        # Verify task types are strings
        for task_type in result["task_type_1"]:
            assert isinstance(
                task_type, str
            ), f"task_type_1 should be string, got {type(task_type)}"

        # Verify probabilities are floats between 0 and 1
        for prob in result["task_type_prob"]:
            assert isinstance(
                prob, float
            ), f"task_type_prob should be float, got {type(prob)}"
            assert (
                0 <= prob <= 1
            ), f"task_type_prob should be between 0 and 1, got {prob}"

        # Verify complexity scores are floats
        for score in result["prompt_complexity_score"]:
            assert isinstance(
                score, float
            ), f"prompt_complexity_score should be float, got {type(score)}"
            assert (
                0 <= score <= 1
            ), f"prompt_complexity_score should be between 0 and 1, got {score}"


def test_health_endpoint() -> None:
    """Test the /health endpoint"""
    with httpx.Client() as client:
        response = client.get(f"{MODAL_URL}/health")

        assert response.status_code == 200
        result = response.json()

        assert result["status"] == "healthy"
        assert result["service"] == "nvidia-prompt-classifier"


def test_auth_required(test_prompts: List[str]) -> None:
    """Test that authentication is required for /classify"""
    data = {"prompts": test_prompts}

    with httpx.Client() as client:
        # No auth headers
        response = client.post(f"{MODAL_URL}/classify", json=data)
        assert response.status_code == 403  # Forbidden without auth


def test_invalid_auth(test_prompts: List[str]) -> None:
    """Test that invalid authentication is rejected"""
    data = {"prompts": test_prompts}
    headers = {
        "Authorization": "Bearer invalid_token",
        "Content-Type": "application/json",
    }

    with httpx.Client() as client:
        response = client.post(f"{MODAL_URL}/classify", headers=headers, json=data)
        assert response.status_code == 401  # Unauthorized with invalid token
