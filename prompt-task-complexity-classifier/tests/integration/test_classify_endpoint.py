#!/usr/bin/env python3
"""Test the /classify endpoint of the deployed Modal app"""


import jwt
import httpx
import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from prompt_task_complexity_classifier.config import get_config


@pytest.fixture
def auth_headers() -> Dict[str, str]:
    """Generate JWT token and return headers"""
    config = get_config()

    payload = {
        "sub": config.test.test_subject,
        "user": config.test.test_user,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc)
        + timedelta(hours=config.auth.token_expiry_hours),
    }
    token_bytes = jwt.encode(
        payload, config.auth.jwt_secret, algorithm=config.auth.algorithm
    )
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
    config = get_config()
    data = {"prompts": test_prompts}

    with httpx.Client(timeout=config.service.timeout) as client:
        response = client.post(
            f"{config.service.modal_url}/classify", headers=auth_headers, json=data
        )

        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"

        result = response.json()

        # API returns a list of ClassificationResult objects (one per prompt)
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) == len(
            test_prompts
        ), f"Expected {len(test_prompts)} results, got {len(result)}"

        # Verify all expected fields are present in each result
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

        for i, classification_result in enumerate(result):
            assert isinstance(
                classification_result, dict
            ), f"Result {i} should be dict, got {type(classification_result)}"

            # Check all expected fields are present
            for field in expected_fields:
                assert (
                    field in classification_result
                ), f"Missing field '{field}' in result {i}"

            # Verify task types are strings
            task_type_1 = classification_result["task_type_1"]
            assert isinstance(
                task_type_1, str
            ), f"task_type_1 should be string, got {type(task_type_1)}"

            # Verify probabilities are floats between 0 and 1
            task_type_prob = classification_result["task_type_prob"]
            assert isinstance(
                task_type_prob, float
            ), f"task_type_prob should be float, got {type(task_type_prob)}"
            assert (
                0 <= task_type_prob <= 1
            ), f"task_type_prob should be between 0 and 1, got {task_type_prob}"

            # Verify complexity scores are floats between 0 and 1
            complexity_score = classification_result["prompt_complexity_score"]
            assert isinstance(
                complexity_score, float
            ), f"prompt_complexity_score should be float, got {type(complexity_score)}"
            assert (
                0 <= complexity_score <= 1
            ), f"prompt_complexity_score should be between 0 and 1, got {complexity_score}"


def test_health_endpoint() -> None:
    """Test the /health endpoint"""
    config = get_config()

    with httpx.Client() as client:
        response = client.get(f"{config.service.modal_url}/health")

        assert response.status_code == 200
        result = response.json()

        assert result["status"] == "healthy"
        assert result["service"] == config.service.name


def test_auth_required(test_prompts: List[str]) -> None:
    """Test that authentication is required for /classify"""
    config = get_config()
    data = {"prompts": test_prompts}

    with httpx.Client() as client:
        # No auth headers
        response = client.post(f"{config.service.modal_url}/classify", json=data)
        assert response.status_code == 403  # Forbidden without auth


def test_invalid_auth(test_prompts: List[str]) -> None:
    """Test that invalid authentication is rejected"""
    config = get_config()
    data = {"prompts": test_prompts}
    headers = {
        "Authorization": "Bearer invalid_token",
        "Content-Type": "application/json",
    }

    with httpx.Client() as client:
        response = client.post(
            f"{config.service.modal_url}/classify", headers=headers, json=data
        )
        assert response.status_code == 401  # Unauthorized with invalid token
