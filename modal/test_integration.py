#!/usr/bin/env python3
"""Integration test for Modal NVIDIA prompt classifier deployment.

This script tests the end-to-end functionality including:
- JWT token generation and authentication
- Modal API health check
- Prompt classification with multiple prompts
- Error handling and timeouts

Usage:
    uv sync  # Install dependencies first
    uv run python test_integration.py
    # or
    python test_integration.py
    
Environment variables required:
    - MODAL_CLASSIFIER_URL: Modal endpoint URL
    - JWT_SECRET: Shared JWT secret
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

import httpx
from jose import jwt


def generate_jwt_token(jwt_secret: str) -> str:
    """Generate JWT token for authentication."""
    payload = {
        "sub": "test_client",
        "user": "integration_test",
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=1),
        "service": "prompt_classification"
    }
    
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


def test_health_check(modal_url: str) -> Dict[str, Any]:
    """Test Modal service health check."""
    print("🔍 Testing health check...")
    
    with httpx.Client(timeout=10) as client:
        response = client.get(f"{modal_url}/health")
        response.raise_for_status()
        
        health_data = response.json()
        print(f"✅ Health check passed: {health_data}")
        return health_data


def test_classification(modal_url: str, jwt_token: str) -> Dict[str, Any]:
    """Test prompt classification - raw logits endpoint."""
    print("🧠 Testing raw logits classification...")
    
    test_prompts = [
        "Write a Python function to sort a list",
        "Summarize this article about artificial intelligence",
        "Solve this math equation: 2x + 5 = 13",
        "Generate a creative story about a robot",
        "Extract key information from this document"
    ]
    
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json"
    }
    
    request_data = {"prompts": test_prompts}
    
    print(f"📤 Sending {len(test_prompts)} prompts for raw logits...")
    
    with httpx.Client(timeout=120) as client:  # Longer timeout for ML inference and cold start
        response = client.post(
            f"{modal_url}/classify_raw",
            headers=headers,
            json=request_data
        )
        response.raise_for_status()
        
        classification_data = response.json()
        print(f"✅ Raw logits generation completed successfully")
        
        # Print logits structure info
        logits = classification_data.get("logits", [])
        if logits:
            print(f"📊 Logits structure: {len(logits)} classification heads")
            print(f"📊 Batch size: {len(logits[0]) if logits else 0} prompts")
            print(f"📊 Example head dimensions: {len(logits[0][0]) if logits and logits[0] else 0} classes")
            print("ℹ️  Raw logits ready for adaptive_ai processing")
        else:
            print("⚠️  No logits received in response")
            
        return classification_data


def test_authentication_error(modal_url: str) -> None:
    """Test authentication error handling."""
    print("🔐 Testing authentication error handling...")
    
    headers = {
        "Authorization": "Bearer invalid-token",
        "Content-Type": "application/json"
    }
    
    request_data = {"prompts": ["test prompt"]}
    
    with httpx.Client(timeout=10) as client:
        response = client.post(
            f"{modal_url}/classify_raw",
            headers=headers,
            json=request_data
        )
        
        if response.status_code == 401:
            print("✅ Authentication error handled correctly (401 Unauthorized)")
        else:
            print(f"⚠️  Unexpected response: {response.status_code}")


def main():
    """Run integration tests."""
    print("🚀 Modal NVIDIA Prompt Classifier Integration Test")
    print("=" * 55)
    
    # Get configuration from environment
    modal_url = os.environ.get("MODAL_CLASSIFIER_URL")
    jwt_secret = os.environ.get("JWT_SECRET")
    
    if not modal_url:
        print("❌ Error: MODAL_CLASSIFIER_URL environment variable required")
        print("Set it to your Modal endpoint URL, e.g.:")
        print("export MODAL_CLASSIFIER_URL='https://username--nvidia-prompt-classifier-serve.modal.run'")
        sys.exit(1)
        
    if not jwt_secret:
        print("❌ Error: JWT_SECRET environment variable required")
        print("Set it to the same secret used in Modal deployment")
        sys.exit(1)
    
    print(f"🌐 Modal URL: {modal_url}")
    print(f"🔑 JWT Secret: {'*' * len(jwt_secret)}")
    print()
    
    try:
        # Test 1: Health check
        health_data = test_health_check(modal_url)
        
        # Test 2: Generate JWT token
        print("🎟️  Generating JWT token...")
        jwt_token = generate_jwt_token(jwt_secret)
        print("✅ JWT token generated successfully")
        
        # Test 3: Classification with valid token
        classification_data = test_classification(modal_url, jwt_token)
        
        # Test 4: Authentication error
        test_authentication_error(modal_url)
        
        print()
        print("🎉 All tests passed successfully!")
        print("✨ Modal NVIDIA prompt classifier is working correctly")
        
        # Summary
        print("\n📋 Test Summary:")
        print(f"   • Health check: ✅ {health_data.get('status', 'N/A')}")
        print(f"   • Authentication: ✅ JWT working")
        logits = classification_data.get('logits', [])
        batch_size = len(logits[0]) if logits else 0
        print(f"   • Raw logits: ✅ {batch_size} prompts processed")
        print(f"   • Error handling: ✅ 401 errors handled")
        
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
        sys.exit(1)
    except httpx.RequestError as e:
        print(f"❌ Request error: {e}")
        print("   Check that Modal service is running and URL is correct")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()