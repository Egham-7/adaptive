#!/usr/bin/env python3
"""
Simple test script for Adaptive API routing.

This script tests the basic Adaptive API request with model_router
to help debug the exact format needed.

Usage:
    uv run python test_adaptive_simple.py
"""

import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
load_dotenv(".env.local")

# Get credentials
api_key = os.getenv("ADAPTIVE_API_KEY")
api_base = os.getenv("ADAPTIVE_API_BASE", "https://api.llmadaptive.uk/v1")

print("=" * 70)
print("Testing Adaptive API with model_router")
print("=" * 70)
print(f"\nAPI Base: {api_base}")
print(f"API Key: {api_key[:20]}..." if api_key else "API Key: NOT FOUND")
print()

if not api_key:
    print("ERROR: ADAPTIVE_API_KEY not found in environment")
    exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=api_key, base_url=api_base)

# Test prompt
prompt = "Write a Python function to add two numbers"

# Models to route between
models = [
    "anthropic:claude-sonnet-4-5-20250929",
    "zai:glm-4.6",
]

print("Models for routing:")
for model in models:
    print(f"  - {model}")
print()

# Test 1: With empty model and extra_body
print("-" * 70)
print("TEST 1: Empty model with extra_body")
print("-" * 70)

print("\nRequest:")
request_data = {
    "model": "",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.8,
    "max_tokens": 100,
    "extra_body": {
        "model_router": {
            "models": models,
            "cost_bias": 0.5,
        }
    },
}
print(json.dumps(request_data, indent=2))

response = client.chat.completions.create(
    model="",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.8,
    max_tokens=100,
    extra_body={
        "model_router": {
            "models": models
        }
    },
)

print("\n✅ SUCCESS!")
print(f"\nResponse:")
print(f"  Model used: {response.model}")
print(f"  Content: {response.choices[0].message.content[:100]}...")
print(f"  Tokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")







print("\n" + "=" * 70)
print("Testing complete!")
print("=" * 70)
