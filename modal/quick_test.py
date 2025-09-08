#!/usr/bin/env python3
"""Quick test with single prompt"""
import os
import httpx
from jose import jwt
from datetime import datetime, timedelta

# Generate token
payload = {
    "sub": "test_client",
    "user": "quick_test", 
    "iat": datetime.utcnow(),
    "exp": datetime.utcnow() + timedelta(hours=1),
}
jwt_secret = "ByzOO6hHOfrHSF21mACgfswC8Qqm7yeNtkjf3Liwgok"
token = jwt.encode(payload, jwt_secret, algorithm="HS256")
print(f"🔑 Generated JWT token: {token[:50]}...")
print(f"🔑 Using JWT secret: {jwt_secret[:10]}...{jwt_secret[-10:]}")

# Test with single prompt - using new raw logits endpoint
url = "https://egham-7--nvidia-prompt-classifier-serve.modal.run/classify_raw"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}
data = {"prompts": ["Write a simple Python function"]}

print("🧪 Testing single prompt raw logits...")
print(f"🔗 URL: {url}")

try:
    with httpx.Client(timeout=180) as client:  # 3 minute timeout
        response = client.post(url, headers=headers, json=data)
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text[:500]}...")  # Truncate long logits output
        response.raise_for_status()
        result = response.json()
        logits = result.get("logits", [])
        print(f"✅ Success! Received raw logits for {len(data['prompts'])} prompt(s)")
        print(f"📊 Logits structure: {len(logits)} classification heads")
        if logits:
            print(f"📊 First head shape: {len(logits[0])} samples x {len(logits[0][0]) if logits[0] else 0} classes")
        print("ℹ️ Note: Raw logits need to be processed by adaptive_ai for final results")
except Exception as e:
    print(f"❌ Error: {e}")