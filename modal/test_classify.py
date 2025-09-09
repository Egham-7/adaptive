#!/usr/bin/env python3
"""Test the /classify endpoint of the deployed Modal app"""

import jwt
import httpx
from datetime import datetime, timedelta, timezone

# Configuration
MODAL_URL = "https://egham-7--nvidia-prompt-classifier-serve.modal.run"
# Note: This JWT secret must match what's configured in Modal secrets
JWT_SECRET = "ByzOO6hHOfrHSF21mACgfswC8Qqm7yeNtkjf3Liwgok"  # Replace with your actual secret

# Generate JWT token
payload = {
    "sub": "test_user",
    "user": "claude_test",
    "iat": datetime.now(timezone.utc),
    "exp": datetime.now(timezone.utc) + timedelta(hours=1)
}

token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
print(f"🔑 Generated JWT token: {token[:50]}...")

# Test prompts
test_prompts = [
    "Write a Python function to sort a list",
    "Summarize this article about climate change",
    "What is the capital of France?",
    "Generate a creative story about a robot",
    "Extract key information from this document"
]

# Make request
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

data = {"prompts": test_prompts}

print(f"\n🚀 Testing /classify endpoint...")
print(f"📍 URL: {MODAL_URL}/classify")
print(f"📝 Sending {len(test_prompts)} prompts for classification\n")

try:
    with httpx.Client(timeout=120) as client:  # 2 minute timeout for model loading
        response = client.post(
            f"{MODAL_URL}/classify",
            headers=headers,
            json=data
        )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Classification successful!")
            print("\n🔍 Results:")
            
            # Display results in a nice format
            for i, prompt in enumerate(test_prompts):
                print(f"\nPrompt {i+1}: '{prompt}'")
                print(f"  • Task Type: {result['task_type_1'][i]} (confidence: {result['task_type_prob'][i]:.3f})")
                print(f"  • Secondary: {result['task_type_2'][i]}")
                print(f"  • Complexity: {result['prompt_complexity_score'][i]:.3f}")
                print(f"  • Reasoning: {result['reasoning'][i]:.3f}")
                print(f"  • Creativity: {result['creativity_scope'][i]:.3f}")
                
        else:
            print(f"\n❌ Error: {response.text}")
            
except httpx.ReadTimeout:
    print("\n⏱️ Request timed out. The model might be loading for the first time.")
    print("   Try again in a minute.")
except Exception as e:
    print(f"\n❌ Error: {e}")