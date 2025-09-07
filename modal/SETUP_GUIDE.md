# Modal Deployment Setup Guide

This guide provides step-by-step instructions for setting up and deploying the NVIDIA prompt classifier to Modal.

## Prerequisites Checklist

- [ ] Modal account created at https://modal.com
- [ ] Python 3.11+ installed
- [ ] uv package manager installed
- [ ] Repository cloned and in `modal/` directory

## Step 1: Install Dependencies

```bash
cd modal/

# Install Modal CLI and dependencies
uv sync
```

## Step 2: Modal Authentication

```bash
# One-time Modal authentication (opens browser)
modal setup
```

This will:
- Open your browser to authenticate with Modal
- Save authentication credentials locally
- Verify your Modal account access

## Step 3: Generate JWT Secret

You need a strong, random JWT secret for secure authentication between adaptive_ai and Modal.

### Option A: Generate with Python
```bash
# Generate a secure random JWT secret
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Option B: Generate with OpenSSL
```bash
# Generate a secure random JWT secret
openssl rand -base64 32
```

### Option C: Use a Strong Manual Key
```bash
# Example strong key (DO NOT use this exact one)
export JWT_SECRET="your-super-secure-random-jwt-key-at-least-32-chars"
```

**Important**: Save this key securely - you'll need it for both Modal and adaptive_ai configuration.

## Step 4: Create Modal Secrets

```bash
# Create JWT secret in Modal (replace with your actual key)
modal secret create jwt-auth JWT_SECRET="your-super-secure-random-jwt-key-at-least-32-chars"

# Optional: Create HuggingFace secret if needed (usually not required for NVIDIA model)
# modal secret create huggingface HF_TOKEN="hf_your_token_here"
```

Verify secrets were created:
```bash
modal secret list
```

You should see:
```
jwt-auth
```

## Step 5: Deploy to Modal

```bash
# Deploy the NVIDIA classifier to Modal
modal deploy app.py
```

This will:
- Build the Modal container image with all dependencies
- Download the NVIDIA prompt classifier model (~2GB)
- Deploy the service with GPU acceleration
- Provide you with a Modal endpoint URL

**Expected output:**
```
✓ Created objects.
├── 🔨 Created mount /Users/you/modal/pyproject.toml
├── 🔨 Created nvidia-prompt-classifier => https://username--nvidia-prompt-classifier-serve.modal.run
└── 🔨 Created NvidiaPromptClassifier => https://username--nvidia-prompt-classifier.modal.run

✓ App deployed! 🎉

View Deployment: https://modal.com/apps/your-app-id
```

**Save the endpoint URL** - you'll need it for adaptive_ai configuration.

## Step 6: Test Modal Deployment

```bash
# Test health endpoint (no authentication required)
curl -X GET "https://YOUR-MODAL-URL/health"
```

Expected response:
```json
{
  "status": "healthy",
  "model": "nvidia/prompt-task-and-complexity-classifier", 
  "gpu": "T4",
  "service": "nvidia-prompt-classifier"
}
```

## Step 7: Configure adaptive_ai Service

Add these environment variables to your adaptive_ai service:

```bash
# Add to your adaptive_ai/.env file or export in shell
export MODAL_CLASSIFIER_URL="https://YOUR-MODAL-URL"  # Replace with actual URL
export JWT_SECRET="your-super-secure-random-jwt-key-at-least-32-chars"  # Same as Modal secret
```

Optional configuration:
```bash
# Request timeout (default: 30 seconds)
export MODAL_REQUEST_TIMEOUT="30"

# Max retries (default: 3)  
export MODAL_MAX_RETRIES="3"

# Retry delay (default: 1.0 seconds)
export MODAL_RETRY_DELAY="1.0"
```

## Step 8: Test End-to-End Integration

```bash
# Set environment variables
export MODAL_CLASSIFIER_URL="https://YOUR-MODAL-URL"
export JWT_SECRET="your-super-secure-random-jwt-key-at-least-32-chars"

# Run integration test
uv run python test_integration.py
```

Expected output:
```
🚀 Modal NVIDIA Prompt Classifier Integration Test
===============================================
🌐 Modal URL: https://your-modal-url
🔑 JWT Secret: ********************************
🔍 Testing health check...
✅ Health check passed: {'status': 'healthy', ...}
🎟️  Generating JWT token...
✅ JWT token generated successfully
🧠 Testing prompt classification...
📤 Sending 5 prompts for classification...
✅ Classification completed successfully
📊 Sample task types: ['Code Generation', 'Summarization', ...]
🔐 Testing authentication error handling...
✅ Authentication error handled correctly (401 Unauthorized)

🎉 All tests passed successfully!
✨ Modal NVIDIA prompt classifier is working correctly
```

## Step 9: Update adaptive_ai Service

Your adaptive_ai service should now automatically use the Modal API client when the environment variables are set. Test by running your adaptive_ai service and checking the logs:

```bash
# Check adaptive_ai logs for Modal client usage
# You should see messages like:
# "Initialized PromptClassifier with Modal API client"
# "Modal service health check passed"
```

## Troubleshooting

### Common Issues

**1. "Secret not found" error**
```bash
# List existing secrets
modal secret list

# Recreate if missing
modal secret create jwt-auth JWT_SECRET="your-secret"
```

**2. Modal authentication fails**
```bash
# Re-authenticate
modal setup --force
```

**3. Deployment takes long time**
- First deployment downloads ~2GB model (takes 5-10 minutes)
- Subsequent deployments are faster due to caching

**4. Health check fails**
- Wait a few minutes after deployment for service to be ready
- Check Modal dashboard for errors: https://modal.com/dashboard

**5. JWT authentication errors (500 - "JWT secret not configured")**

If you get this error during testing, the FastAPI app can't access the Modal secret:

```bash
# Step 1: Recreate the JWT secret (if you deleted it accidentally)
modal secret create jwt-auth JWT_SECRET="your-actual-jwt-secret-key"

# Step 2: Verify secret exists
modal secret list

# Step 3: Ensure the serve() function has secrets access
# Check that app.py line 360 has: 
# @app.function(image=image, secrets=[modal.Secret.from_name("jwt-auth")])

# Step 4: Redeploy the app
modal deploy app.py

# Step 5: Test with the EXACT same JWT secret
export JWT_SECRET="your-actual-jwt-secret-key"  # MUST match Modal secret
export MODAL_CLASSIFIER_URL="https://YOUR-MODAL-URL"
uv run python test_integration.py
```

- Ensure JWT_SECRET is identical in Modal and adaptive_ai  
- Verify secret is properly set with `modal secret list`
- Check that JWT_SECRET environment variable is set in adaptive_ai

### Debug Commands

```bash
# Check Modal app status
modal app list

# View deployment logs  
modal logs nvidia-prompt-classifier

# View real-time logs
modal logs nvidia-prompt-classifier --follow

# Check Modal dashboard
# Visit: https://modal.com/dashboard
```

## Security Notes

1. **Use a strong JWT secret**: At least 32 characters, cryptographically random
2. **Keep secrets secure**: Don't commit JWT secrets to version control
3. **Rotate secrets regularly**: Update JWT secret periodically for security
4. **Monitor access**: Use Modal dashboard to monitor API usage

## Cost Optimization

1. **Container idle timeout**: Set to 300 seconds (5 minutes) in app.py
2. **GPU selection**: T4 is cost-effective for DeBERTa-v3-base
3. **Batch requests**: Send multiple prompts per request when possible
4. **Monitor usage**: Check Modal dashboard for cost tracking

## Next Steps

1. **Monitor performance**: Check Modal dashboard for latency and usage
2. **Scale if needed**: Upgrade GPU (T4 → A10G → A100) if more throughput required
3. **Integrate with applications**: Use the adaptive_ai service in your applications
4. **Set up alerts**: Configure Modal alerts for errors or high usage

Your NVIDIA prompt classifier is now deployed on Modal with GPU acceleration! 🚀