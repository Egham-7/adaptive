# Modal Deployment - NVIDIA Prompt Classifier

This directory contains the Modal deployment for the NVIDIA prompt classifier, which provides GPU-accelerated prompt classification with JWT authentication.

## Overview

The NVIDIA `prompt-task-and-complexity-classifier` model has been moved from local GPU inference to Modal's cloud infrastructure for better scalability and cost optimization.

### Architecture
```
adaptive_ai (local) --JWT--> Modal (GPU) --NVIDIA DeBERTa-v3--> Classification Results
```

### Benefits
- **GPU Acceleration**: NVIDIA T4 GPU for fast DeBERTa-v3-base inference
- **Auto-scaling**: Scales to zero when not in use, scales up on demand
- **Cost Optimization**: Pay only for GPU compute time used
- **Security**: JWT authentication between services
- **Performance**: Modal's optimized GPU infrastructure

## Prerequisites

1. **Modal Account**: Sign up at https://modal.com
2. **Python 3.11+**: Required for Modal CLI  
3. **uv**: Fast Python package manager (consistent with repo)
4. **Shared JWT Secret**: Same secret used by adaptive_ai service

## Quick Start

> 📋 **New to Modal?** See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed step-by-step instructions
> 
> ✅ **Ready to deploy?** Use [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) to verify everything

### 1. Install Dependencies with uv
```bash
# Install core dependencies (Modal CLI, FastAPI, etc.)
uv sync

# Optional: Install ML dependencies for local development  
# (Note: Heavy ML dependencies are automatically installed in Modal container)
uv sync --group ml
```

**Note**: The heavy ML dependencies (PyTorch, Transformers) are automatically installed in the Modal container. For local development, you only need the core dependencies unless you want to run the model locally.

### 2. Authenticate with Modal
```bash
modal setup
```
Follow the prompts to authenticate with your Modal account.

### 3. Create Secrets
```bash
# JWT secret for authentication (use a strong, random key)
modal secret create jwt-auth JWT_SECRET="your-super-secure-secret-key-here"

# Optional: HuggingFace token (usually not needed for NVIDIA model)
# modal secret create huggingface HF_TOKEN="hf_your_token_here"
```

**Important**: Use the same `JWT_SECRET` value in your adaptive_ai service environment.

### 4. Deploy to Modal
```bash
# Development deployment (auto-reloads on changes)
modal serve app.py

# Production deployment (persistent)
modal deploy app.py
```

Modal will provide a URL like: `https://username--nvidia-prompt-classifier-serve.modal.run`

### 5. Configure adaptive_ai Service
Add these environment variables to your adaptive_ai service:

```bash
# Modal endpoint URL (replace with your actual URL)
MODAL_CLASSIFIER_URL="https://your-username--nvidia-prompt-classifier-serve.modal.run"

# JWT secret (same as Modal secret)
JWT_SECRET="your-super-secure-secret-key-here"

# Optional: Request timeout and retry settings
MODAL_REQUEST_TIMEOUT="30"
MODAL_MAX_RETRIES="3"
MODAL_RETRY_DELAY="1.0"
```

## Testing

### Health Check
```bash
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

### Classification Test
```bash
# Test classification with JWT token
MODAL_URL="https://YOUR-MODAL-URL"
JWT_SECRET="your-super-secure-secret-key-here"

# Generate test JWT token (using Python)
python3 -c "
import jwt
from datetime import datetime, timedelta

token = jwt.encode({
    'sub': 'adaptive_ai_service',
    'user': 'adaptive_ai', 
    'exp': datetime.utcnow() + timedelta(hours=1),
    'iat': datetime.utcnow()
}, '$JWT_SECRET', algorithm='HS256')
print(token)
" > token.txt

TOKEN=$(cat token.txt)

# Test classification
curl -X POST "$MODAL_URL/classify" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "Write a Python function to sort a list", 
      "Summarize this article about AI"
    ]
  }'
```

## API Documentation

### Endpoints

#### `GET /health`
- **Purpose**: Health check
- **Authentication**: None required
- **Response**: Service status and model information

#### `POST /classify`
- **Purpose**: Classify prompts using NVIDIA model
- **Authentication**: JWT Bearer token required
- **Request Body**:
  ```json
  {
    "prompts": ["prompt1", "prompt2", ...]
  }
  ```
- **Response**: Classification results with task types, complexity scores, etc.

### Authentication

Uses JWT Bearer tokens in the Authorization header:
```
Authorization: Bearer <jwt_token>
```

JWT payload should include:
- `sub` or `user`: Service/user identifier
- `exp`: Expiration timestamp
- `iat`: Issued at timestamp

## GPU Configuration

### Current Setup
- **GPU**: NVIDIA T4 (16GB VRAM)
- **Model**: DeBERTa-v3-base (~1.4GB)
- **Batch Size**: Configurable, optimized for T4

### Upgrade Options
To use more powerful GPUs, edit `app.py`:

```python
# For higher throughput
@app.cls(gpu=modal.gpu.L4())  # 24GB VRAM

# For maximum performance  
@app.cls(gpu=modal.gpu.A100())  # 40GB VRAM
```

## Cost Optimization

### Current Settings
- **Container Idle Timeout**: 300 seconds (5 minutes)
- **Auto-scaling**: Scales to zero when unused
- **GPU**: T4 (most cost-effective for DeBERTa-v3)

### Tips
1. **Batch Requests**: Send multiple prompts per request
2. **Appropriate GPU**: T4 is sufficient for DeBERTa-v3-base
3. **Idle Timeout**: Adjust based on usage patterns
4. **Monitor Usage**: Use Modal dashboard to track costs

## Monitoring

### Modal Dashboard
Visit https://modal.com/dashboard to monitor:
- Request volume and latency
- GPU utilization
- Costs and billing
- Error rates

### Logs
```bash
# View real-time logs
modal logs nvidia-prompt-classifier

# View specific deployment
modal logs nvidia-prompt-classifier --environment=prod
```

## Troubleshooting

### Common Issues

1. **"Secret not found" error**
   ```bash
   # List secrets
   modal secret list
   
   # Recreate if missing
   modal secret create jwt-auth JWT_SECRET="your-secret"
   ```

2. **JWT authentication fails**
   - Verify JWT_SECRET matches between services
   - Check token expiration
   - Ensure proper token format

3. **Model loading timeout**
   - First deployment downloads ~2GB model (takes time)
   - Subsequent starts are faster due to caching

4. **Classification errors**
   - Check prompts are valid strings
   - Verify request format
   - Monitor Modal logs for errors

5. **Network/timeout issues**
   - Check Modal service URL
   - Verify network connectivity
   - Adjust timeout settings

### Debug Commands
```bash
# Check deployment status
modal app list

# View detailed logs
modal logs nvidia-prompt-classifier --follow

# Test endpoint directly
curl -X GET "https://YOUR-MODAL-URL/health"
```

## Development

### Local Development
```bash
# Install core dependencies
uv sync

# Serve locally for development
modal serve app.py

# Deploy to production  
modal deploy app.py
```

## Security

### Best Practices
1. **Use strong JWT secrets**: Generate cryptographically secure keys
2. **Rotate secrets**: Regularly update JWT secrets
3. **Monitor access**: Use Modal dashboard to monitor requests
4. **Network security**: Modal provides HTTPS by default

### Updating Secrets
```bash
# Update existing secret
modal secret create jwt-auth JWT_SECRET="new-secret-key" --overwrite
```

## Integration with adaptive_ai

The adaptive_ai service automatically uses the Modal API client when:
1. `MODAL_CLASSIFIER_URL` environment variable is set
2. `JWT_SECRET` matches the Modal secret
3. Modal service is healthy and accessible

The interface remains the same - existing code works without changes.

## Next Steps

1. **Monitor Performance**: Track classification latency and accuracy
2. **Scale as Needed**: Upgrade GPU if throughput requirements increase  
3. **Cost Optimization**: Adjust idle timeout based on usage patterns
4. **Security**: Regularly rotate JWT secrets
5. **Updates**: Keep Modal CLI and dependencies updated

Your NVIDIA prompt classifier is now running on Modal with GPU acceleration! 🚀