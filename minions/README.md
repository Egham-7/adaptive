# Azure Container Apps vLLM Deployment

A configurable template for deploying vLLM (Large Language Model serving) on Azure Container Apps with GPU support.

## 🚀 Features

- **GPU Support**: Deploy on NC24-A100, NC8as-T4, or Consumption workload profiles
- **Configurable Models**: Easy model switching via environment variables
- **Resource Scaling**: Auto-scaling with HTTP-based rules
- **Security**: Secure token management with Azure Container Apps secrets
- **Multi-Environment**: Support for dev, staging, and production environments
- **Performance Tuning**: Advanced vLLM configuration options

## 📋 Prerequisites

- Azure CLI installed and logged in (`az login`)
- Azure Container Apps Environment with GPU workload profiles
- Hugging Face account and token (for model access)
- `envsubst` utility (usually pre-installed on Linux/macOS)

### Install envsubst (if needed)

```bash
# Ubuntu/Debian
sudo apt-get install gettext-base

# macOS
brew install gettext
```

## 📁 File Structure

```
├── minion.yaml               # Main deployment template
├── deploy.sh                 # Deployment script
├── examples/
│   ├── saul-7b.env          # Legal model configuration
│   ├── small-model.env      # Smaller model for T4 GPUs
│   └── production.env       # Production settings
└── README.md                # This file
```

## ⚙️ Configuration

### Environment Variables

| Variable                 | Description                         | Default                     | Options                                |
| ------------------------ | ----------------------------------- | --------------------------- | -------------------------------------- |
| `APP_NAME`               | Container app name                  | `vllm-model-server`         | Lowercase, alphanumeric + hyphens only |
| `ENVIRONMENT_ID`         | Azure Container Apps Environment ID | Required                    | Full resource ID                       |
| `MODEL_ID`               | Hugging Face model identifier       | `microsoft/DialoGPT-medium` | Any HF model                           |
| `WORKLOAD_PROFILE`       | GPU workload profile                | `NC24-A100`                 | `Consumption`, `NC24-A100`, `NC8as-T4` |
| `HF_TOKEN`               | Hugging Face API token              | Required                    | Your HF token                          |
| `MAX_MODEL_LEN`          | Maximum sequence length             | `4096`                      | Model-dependent                        |
| `GPU_MEMORY_UTILIZATION` | GPU memory usage                    | `0.9`                       | 0.1-0.95                               |
| `MIN_REPLICAS`           | Minimum instances                   | `0`                         | 0+                                     |
| `MAX_REPLICAS`           | Maximum instances                   | `1`                         | 1+                                     |
| `STORAGE_NAME`           | Azure File Share storage config     | Optional                    | Storage configuration name             |
| `CONTAINER_IMAGE`        | vLLM container image                | `vllm/vllm-openai:latest`   | Any vLLM-compatible image              |

### Container App Naming Requirements

⚠️ **Important**: Container app names must follow strict rules:
- Only lowercase letters, numbers, and hyphens (`-`)
- Start with a letter
- End with a letter or number  
- No double hyphens (`--`)
- Length: 2-32 characters

❌ Bad: `financeLlama38B`, `my--app`, `App-Name`  
✅ Good: `finance-llama-38b`, `my-app`, `app-name`

### Get Your Environment ID

```bash
az containerapp env show \
  --name <your-environment-name> \
  --resource-group <your-resource-group> \
  --query "id" --output tsv
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Create your environment file
cp .env my-model.env

# Edit with your configuration
nano my-model.env
```

### 2. Configure Environment

```bash
# Example configuration for Saul-7B legal model
export APP_NAME="vllm-saul-7b-law"
export ENVIRONMENT_ID="/subscriptions/YOUR-SUB/resourceGroups/YOUR-RG/providers/Microsoft.App/managedEnvironments/YOUR-ENV"
export MODEL_ID="Equall/Saul-7B-Instruct-v1"
export SERVED_MODEL_NAME="saul-7b-law"
export HF_TOKEN="hf_your_token_here"
export WORKLOAD_PROFILE="NC24-A100"
```

### 3. Deploy

```bash
# Method 1: Using the deployment script (recommended)
chmod +x deploy.sh
./deploy.sh my-model.env create

# Method 2: Direct deployment
source my-model.env
envsubst < containerapp-template.yaml | az containerapp create --yaml -
```

## 📖 Usage Examples

### Deploy Legal AI Model (Saul-7B)

```bash
# Create environment file
cat > saul-7b.env << EOF
export APP_NAME="legal-ai-saul"
export MODEL_ID="Equall/Saul-7B-Instruct-v1"
export SERVED_MODEL_NAME="saul-7b"
export WORKLOAD_PROFILE="NC24-A100"
export MAX_MODEL_LEN="4096"
export HF_TOKEN="your_token_here"
export ENVIRONMENT_ID="your_environment_id"
EOF

# Deploy
./deploy.sh saul-7b.env create
```

### Deploy Smaller Model on T4

```bash
# Create environment file for T4 GPU
cat > dialog-model.env << EOF
export APP_NAME="dialog-ai"
export MODEL_ID="microsoft/DialoGPT-medium"
export SERVED_MODEL_NAME="dialog-gpt"
export WORKLOAD_PROFILE="NC8as-T4"
export CPU_LIMIT="2.0"
export MEMORY_LIMIT="16Gi"
export HF_TOKEN="your_token_here"
export ENVIRONMENT_ID="your_environment_id"
EOF

# Deploy
./deploy.sh dialog-model.env create
```

### Update Existing Deployment

```bash
# Modify your environment file
nano my-model.env

# Update the deployment
./deploy.sh my-model.env update
```

## 💾 Persistent Model Storage

To avoid downloading models repeatedly, mount an Azure File Share:

### 1. Create Azure File Share Storage Configuration

```bash
# Add storage to your Container Apps environment
az containerapp env storage set \
  --name <environment-name> \
  --resource-group <resource-group> \
  --storage-name model-cache \
  --azure-file-account-name <storage-account-name> \
  --azure-file-account-key <storage-account-key> \
  --azure-file-share-name <file-share-name> \
  --access-mode ReadWrite
```

### 2. Update Environment Configuration

Add to your `.env` file:
```bash
export STORAGE_NAME="model-cache"  # Name used in step 1
```

The deployment will automatically mount the file share to `/root/.cache/huggingface` for persistent model caching.

## 🔧 Advanced Configuration

### Model Performance Tuning

```bash
# High-performance configuration
export GPU_MEMORY_UTILIZATION="0.95"
export TENSOR_PARALLEL_SIZE="2"
export MAX_NUM_SEQS="512"
export DTYPE="float16"
```

### Quantization (Reduce Memory Usage)

```bash
# Enable quantization for smaller memory footprint
export QUANTIZATION="awq"  # Options: awq, gptq, squeezellm, fp8
export DTYPE="auto"
```

### LoRA Support

```bash
# Enable LoRA adapters
export ENABLE_LORA="true"
export MAX_LORA_RANK="64"
```

### Scaling Configuration

```bash
# Auto-scaling settings
export MIN_REPLICAS="1"
export MAX_REPLICAS="5"
export CONCURRENT_REQUESTS="16"
```

## 🛠️ Management Commands

### Check Deployment Status

```bash
az containerapp show \
  --name $APP_NAME \
  --resource-group <your-rg> \
  --query "properties.provisioningState"
```

### Get Application URL

```bash
az containerapp show \
  --name $APP_NAME \
  --resource-group <your-rg> \
  --query "properties.configuration.ingress.fqdn" \
  --output tsv
```

### View Logs

```bash
az containerapp logs show \
  --name $APP_NAME \
  --resource-group <your-rg> \
  --follow
```

### Scale Application

```bash
# Scale to specific replica count
az containerapp update \
  --name $APP_NAME \
  --resource-group <your-rg> \
  --min-replicas 2 \
  --max-replicas 10
```

### Delete Application

```bash
az containerapp delete \
  --name $APP_NAME \
  --resource-group <your-rg> \
  --yes
```

## 🧪 Testing Your Deployment

### Health Check

```bash
# Get your app URL
APP_URL=$(az containerapp show --name $APP_NAME --resource-group <your-rg> --query "properties.configuration.ingress.fqdn" --output tsv)

# Test health endpoint
curl https://$APP_URL/health
```

### Test Model Inference

```bash
# OpenAI-compatible API call
curl -X POST "https://$APP_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'
```

### Chat Completions

```bash
curl -X POST "https://$APP_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [
      {"role": "user", "content": "What is artificial intelligence?"}
    ],
    "max_tokens": 150
  }'
```

## 🔍 Troubleshooting

### Common Issues

1. **Container Name Validation Error**
   ```
   Error: Invalid ContainerApp name 'MyApp123'. A name must consist of lower case alphanumeric characters or '-'
   ```
   **Solution**: Use only lowercase letters, numbers, and hyphens. Example: `my-app-123`

2. **vLLM Boolean Argument Errors**
   ```
   api_server.py: error: unrecognized arguments: false false false
   ```
   **Solution**: Boolean flags should not have values. Use only the flag name:
   - ❌ Wrong: `--trust-remote-code false`
   - ✅ Correct: `--trust-remote-code` (to enable) or omit entirely (to disable)

3. **Deployment Fails with "Invalid YAML"**

   ```bash
   # Preview generated YAML before deployment
   source .env
   envsubst < minion.yaml > preview.yaml
   cat preview.yaml
   ```

4. **Model Download Timeout**
   - Increase `initialDelaySeconds` in health probes
   - Use smaller models for testing
   - Ensure sufficient memory allocation

5. **GPU Not Available**

   ```bash
   # Check available workload profiles
   az containerapp env show \
     --name <env-name> \
     --resource-group <rg> \
     --query "properties.workloadProfiles"
   ```

6. **Out of Memory Errors**
   - Reduce `GPU_MEMORY_UTILIZATION`
   - Enable quantization
   - Use smaller `MAX_MODEL_LEN`

### Debugging Commands

```bash
# View environment details
az containerapp env show --name <env-name> --resource-group <rg>

# Check container app configuration
az containerapp show --name $APP_NAME --resource-group <rg>

# Stream logs
az containerapp logs show --name $APP_NAME --resource-group <rg> --follow

# Check revision status
az containerapp revision list --name $APP_NAME --resource-group <rg>
```

## 📚 Additional Resources

- [Azure Container Apps Documentation](https://docs.microsoft.com/en-us/azure/container-apps/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [Azure CLI Reference](https://docs.microsoft.com/en-us/cli/azure/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different models/configurations
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Need Help?** Open an issue or check the troubleshooting section above.
