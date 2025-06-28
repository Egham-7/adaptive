#!/bin/bash
set -e

echo "🚀 Starting the application..."
echo "📁 Working directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo "🔥 PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Check if CUDA is available
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())"; then
  echo "🎮 CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
fi

cd /app

echo "✅ Application ready to start"
exec uv run python -u adaptive_ai/main.py "$@"
