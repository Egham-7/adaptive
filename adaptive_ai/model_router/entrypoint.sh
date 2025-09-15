#!/bin/bash
set -e

echo "🚀 Starting the application..."
echo "📁 Working directory: $(pwd)"
echo "🐍 Python version: $(python --version)"

cd /app

echo "✅ Application ready to start"
exec uv run python -u model_router/main.py "$@"

