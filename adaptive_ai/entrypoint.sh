#!/bin/sh
set -e

echo "🚀 Starting the application..."

cd /app
exec python -u adaptive_ai/main.py "$@"
