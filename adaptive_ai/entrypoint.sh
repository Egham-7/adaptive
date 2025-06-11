#!/bin/sh
set -e

echo "🚀 Starting the application..."

cd /app
exec python adaptive_ai/main.py "$@"
