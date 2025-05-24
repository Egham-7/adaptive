#!/bin/sh
# entrypoint.sh

set -e # Exit immediately on error

echo "🚀 Starting the application..."

# Run the app
exec python /app/main.py
