"""CLI entry point for adaptive_router server.

Starts the FastAPI server using Hypercorn ASGI server.
"""

import logging
import os
import sys
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Start the adaptive_router FastAPI server.

    Reads configuration from environment variables:
    - HOST: Server host (default: 0.0.0.0)
    - PORT: Server port (default: 8000)
    - FASTAPI_LOG_LEVEL: Log level (default: info)
    - FASTAPI_ACCESS_LOG: Enable access logging (default: true)
    """
    try:
        # Import here to show startup banner before heavy imports
        print_banner()

        # Now import heavy dependencies
        from hypercorn.asyncio import serve
        from hypercorn.config import Config

        from adaptive_router.api.app import app

        # Get configuration from environment
        host = os.getenv("HOST", "0.0.0.0")
        port_str = os.getenv("PORT", "8000")
        log_level = os.getenv("FASTAPI_LOG_LEVEL", "info")
        access_log_enabled = os.getenv("FASTAPI_ACCESS_LOG", "true").lower() == "true"

        try:
            port = int(port_str)
        except ValueError:
            logger.error(f"Invalid PORT value: {port_str}. Using default 8000")
            port = 8000

        # Configure Hypercorn
        config = Config()
        config.bind = [f"{host}:{port}"]
        config.loglevel = log_level
        config.accesslog = "-" if access_log_enabled else None
        config.errorlog = "-"

        # Log startup configuration
        logger.info("=" * 60)
        logger.info("🚀 Starting Adaptive Router FastAPI Server")
        logger.info("=" * 60)
        logger.info(f"📍 Server: http://{host}:{port}")
        logger.info(f"📖 API Docs: http://{host}:{port}/docs")
        logger.info(f"📚 ReDoc: http://{host}:{port}/redoc")
        logger.info(f"🔍 Log Level: {log_level}")
        logger.info(f"📝 Access Log: {'enabled' if access_log_enabled else 'disabled'}")
        logger.info("=" * 60)
        logger.info("✨ Features:")
        logger.info("  • Cluster-based intelligent model routing")
        logger.info("  • Cost optimization with configurable bias")
        logger.info("  • MinIO S3 storage for cluster profiles")
        logger.info("  • OpenAI-compatible request/response format")
        logger.info("=" * 60)
        logger.info("⚠️  Press CTRL+C to stop the server")
        logger.info("=" * 60)

        # Start server
        import asyncio

        asyncio.run(serve(app, config))

    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}", exc_info=True)
        sys.exit(1)


def print_banner() -> None:
    """Print startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║              🤖 Adaptive Router API Server  🚀            ║
    ║                                                           ║
    ║        Intelligent LLM Model Selection Service           ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


if __name__ == "__main__":
    main()
