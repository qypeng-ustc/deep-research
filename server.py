import argparse
import logging
import signal
import sys
import os
from datetime import datetime
import uvicorn
from dotenv import load_dotenv


log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"main-{timestamp}.log")

log_format = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"

logging.basicConfig(
    filename=log_file, encoding='utf-8', level=logging.INFO, format=log_format
)

logger = logging.getLogger(__name__)

load_dotenv()

if os.getenv("DEBUG", False):
    logging.getLogger("src").setLevel(logging.DEBUG)


def handle_shutdown(signum, frame):
    """Handle graceful shutdown on SIGTERM/SIGINT"""
    logger.info("Received shutdown signal. Starting graceful shutdown...")
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the DeepResearch API server"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (default: True except on Windows)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind the server to (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    # Determine reload setting
    reload = True if args.reload else False

    try:
        logger.info(f"Starting DeepResearch API server on {args.host}:{args.port}")
        uvicorn.run(
            "src:app",
            host=args.host,
            port=args.port,
            reload=reload,
            log_level=args.log_level,
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)
