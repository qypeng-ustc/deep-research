import argparse
import logging
import signal
import sys
import os
import uvicorn
from uuid import uuid4
from dotenv import load_dotenv
from typing import Annotated, cast
from typing import List, Optional, Union
from pydantic import BaseModel, Field

from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse

from src.workflow import _astream_agent_generator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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


class ContentItem(BaseModel):
    type: str = Field(..., description="The type of content (text, image, etc.)")
    text: Optional[str] = Field(None, description="The text content if type is 'text'")
    image_url: Optional[str] = Field(
        None, description="The image URL if type is 'image'"
    )


class ChatMessage(BaseModel):
    role: str = Field(
        ..., description="The role of the message sender (user or assistant)"
    )
    content: Union[str, List[ContentItem]] = Field(
        ...,
        description="The content of the message, either a string or a list of content items",
    )


class ChatRequest(BaseModel):
    messages: Optional[List[ChatMessage]] = Field(
        [], description="History of messages between the user and the assistant"
    )
    thread_id: Optional[str] = Field(
        "__default__", description="A specific conversation identifier"
    )
    replan_instruction: Optional[str] = Field(
        None, description="Interrupt feedback from the user on the plan"
    )


app = FastAPI(
    title="DeepResearch API",
    description="API for DeepResearch",
    version="0.1.0"
)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id
    thread_id = str(uuid4()) if thread_id == "__default__" else thread_id
    return StreamingResponse(
        _astream_agent_generator(
            request.model_dump()["messages"],
            thread_id,
            request.replan_instruction
        ),
        media_type="text/event-stream"
    )


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
            "server:app",
            host=args.host,
            port=args.port,
            reload=reload,
            log_level=args.log_level,
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)
