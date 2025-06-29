import logging
from uuid import uuid4
from typing import List, Optional, Union
from pydantic import BaseModel, Field

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from src.workflow import _astream_agent_generator


logger = logging.getLogger(__name__)


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
