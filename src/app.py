import base64
import json
import logging
import os
from typing import Annotated, cast
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from uuid import uuid4
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langchain_core.messages import AIMessageChunk, ToolMessage, BaseMessage
from langgraph.types import Command

from src.graph import build_graph_with_memory
from src.configuration import Configuration


load_dotenv()

if os.getenv("DEBUG", False):
    logging.getLogger("src").setLevel(logging.DEBUG)

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


def _make_event(event_type: str, data: dict[str, any]):
    if data.get("content") == "":
        data.pop("content")
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


app = FastAPI(
    title="DeepResearch API",
    description="API for DeepResearch",
    version="0.1.0"
)

graph = build_graph_with_memory()


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


async def _astream_agent_generator(
    messages: List[dict],
    thread_id: str,
    replan_instruction: str
):
    config = Configuration.from_runnable_config()

    if not config.auto_accepted_plan and replan_instruction:
        resume_msg = f"[{replan_instruction}]"
        if messages:
            resume_msg += f" {messages[-1]['content']}"
        input_ = Command(resume=resume_msg)
    else:
        input_ = {"messages": messages}

    async for agent, _, event_data in graph.astream(
        input_,
        config={"configurable": dict(thread_id=thread_id, **config.model_dump())},
        stream_mode=["messages", "updates"],
        subgraphs=True
    ):
        if isinstance(event_data, dict):
            if "__interrupt__" in event_data:
                yield _make_event(
                    "interrupt",
                    {
                        "thread_id": thread_id,
                        "id": event_data["__interrupt__"][0].ns[0],
                        "role": "assistant",
                        "content": event_data["__interrupt__"][0].value,
                        "finish_reason": "interrupt",
                        "options": [
                            {"text": "Edit plan", "value": "edit_plan"},
                            {"text": "Start research", "value": "accepted"},
                        ],
                    },
                )
            continue
        message_chunk, message_metadata = cast(
            tuple[BaseMessage, dict[str, any]], event_data
        )
        event_stream_message: dict[str, any] = {
            "thread_id": thread_id,
            "agent": agent[0].split(":")[0],
            "id": message_chunk.id,
            "role": "assistant",
            "content": message_chunk.content,
        }
        if message_chunk.additional_kwargs.get("reasoning_content"):
            event_stream_message["reasoning_content"] = message_chunk.additional_kwargs[
                "reasoning_content"
            ]
        if message_chunk.response_metadata.get("finish_reason"):
            event_stream_message["finish_reason"] = message_chunk.response_metadata.get(
                "finish_reason"
            )
        if isinstance(message_chunk, ToolMessage):
            # Tool Message - Return the result of the tool call
            event_stream_message["tool_call_id"] = message_chunk.tool_call_id
            yield _make_event("tool_call_result", event_stream_message)
        elif isinstance(message_chunk, AIMessageChunk):
            # AI Message - Raw message tokens
            if message_chunk.tool_calls:
                # AI Message - Tool Call
                event_stream_message["tool_calls"] = message_chunk.tool_calls
                event_stream_message["tool_call_chunks"] = (
                    message_chunk.tool_call_chunks
                )
                yield _make_event("tool_calls", event_stream_message)
            elif message_chunk.tool_call_chunks:
                # AI Message - Tool Call Chunks
                event_stream_message["tool_call_chunks"] = (
                    message_chunk.tool_call_chunks
                )
                yield _make_event("tool_call_chunks", event_stream_message)
            else:
                # AI Message - Raw message tokens
                yield _make_event("message_chunk", event_stream_message)
