import base64
import json
import logging
from typing import cast, List, Any
from langchain_core.messages import AIMessageChunk, ToolMessage, BaseMessage
from langgraph.types import Command

from src.graph import build_graph_with_memory
from src.configuration import Configuration


logger = logging.getLogger(__name__)


graph = build_graph_with_memory()


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

    async for _, stream_mode, event_data in graph.astream(
        input_,
        config={
            "configurable": dict(thread_id=thread_id, **config.model_dump())
        },
        stream_mode=["updates"],
        subgraphs=True
    ):
        logger.info(f"{stream_mode} -> {event_data}")

        # if isinstance(event_data, dict):  # updates mode
        if stream_mode == 'updates':
            if "__interrupt__" in event_data:
                yield (
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
            elif "Coordinator" in event_data:
                if 'messages' in event_data["Coordinator"]:
                    yield(
                        "chat",
                        {
                            "thread_id": thread_id,
                            "role": "assistant",
                            "content": event_data["Coordinator"]['messages'][0].content,
                            "finish_reason": "stop"
                        }
                    )
                else:
                    yield (
                        "topic",
                        {
                            "thread_id": thread_id,
                            "content": event_data["Coordinator"]['research_topic']
                        }
                    )
            elif "Planner" in event_data:
                yield(
                    "plan",
                    {
                        "thread_id": thread_id,
                        "role": "assistant",
                        "content": event_data["Planner"]['messages'][0].content,
                        "finish_reason": "stop"
                    }
                )
            continue

        chunk, metadata = cast(
            tuple[BaseMessage, dict[str, Any]], event_data
        )
    
        event_stream: dict[str, Any] = {
            "thread_id": thread_id,
            "agent": metadata.get('langgraph_node'),
            "id": chunk.id,
            "role": "assistant",
            "content": chunk.content,
        }

        if chunk.additional_kwargs.get("reasoning_content"):
            event_stream["reasoning_content"] = chunk.additional_kwargs[
                "reasoning_content"
            ]
        if chunk.response_metadata.get("finish_reason"):
            event_stream["finish_reason"] = chunk.response_metadata.get(
                "finish_reason"
            )
        if isinstance(chunk, ToolMessage):
            event_stream["tool_call_id"] = chunk.tool_call_id
            yield "tool_call_result", event_stream
        elif isinstance(chunk, AIMessageChunk):
            if chunk.tool_calls:
                event_stream["tool_calls"] = chunk.tool_calls
                event_stream["tool_call_chunks"] = (
                    chunk.tool_call_chunks
                )
                yield "tool_calls", event_stream
            elif chunk.tool_call_chunks:
                event_stream["tool_call_chunks"] = (
                    chunk.tool_call_chunks
                )
                yield "tool_call_chunks", event_stream
            else:
                yield "message_chunk", event_stream
        
        yield "message", event_stream
