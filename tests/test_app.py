from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from src.app import app, _make_event, _astream_workflow_generator
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_core.messages import AIMessageChunk
from src.app.chat_request import ChatRequest


@pytest.fixture
def client():
    return TestClient(app)


class TestMakeEvent:
    def test_make_event_with_content(self):
        event_type = "message_chunk"
        data = {"content": "Hello", "role": "assistant"}
        result = _make_event(event_type, data)
        expected = (
            'event: message_chunk\ndata: {"content": "Hello", "role": "assistant"}\n\n'
        )
        assert result == expected

    def test_make_event_with_empty_content(self):
        event_type = "message_chunk"
        data = {"content": "", "role": "assistant"}
        result = _make_event(event_type, data)
        expected = 'event: message_chunk\ndata: {"role": "assistant"}\n\n'
        assert result == expected

    def test_make_event_without_content(self):
        event_type = "tool_calls"
        data = {"role": "assistant", "tool_calls": []}
        result = _make_event(event_type, data)
        expected = (
            'event: tool_calls\ndata: {"role": "assistant", "tool_calls": []}\n\n'
        )
        assert result == expected


class TestChatStreamEndpoint:
    @patch("src.app.graph")
    def test_chat_stream_with_default_thread_id(self, mock_graph, client):
        # Mock the async stream
        async def mock_astream(*args, **kwargs):
            yield ("agent1", "step1", {"test": "data"})

        mock_graph.astream = mock_astream

        request_data = {
            "thread_id": "__default__",
            "messages": [{"role": "user", "content": "Hello"}],
            "interrupt_feedback": ""
        }

        response = client.post("/api/chat/stream", json=request_data)

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestAstreamWorkflowGenerator:
    @pytest.mark.asyncio
    @patch("src.app.graph")
    async def test_astream_workflow_generator_basic_flow(self, mock_graph):
        # Mock AI message chunk
        mock_message = AIMessageChunk(content="Hello world")
        mock_message.id = "msg_123"
        mock_message.response_metadata = {}
        mock_message.tool_calls = []
        mock_message.tool_call_chunks = []

        # Mock the async stream - yield messages in the correct format
        async def mock_astream(*args, **kwargs):
            # Yield a tuple (message, metadata) instead of just [message]
            yield ("agent1:subagent", "messages", (mock_message, {}))

        mock_graph.astream = mock_astream

        messages = [{"role": "user", "content": "Hello"}]
        thread_id = "test_thread"
        resources = []

        generator = _astream_workflow_generator(
            messages=messages,
            thread_id=thread_id,
            interrupt_feedback=""
        )

        events = []
        async for event in generator:
            events.append(event)

        assert len(events) == 1
        assert "event: message_chunk" in events[0]
        assert "Hello world" in events[0]
        # Check for the actual agent name that appears in the output
        assert '"agent": "a"' in events[0]

    @pytest.mark.asyncio
    @patch("src.app.graph")
    async def test_astream_workflow_generator_with_interrupt_feedback(self, mock_graph):

        # Mock the async stream
        async def mock_astream(*args, **kwargs):
            # Verify that Command is passed as input when interrupt_feedback is provided
            assert isinstance(args[0], Command)
            assert "[edit_plan] Hello" in args[0].resume
            yield ("agent1", "step1", {"test": "data"})

        mock_graph.astream = mock_astream

        messages = [{"role": "user", "content": "Hello"}]

        generator = _astream_workflow_generator(
            messages=messages,
            thread_id="test_thread",
            interrupt_feedback="edit_plan"
        )

        events = []
        async for event in generator:
            events.append(event)

    @pytest.mark.asyncio
    @patch("src.app.graph")
    async def test_astream_workflow_generator_interrupt_event(self, mock_graph):
        # Mock interrupt data
        mock_interrupt = MagicMock()
        mock_interrupt.ns = ["interrupt_id"]
        mock_interrupt.value = "Plan requires approval"

        interrupt_data = {"__interrupt__": [mock_interrupt]}

        async def mock_astream(*args, **kwargs):
            yield ("agent1", "step1", interrupt_data)

        mock_graph.astream = mock_astream

        generator = _astream_workflow_generator(
            messages=[],
            thread_id="test_thread",
            interrupt_feedback=""
        )

        events = []
        async for event in generator:
            events.append(event)

        assert len(events) == 1
        assert "event: interrupt" in events[0]
        assert "Plan requires approval" in events[0]
        assert "interrupt_id" in events[0]

    @pytest.mark.asyncio
    @patch("src.app.graph")
    async def test_astream_workflow_generator_tool_message(self, mock_graph):

        # Mock tool message
        mock_tool_message = ToolMessage(content="Tool result", tool_call_id="tool_123")
        mock_tool_message.id = "msg_456"

        async def mock_astream(*args, **kwargs):
            yield ("agent1:subagent", "step1", (mock_tool_message, {}))

        mock_graph.astream = mock_astream

        generator = _astream_workflow_generator(
            messages=[],
            thread_id="test_thread",
            interrupt_feedback=""
        )

        events = []
        async for event in generator:
            events.append(event)

        assert len(events) == 1
        assert "event: tool_call_result" in events[0]
        assert "Tool result" in events[0]
        assert "tool_123" in events[0]

    @pytest.mark.asyncio
    @patch("src.app.graph")
    async def test_astream_workflow_generator_ai_message_with_tool_calls(
        self, mock_graph
    ):

        # Mock AI message with tool calls
        mock_ai_message = AIMessageChunk(content="Making tool call")
        mock_ai_message.id = "msg_789"
        mock_ai_message.response_metadata = {"finish_reason": "tool_calls"}
        mock_ai_message.tool_calls = [{"name": "search", "args": {"query": "test"}}]
        mock_ai_message.tool_call_chunks = [{"name": "search"}]

        async def mock_astream(*args, **kwargs):
            yield ("agent1:subagent", "step1", (mock_ai_message, {}))

        mock_graph.astream = mock_astream

        generator = _astream_workflow_generator(
            messages=[],
            thread_id="test_thread",
            interrupt_feedback=""
        )

        events = []
        async for event in generator:
            events.append(event)

        assert len(events) == 1
        assert "event: tool_calls" in events[0]
        assert "Making tool call" in events[0]
        assert "tool_calls" in events[0]

    @pytest.mark.asyncio
    @patch("src.app.graph")
    async def test_astream_workflow_generator_ai_message_with_tool_call_chunks(
        self, mock_graph
    ):

        # Mock AI message with only tool call chunks
        mock_ai_message = AIMessageChunk(content="Streaming tool call")
        mock_ai_message.id = "msg_101"
        mock_ai_message.response_metadata = {}
        mock_ai_message.tool_calls = []
        mock_ai_message.tool_call_chunks = [{"name": "search", "index": 0}]

        async def mock_astream(*args, **kwargs):
            yield ("agent1:subagent", "step1", (mock_ai_message, {}))

        mock_graph.astream = mock_astream

        generator = _astream_workflow_generator(
            messages=[],
            thread_id="test_thread",
            interrupt_feedback=""
        )

        events = []
        async for event in generator:
            events.append(event)

        assert len(events) == 1
        assert "event: tool_call_chunks" in events[0]
        assert "Streaming tool call" in events[0]

    @pytest.mark.asyncio
    @patch("src.app.graph")
    async def test_astream_workflow_generator_with_finish_reason(self, mock_graph):

        # Mock AI message with finish reason
        mock_ai_message = AIMessageChunk(content="Complete response")
        mock_ai_message.id = "msg_finish"
        mock_ai_message.response_metadata = {"finish_reason": "stop"}
        mock_ai_message.tool_calls = []
        mock_ai_message.tool_call_chunks = []

        async def mock_astream(*args, **kwargs):
            yield ("agent1:subagent", "step1", (mock_ai_message, {}))

        mock_graph.astream = mock_astream

        generator = _astream_workflow_generator(
            messages=[],
            thread_id="test_thread",
            interrupt_feedback=""
        )

        events = []
        async for event in generator:
            events.append(event)

        assert len(events) == 1
        assert "event: message_chunk" in events[0]
        assert "finish_reason" in events[0]
        assert "stop" in events[0]

    @pytest.mark.asyncio
    @patch("src.app.graph")
    async def test_astream_workflow_generator_config_passed_correctly(self, mock_graph):

        mock_ai_message = AIMessageChunk(content="Test")
        mock_ai_message.id = "test_id"
        mock_ai_message.response_metadata = {}
        mock_ai_message.tool_calls = []
        mock_ai_message.tool_call_chunks = []

        async def verify_config(*args, **kwargs):
            config = kwargs.get("config", {})
            assert config["thread_id"] == "test_thread"
            yield ("agent1", "messages", [mock_ai_message])
