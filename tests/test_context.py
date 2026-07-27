"""Tests for ContextManager compression."""

import pytest

from own_agent.context.manager import ContextManager, _count_messages_tokens
from own_agent.providers.types import ChatMessage


def _make_msg(role: str, text: str) -> ChatMessage:
    return ChatMessage(role=role, content=text)


@pytest.mark.asyncio
async def test_empty():
    cm = ContextManager(max_tokens=1000)
    msgs: list[ChatMessage] = []
    cm.check(msgs)
    assert not cm.needs_compression
    result = await cm.compress(msgs)
    assert result == msgs


@pytest.mark.asyncio
async def test_below_threshold():
    cm = ContextManager(max_tokens=10000)
    msgs = [_make_msg("user", "hello"), _make_msg("assistant", "hi")]
    cm.check(msgs)
    assert not cm.needs_compression
    result = await cm.compress(msgs)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_above_threshold():
    cm = ContextManager(max_tokens=100)
    msgs = [_make_msg("user", "A" * 400)]
    cm.check(msgs)
    assert cm.needs_compression
    result = await cm.compress(msgs)
    assert len(result) == 1
    assert result[0].role == "user"


@pytest.mark.asyncio
async def test_compression_with_summary():
    async def fake_summarize(msgs):
        return "Summarized: user said hello"

    cm = ContextManager(max_tokens=100, llm_summarize=fake_summarize)
    msgs = [
        _make_msg("user", "A" * 200),
        _make_msg("assistant", "B" * 200),
        _make_msg("user", "C" * 200),
    ]
    cm.check(msgs)
    assert cm.needs_compression

    result = await cm.compress(msgs)
    assert len(result) >= 1
    assert result[0].role == "system"
    assert "compressed_history" in result[0].content
    assert "Summarized" in result[0].content


def test_count_tokens():
    msg = ChatMessage(role="user", content="hello world", tool_calls=[])
    count = _count_messages_tokens([msg])
    assert count == 3


def test_count_with_tool_calls():
    from own_agent.providers.types import ToolCall
    tc = ToolCall(id="1", name="test_tool", arguments={"arg": "value"})
    msg = ChatMessage(role="assistant", content="", tool_calls=[tc])
    count = _count_messages_tokens([msg])
    assert count > 0
