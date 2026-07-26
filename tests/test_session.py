"""Tests for session module."""

import json
import tempfile
from pathlib import Path

from own_agent.providers.types import ChatMessage
from own_agent.session import SessionManager, SessionStore


def test_session_create():
    tmp = tempfile.mkdtemp()
    store = SessionStore(base=tmp)
    mgr = SessionManager(store)
    s = mgr.new(title="test")
    assert s.id
    assert s.title == "test"


def test_session_add_messages():
    tmp = tempfile.mkdtemp()
    store = SessionStore(base=tmp)
    mgr = SessionManager(store)
    mgr.new()
    mgr.add_message(ChatMessage(role="user", content="hello"))
    mgr.add_message(ChatMessage(role="assistant", content="world"))
    assert len(mgr.all_messages()) == 2
    assert mgr.all_messages()[0].content == "hello"
    assert mgr.all_messages()[1].content == "world"


def test_session_persistence():
    tmp = tempfile.mkdtemp()
    store = SessionStore(base=tmp)
    mgr = SessionManager(store)
    mgr.new(title="persist")
    mgr.add_message(ChatMessage(role="user", content="ping"))

    # Reload from disk
    sid = mgr.current.id
    store2 = SessionStore(base=tmp)
    mgr2 = SessionManager(store2)
    loaded = mgr2.resume(sid)
    assert loaded is not None
    assert loaded.title == "persist"
    assert len(loaded.messages) == 1
    assert loaded.messages[0].content == "ping"


def test_session_list():
    tmp = tempfile.mkdtemp()
    store = SessionStore(base=tmp)
    mgr = SessionManager(store)
    mgr.new(title="s1")
    mgr.new(title="s2")
    sessions = mgr.list_sessions()
    assert len(sessions) == 2


def test_session_delete():
    tmp = tempfile.mkdtemp()
    store = SessionStore(base=tmp)
    mgr = SessionManager(store)
    mgr.new(title="delete_me")
    sid = mgr.current.id
    assert mgr.delete(sid)
    assert mgr.resume(sid) is None


def test_session_file_format():
    tmp = tempfile.mkdtemp()
    store = SessionStore(base=tmp)
    mgr = SessionManager(store)
    mgr.new()
    mgr.add_message(ChatMessage(role="user", content="hello", tool_calls=None))
    mgr.add_message(ChatMessage(role="assistant", content="hi", tool_calls=[type("ToolCall", (), {"id": "c1", "name": "test", "arguments": {}})()]))

    # Verify JSONL format
    sid = mgr.current.id
    fpath = Path(tmp) / "sessions" / f"{sid}.jsonl"
    assert fpath.exists()
    lines = fpath.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    for line in lines:
        obj = json.loads(line)
        assert "role" in obj
        assert "content" in obj
