from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from own_agent.providers.types import ChatMessage
from own_agent.session.types import Session


def _session_dir(base: Path) -> Path:
    return base / "sessions"


def _session_file(base: Path, sid: str) -> Path:
    return _session_dir(base) / f"{sid}.jsonl"


def _index_file(base: Path) -> Path:
    return _session_dir(base) / "index.json"


def _msg_to_dict(msg: ChatMessage) -> dict:
    d: dict = {"role": msg.role, "content": msg.content}
    if msg.name:
        d["name"] = msg.name
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    if msg.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in msg.tool_calls
        ]
    return d


def _msg_from_dict(d: dict) -> ChatMessage:
    return ChatMessage(
        role=d.get("role", "user"),
        content=d.get("content", ""),
        name=d.get("name"),
        tool_call_id=d.get("tool_call_id"),
        tool_calls=[
            type("ToolCall", (), {"id": tc["id"], "name": tc["name"], "arguments": tc.get("arguments", {})})
            for tc in d.get("tool_calls", [])
        ] if d.get("tool_calls") else None,
    )


def _load_index(base: Path) -> dict[str, dict]:
    idxf = _index_file(base)
    if idxf.exists():
        return json.loads(idxf.read_text(encoding="utf-8"))
    return {}


def _save_index(base: Path, index: dict[str, dict]) -> None:
    idxf = _index_file(base)
    _session_dir(base).mkdir(parents=True, exist_ok=True)
    idxf.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


class SessionStore:
    def __init__(self, base: Path | None = None) -> None:
        self._base = Path(base or Path.cwd())
        _session_dir(self._base).mkdir(parents=True, exist_ok=True)

    def create(self, title: str = "", metadata: dict[str, str] | None = None) -> Session:
        sid = uuid.uuid4().hex[:12]
        now = datetime.now()
        session = Session(id=sid, title=title, created_at=now, updated_at=now, metadata=metadata or {})
        return session

    def save(self, session: Session) -> None:
        fpath = _session_file(self._base, session.id)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        with fpath.open("a", encoding="utf-8") as f:
            for msg in session.messages[session.message_count:]:
                f.write(json.dumps(_msg_to_dict(msg), ensure_ascii=False) + "\n")

        session.message_count = len(session.messages)
        session.updated_at = datetime.now()
        self._update_index(session)

    def load(self, sid: str) -> Session | None:
        fpath = _session_file(self._base, sid)
        if not fpath.exists():
            return None
        index = _load_index(self._base)
        entry = index.get(sid)
        messages: list[ChatMessage] = []
        for line in fpath.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                messages.append(_msg_from_dict(json.loads(line)))

        if entry is None:
            return None
        return Session(
            id=sid,
            title=entry.get("title", ""),
            created_at=datetime.fromisoformat(entry["created_at"]) if "created_at" in entry else datetime.now(),
            updated_at=datetime.fromisoformat(entry["updated_at"]) if "updated_at" in entry else datetime.now(),
            messages=messages,
            metadata=entry.get("metadata", {}),
            message_count=len(messages),
        )

    def list_sessions(self) -> list[dict]:
        index = _load_index(self._base)
        return [
            {"id": sid, **entry}
            for sid, entry in sorted(index.items(), key=lambda x: x[1].get("updated_at", ""), reverse=True)
        ]

    def delete(self, sid: str) -> bool:
        fpath = _session_file(self._base, sid)
        deleted = False
        if fpath.exists():
            fpath.unlink()
            deleted = True
        index = _load_index(self._base)
        if sid in index:
            del index[sid]
            _save_index(self._base, index)
            deleted = True
        return deleted

    def _update_index(self, session: Session) -> None:
        index = _load_index(self._base)
        index[session.id] = {
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": len(session.messages),
            "metadata": session.metadata,
        }
        _save_index(self._base, index)
