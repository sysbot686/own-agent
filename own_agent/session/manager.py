from __future__ import annotations

from pathlib import Path

from own_agent.providers.types import ChatMessage
from own_agent.session.store import SessionStore
from own_agent.session.types import Session


class SessionManager:
    def __init__(self, store: SessionStore) -> None:
        self._store = store
        self._current: Session | None = None

    @property
    def current(self) -> Session | None:
        return self._current

    def new(self, title: str = "") -> Session:
        session = self._store.create(title=title)
        self._current = session
        self._store.save(session)
        return session

    def resume(self, sid: str) -> Session | None:
        session = self._store.load(sid)
        if session is not None:
            self._current = session
        return session

    def add_message(self, msg: ChatMessage) -> None:
        if self._current is None:
            self.new()
        self._current.messages.append(msg)
        self._store.save(self._current)

    def all_messages(self) -> list[ChatMessage]:
        if self._current is None:
            return []
        return list(self._current.messages)

    def list_sessions(self) -> list[dict]:
        return self._store.list_sessions()

    def delete(self, sid: str) -> bool:
        return self._store.delete(sid)

    def save(self) -> None:
        if self._current is not None:
            self._store.save(self._current)
