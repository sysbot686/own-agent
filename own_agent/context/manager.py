from __future__ import annotations

from collections.abc import Callable
from typing import Any

from own_agent.providers.types import ChatMessage


def _rough_tokens(text: str) -> int:
    return len(text) // 4 + 1


def _count_messages_tokens(messages: list[ChatMessage]) -> int:
    total = 0
    for m in messages:
        total += _rough_tokens(m.content or "")
        if m.tool_calls:
            for tc in m.tool_calls:
                total += _rough_tokens(tc.name)
                total += _rough_tokens(str(tc.arguments))
    return total


LEVEL1_TRIGGER = 0.85


class ContextManager:
    def __init__(
        self,
        max_tokens: int = 128000,
        llm_summarize: Callable[[list[ChatMessage]], str] | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._llm_summarize = llm_summarize
        self._summary: str = ""
        self._summary_stale = False
        self._last_archived_idx = 0
        self._messages: list[ChatMessage] = []

    @property
    def messages(self) -> list[ChatMessage]:
        return list(self._messages)

    def add_messages(self, msgs: list[ChatMessage]) -> None:
        self._messages.extend(msgs)
        if _count_messages_tokens(self._messages) > self._max_tokens * LEVEL1_TRIGGER:
            self._archive()

    def compressed(self) -> list[ChatMessage]:
        if not self._summary:
            return list(self._messages)
        placeholder = ChatMessage(
            role="system",
            content=f"<compressed_history>\n{self._summary}\n</compressed_history>",
        )
        recent = self._messages[self._last_archived_idx:]
        return [placeholder] + recent

    def _archive(self) -> None:
        cutoff = max(self._last_archived_idx, len(self._messages) // 2)
        archived = self._messages[self._last_archived_idx:cutoff]
        self._last_archived_idx = cutoff

        if archived and self._llm_summarize is not None:
            try:
                new_summary = self._llm_summarize(archived)
                if new_summary:
                    self._summary = new_summary
            except Exception:
                pass

        self._summary_stale = True

    def needs_summary(self) -> bool:
        return self._summary_stale and self._llm_summarize is not None

    def can_skip_summary(self) -> bool:
        return _count_messages_tokens(self._messages) < self._max_tokens
