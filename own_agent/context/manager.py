from __future__ import annotations

from collections.abc import Callable

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
        self._archived_up_to: int = 0
        self._pending = False

    def check(self, messages: list[ChatMessage]) -> None:
        """Check if compression is needed. Call after new messages arrive."""
        if _count_messages_tokens(messages) > self._max_tokens * LEVEL1_TRIGGER:
            self._pending = True

    @property
    def needs_compression(self) -> bool:
        return self._pending

    async def compress(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        if not self._pending:
            return messages
        cutoff = max(self._archived_up_to, len(messages) // 2)
        to_summarize = messages[self._archived_up_to:cutoff]
        self._archived_up_to = cutoff

        if to_summarize and self._llm_summarize:
            try:
                new_summary = await self._llm_summarize(to_summarize)
                if new_summary:
                    self._summary = new_summary
            except Exception:
                pass

        self._pending = False

        if not self._summary:
            return messages

        placeholder = ChatMessage(
            role="system",
            content=f"<compressed_history>\n{self._summary}\n</compressed_history>",
        )
        return [placeholder] + messages[self._archived_up_to:]
