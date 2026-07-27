from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from own_agent.providers.types import ChatMessage


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Session:
    id: str
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    title: str = ""
    messages: list[ChatMessage] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    message_count: int = 0
