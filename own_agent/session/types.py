from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from own_agent.providers.types import ChatMessage


@dataclass
class Session:
    id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    title: str = ""
    messages: list[ChatMessage] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    message_count: int = 0
