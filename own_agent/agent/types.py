from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from own_agent.providers.types import ChatMessage, FinishReason, ToolCall, TokenUsage


@dataclass
class AgentEvent:
    kind: str
    text: str = ""
    tool_call: ToolCall | None = None
    tool_result: str = ""
    tool_name: str = ""
    tool_call_id: str = ""
    error: str = ""
    finish_reason: FinishReason = "stop"
    usage: Any = None


@dataclass
class AgentConfig:
    max_tool_rounds: int = 20
    max_tool_errors: int = 5
    temperature: float = 0.0
    max_tokens: int = 16384
    max_retries: int = 2
    retry_delay: float = 2.0
