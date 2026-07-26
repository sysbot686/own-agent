"""Shared types for the provider layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


MessageRole = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal["stop", "tool_calls", "length", "content_filter", "error", "unknown", "tool_round_limit"]
StreamEventKind = Literal[
    "message_started", "reasoning_delta", "text_delta",
    "tool_call_started", "tool_call_delta", "tool_call_completed",
    "message_completed", "error",
]


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    supports_tools: bool = True
    supports_forced_tool_choice: bool = True
    supports_streaming: bool = False
    supports_parallel_tool_calls: bool = False
    token_param: str = "max_tokens"


@dataclass(slots=True)
@dataclass
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(slots=True)
class ChatMessage:
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] | str


@dataclass(frozen=True, slots=True)
class ToolChoiceFunction:
    name: str


ToolChoice = Literal["auto", "none", "required"] | ToolChoiceFunction


@dataclass(slots=True)
class ChatRequest:
    messages: list[ChatMessage]
    tools: list[ToolDefinition] = field(default_factory=list)
    tool_choice: ToolChoice | None = "auto"
    temperature: float | None = None
    max_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderDiagnostics:
    reasoning: str | None = None
    raw_finish_reason: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ChatResponse:
    provider: str
    model: str
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: FinishReason | None = None
    usage: TokenUsage | None = None
    diagnostics: ProviderDiagnostics = field(default_factory=ProviderDiagnostics)
    raw: Any | None = None


@dataclass(slots=True)
class ChatStreamEvent:
    kind: StreamEventKind
    text: str = ""
    tool_call: ToolCall | None = None
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str = ""
    response: ChatResponse | None = None
    diagnostics: ProviderDiagnostics = field(default_factory=ProviderDiagnostics)
