"""Anthropic Messages API provider."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from own_agent.providers.base import ChatProvider
from own_agent.providers.errors import ProviderError, ProviderErrorKind, classify_exception
from own_agent.providers.types import (
    ChatMessage, ChatRequest, ChatResponse, ChatStreamEvent,
    FinishReason, ProviderCapabilities, ProviderDiagnostics,
    TokenUsage, ToolCall, ToolChoice, ToolChoiceFunction,
)


class AnthropicProvider(ChatProvider):
    def __init__(
        self,
        *,
        name: str = "anthropic",
        model: str,
        api_key: str,
        base_url: str | None = None,
        capabilities: ProviderCapabilities | None = None,
        extra_headers: dict[str, str] | None = None,
        client: Any | None = None,
    ) -> None:
        self._name = name
        self._model = model
        self._base_url = base_url
        self._capabilities = capabilities or ProviderCapabilities(
            supports_streaming=True, supports_forced_tool_choice=True, supports_parallel_tool_calls=True,
        )
        self._extra_headers = dict(extra_headers or {})
        if client is not None:
            self._client = client
        else:
            from anthropic import Anthropic
            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            if extra_headers:
                kwargs["default_headers"] = extra_headers
            self._client = Anthropic(**kwargs)

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: ChatRequest) -> ChatResponse:
        params = self._build_params(request)
        try:
            response = self._client.messages.create(**params)
        except Exception as exc:
            raise ProviderError(classify_exception(exc), str(exc)) from exc

        content = _field(response, "content", []) or []
        raw_finish = _field(response, "stop_reason")
        finish = _normalize_stop(raw_finish)
        diag = ProviderDiagnostics(raw_finish_reason=raw_finish)
        diag.reasoning = _collect_thinking(content) or None
        tool_calls = self._parse_tool_calls(content, diag)
        if finish == "length" and tool_calls:
            diag.warnings.append("finish_reason=length, discarding possibly incomplete tool_calls")
            tool_calls = []

        return ChatResponse(
            provider=self._name, model=_field(response, "model", self._model),
            content=_collect_text(content), tool_calls=tool_calls,
            finish_reason=finish, usage=_parse_usage(_field(response, "usage")),
            diagnostics=diag, raw=response,
        )

    def _build_params(self, request: ChatRequest) -> dict[str, Any]:
        if request.tools and not self._capabilities.supports_tools:
            raise ProviderError(ProviderErrorKind.CONFIG_ERROR, f"{self._name} does not support tool calling")

        params: dict[str, Any] = {
            "model": self._model,
            "messages": _to_anthropic_msgs(request.messages),
            "max_tokens": request.max_tokens or 4096,
        }
        system = _collect_system(request.messages)
        if system:
            params["system"] = system
        if request.tools:
            params["tools"] = [_to_anthropic_tool(t) for t in request.tools]
            params["tool_choice"] = self._to_anthropic_tc(request.tool_choice)
        if request.temperature is not None:
            params["temperature"] = request.temperature
        extra = {**self._extra_headers, **request.extra_body}
        if extra:
            reserved = {"model", "messages", "max_tokens", "tools", "tool_choice", "system", "stream", "temperature"}
            for k, v in extra.items():
                if k not in reserved:
                    params[k] = v
        return params

    def _to_anthropic_tc(self, tc: ToolChoice | None) -> dict[str, Any]:
        if tc is None or tc == "auto":
            payload: dict[str, Any] = {"type": "auto"}
        elif tc == "none":
            payload = {"type": "none"}
        elif tc == "required":
            payload = {"type": "any"}
        elif isinstance(tc, ToolChoiceFunction):
            if not self._capabilities.supports_forced_tool_choice:
                raise ProviderError(ProviderErrorKind.CONFIG_ERROR, "forced tool_choice not enabled")
            payload = {"type": "tool", "name": tc.name}
        else:
            raise ProviderError(ProviderErrorKind.CONFIG_ERROR, f"unsupported tool_choice: {tc!r}")
        if not self._capabilities.supports_parallel_tool_calls:
            payload["disable_parallel_tool_use"] = True
        return payload

    @staticmethod
    def _parse_tool_calls(content: list[Any], diag: ProviderDiagnostics) -> list[ToolCall]:
        parsed: list[ToolCall] = []
        for block in content:
            if _field(block, "type") != "tool_use":
                continue
            raw_input = _field(block, "input", {}) or {}
            if isinstance(raw_input, str):
                try:
                    args = json.loads(raw_input)
                except json.JSONDecodeError:
                    diag.warnings.append(f"tool_call arguments not valid JSON, discarding")
                    return []
            elif isinstance(raw_input, dict):
                args = raw_input
            else:
                diag.warnings.append(f"tool_call arguments not valid, discarding")
                return []
            if not isinstance(args, dict):
                diag.warnings.append(f"tool_call arguments not an object, discarding")
                return []
            parsed.append(ToolCall(id=_field(block, "id", ""), name=_field(block, "name", ""), arguments=args))
        return parsed


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize_stop(reason: Any) -> FinishReason:
    if reason in ("end_turn", "stop_sequence"):
        return "stop"
    if reason == "tool_use":
        return "tool_calls"
    if reason == "max_tokens":
        return "length"
    return "unknown"


def _collect_text(blocks: list[Any]) -> str:
    return "".join(_field(b, "text", "") or "" for b in blocks if _field(b, "type") == "text")


def _collect_thinking(blocks: list[Any]) -> str:
    parts: list[str] = []
    for b in blocks:
        bt = _field(b, "type")
        if bt in ("thinking", "reasoning"):
            t = _field(b, "thinking") or _field(b, "text") or ""
            if t:
                parts.append(t)
    return "".join(parts)


def _collect_system(messages: list[ChatMessage]) -> str:
    return "\n\n".join(m.content for m in messages if m.role == "system")


def _to_anthropic_msgs(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "tool":
            block = {"type": "tool_result", "tool_use_id": msg.tool_call_id or "", "content": msg.content}
            if converted and converted[-1]["role"] == "user" and isinstance(converted[-1]["content"], list):
                converted[-1]["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})
            continue
        if msg.role == "assistant" and msg.tool_calls:
            content: list[dict[str, Any]] = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments if isinstance(tc.arguments, dict) else {}})
            converted.append({"role": "assistant", "content": content})
            continue
        converted.append({"role": msg.role, "content": msg.content})
    return converted


def _to_anthropic_tool(tool) -> dict[str, Any]:
    return {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}


def _parse_usage(usage: Any):
    if usage is None:
        return None
    inp = _field(usage, "input_tokens") or _field(usage, "prompt_tokens")
    out = _field(usage, "output_tokens") or _field(usage, "completion_tokens")
    if inp is None and out is None:
        return None
    return TokenUsage(input_tokens=inp, output_tokens=out, total_tokens=(inp or 0) + (out or 0) if inp is not None and out is not None else None)
