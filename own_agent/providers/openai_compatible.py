"""OpenAI-compatible Chat Completions provider."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections.abc import AsyncIterator
from typing import Any

from openai import OpenAI

from own_agent.providers.base import ChatProvider
from own_agent.providers.errors import ProviderError, ProviderErrorKind, classify_error, classify_exception
from own_agent.providers.types import (
    ChatMessage, ChatRequest, ChatResponse, ChatStreamEvent,
    FinishReason, ProviderCapabilities, ProviderDiagnostics,
    TokenUsage, ToolCall, ToolChoice, ToolChoiceFunction,
)


class OpenAICompatibleProvider(ChatProvider):
    def __init__(
        self,
        *,
        name: str,
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
        self._capabilities = capabilities or ProviderCapabilities(supports_streaming=True)
        self._extra_headers = dict(extra_headers or {})
        if client is not None:
            self._client = client
        else:
            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            if extra_headers:
                kwargs["default_headers"] = extra_headers
            self._client = OpenAI(**kwargs)

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: ChatRequest) -> ChatResponse:
        params = self._build_params(request)
        try:
            response = self._client.chat.completions.create(**params)
        except Exception as exc:
            raise ProviderError(classify_exception(exc), str(exc)) from exc

        choice = _field(response, "choices", [])[0]
        message = _field(choice, "message")
        raw_finish = _field(choice, "finish_reason")
        finish = _normalize_finish(raw_finish)
        diag = ProviderDiagnostics(raw_finish_reason=raw_finish)
        diag.reasoning = _read_reasoning(message) or None
        tool_calls = self._parse_tool_calls(_field(message, "tool_calls", []) or [], diag)
        if finish == "length" and tool_calls:
            diag.warnings.append("finish_reason=length, discarding possibly incomplete tool_calls")
            tool_calls = []

        return ChatResponse(
            provider=self._name,
            model=_field(response, "model", self._model),
            content=_field(message, "content", "") or "",
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=_parse_usage(_field(response, "usage")),
            diagnostics=diag,
            raw=response,
        )

    async def astream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        if not self._capabilities.supports_streaming:
            raise ProviderError(ProviderErrorKind.UNSUPPORTED, f"{self._name} does not support streaming")

        params = self._build_params(request)
        params["stream"] = True
        diag = ProviderDiagnostics()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_accums: dict[int, _ToolAccum] = {}
        raw_finish: Any = None
        response_model = self._model

        try:
            stream = await asyncio.to_thread(self._client.chat.completions.create, **params)
        except Exception as exc:
            raise ProviderError(classify_exception(exc), str(exc)) from exc

        yield ChatStreamEvent(kind="message_started")
        q, stop = _start_stream_worker(stream)

        try:
            while True:
                item = await asyncio.to_thread(q.get)
                if item is _STREAM_END:
                    break
                if isinstance(item, _StreamFailure):
                    raise ProviderError(classify_exception(item.error), str(item.error))

                chunk = item
                response_model = _field(chunk, "model", response_model) or response_model
                choices = _field(chunk, "choices", []) or []
                if not choices:
                    continue
                delta = _field(choices[0], "delta", {}) or {}
                fr = _field(choices[0], "finish_reason")
                if fr is not None:
                    raw_finish = fr

                text = _field(delta, "content")
                if text:
                    content_parts.append(text)
                    yield ChatStreamEvent(kind="text_delta", text=text)

                reasoning = _read_reasoning(delta)
                if reasoning:
                    reasoning_parts.append(reasoning)
                    yield ChatStreamEvent(kind="reasoning_delta", text=reasoning)

                for event in _accum_tool_deltas(_field(delta, "tool_calls", []) or [], tool_accums, diag):
                    yield event
        finally:
            await asyncio.to_thread(stop)

        finish = _normalize_finish(raw_finish)
        diag.raw_finish_reason = raw_finish
        if reasoning_parts:
            diag.reasoning = "".join(reasoning_parts)

        tool_calls: list[ToolCall] = []
        if tool_accums and finish != "tool_calls":
            diag.warnings.append(f"finish_reason={finish}, discarding incomplete tool_calls")
        elif tool_accums:
            tool_calls = _complete_tool_calls(tool_accums, diag)

        for tc in tool_calls:
            yield ChatStreamEvent(kind="tool_call_completed", tool_call=tc, tool_call_id=tc.id, tool_name=tc.name)

        response = ChatResponse(
            provider=self._name, model=response_model,
            content="".join(content_parts), tool_calls=tool_calls,
            finish_reason=finish, diagnostics=diag,
        )
        yield ChatStreamEvent(kind="message_completed", response=response, diagnostics=diag)

    def _build_params(self, request: ChatRequest) -> dict[str, Any]:
        if request.tools and not self._capabilities.supports_tools:
            raise ProviderError(ProviderErrorKind.CONFIG_ERROR, f"{self._name} does not support tool calling")

        params: dict[str, Any] = {
            "model": self._model,
            "messages": [_to_openai_msg(m) for m in request.messages],
        }
        if request.tools:
            params["tools"] = [_to_openai_tool(t) for t in request.tools]
            params["tool_choice"] = _to_openai_tool_choice(request.tool_choice)
            if self._capabilities.supports_parallel_tool_calls:
                params["parallel_tool_calls"] = True
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params[self._capabilities.token_param] = request.max_tokens
        if request.extra_body:
            params["extra_body"] = dict(request.extra_body)
        return params

    @staticmethod
    def _parse_tool_calls(raw_calls: list[Any], diag: ProviderDiagnostics) -> list[ToolCall]:
        parsed: list[ToolCall] = []
        for call in raw_calls:
            func = _field(call, "function", {})
            raw_args = _field(func, "arguments", "")
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                if not isinstance(args := raw_args, dict):
                    diag.warnings.append(f"tool_call arguments not valid JSON, discarding: id={_field(call,'id')}")
                    return []
            if not isinstance(args, dict):
                diag.warnings.append(f"tool_call arguments not an object, discarding")
                return []
            parsed.append(ToolCall(id=_field(call, "id", ""), name=_field(func, "name", ""), arguments=args))
        return parsed


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize_finish(reason: Any) -> FinishReason:
    if reason in ("stop", "tool_calls", "length", "content_filter"):
        return reason
    return "unknown"


def _read_reasoning(obj: Any) -> str:
    if isinstance(obj, dict):
        for key in ("reasoning_content", "reasoning"):
            val = obj.get(key)
            if isinstance(val, str):
                return val
    return ""


def _parse_usage(usage: Any):
    if usage is None:
        return None
    inp = _field(usage, "prompt_tokens")
    out = _field(usage, "completion_tokens")
    total = _field(usage, "total_tokens")
    if inp is None and out is None and total is None:
        return None
    return TokenUsage(input_tokens=inp, output_tokens=out, total_tokens=total)


def _to_openai_msg(msg: ChatMessage) -> dict[str, Any]:
    d: dict[str, Any] = {"role": msg.role, "content": msg.content}
    if msg.tool_calls:
        calls: list[dict[str, Any]] = []
        for tc in msg.tool_calls:
            if isinstance(tc.arguments, dict):
                try:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                except (TypeError, ValueError):
                    args_str = str(tc.arguments)
            else:
                args_str = tc.arguments
            calls.append({"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": args_str}})
        d["tool_calls"] = calls
    if msg.name and msg.role != "tool":
        d["name"] = msg.name
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    return d


def _to_openai_tool(tool) -> dict[str, Any]:
    return {"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters}}


def _to_openai_tool_choice(tc: ToolChoice | None) -> str | dict[str, Any] | None:
    if tc is None:
        return None
    if isinstance(tc, ToolChoiceFunction):
        return {"type": "function", "function": {"name": tc.name}}
    if isinstance(tc, str):
        return tc
    raise ProviderError(ProviderErrorKind.CONFIG_ERROR, f"unsupported tool_choice: {tc!r}")


class _ToolAccum:
    def __init__(self, index: int) -> None:
        self.index = index
        self.id = ""
        self.name = ""
        self.arguments_text = ""
        self.saw_arguments = False


_STREAM_END = object()


class _StreamFailure:
    def __init__(self, error: BaseException) -> None:
        self.error = error


def _start_stream_worker(stream):
    q: queue.Queue[Any] = queue.Queue()
    stop_event = threading.Event()

    def stop():
        stop_event.set()
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def worker():
        try:
            for item in stream:
                if stop_event.is_set():
                    break
                q.put(item)
        except BaseException as exc:
            q.put(_StreamFailure(exc))
        finally:
            stop()
            q.put(_STREAM_END)

    threading.Thread(target=worker, daemon=True).start()
    return q, stop


def _accum_tool_deltas(
    deltas: list[Any], accums: dict[int, _ToolAccum], diag: ProviderDiagnostics,
) -> list[ChatStreamEvent]:
    events: list[ChatStreamEvent] = []
    for delta in deltas:
        idx = _field(delta, "index")
        if not isinstance(idx, int):
            continue
        is_new = idx not in accums
        accum = accums.setdefault(idx, _ToolAccum(idx))
        cid = _field(delta, "id")
        if cid:
            accum.id = cid
        func = _field(delta, "function", {}) or {}
        nd = _field(func, "name", "") or ""
        ad = _field(func, "arguments", "") or ""
        if nd:
            accum.name += nd
        if ad:
            accum.arguments_text += ad
            accum.saw_arguments = True
        if is_new:
            events.append(ChatStreamEvent(kind="tool_call_started", tool_call_index=idx, tool_call_id=accum.id or None, tool_name=accum.name or None))
        events.append(ChatStreamEvent(kind="tool_call_delta", tool_call_index=idx, tool_call_id=accum.id or None, tool_name=accum.name or None, arguments_delta=ad))
    return events


def _complete_tool_calls(accums: dict[int, _ToolAccum], diag: ProviderDiagnostics) -> list[ToolCall]:
    parsed: list[ToolCall] = []
    for idx in sorted(accums):
        item = accums[idx]
        if not item.id or not item.name or not item.saw_arguments:
            diag.warnings.append(f"incomplete tool_call discarded: index={idx}")
            return []
        try:
            args = json.loads(item.arguments_text)
        except json.JSONDecodeError:
            diag.warnings.append(f"tool_call arguments not valid JSON: index={idx}")
            return []
        if not isinstance(args, dict):
            diag.warnings.append(f"tool_call arguments not an object: index={idx}")
            return []
        parsed.append(ToolCall(id=item.id, name=item.name, arguments=args))
    return parsed
