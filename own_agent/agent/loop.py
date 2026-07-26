from __future__ import annotations

from collections.abc import AsyncIterator

from own_agent.agent.types import AgentConfig, AgentEvent
from own_agent.context.rag.manager import RagManager
from own_agent.permissions.manager import PermissionManager
from own_agent.providers.base import ChatProvider
from own_agent.providers.types import (
    ChatMessage, ChatRequest, FinishReason, ToolCall, ToolChoice, ToolChoiceFunction, ToolDefinition,
    TokenUsage,
)
from own_agent.session.manager import SessionManager
from own_agent.tools.context import ExecutionContext
from own_agent.tools.registry import ToolRegistry
from own_agent.tools.types import ToolResult, ToolSpec

SYSTEM_PROMPT = """You are own-agent, a coding agent that helps users with software engineering tasks.
You have access to a set of tools to explore codebases, read and edit files, and run commands.

Available tools:
{tools_description}

Instructions:
1. Think through problems step by step before taking action
2. Use the think tool to record your reasoning
3. Always provide clear descriptions for shell commands
4. When editing files, prefer targeted edits over full rewrites
5. If you encounter errors, analyze and fix them
6. When a task is complete, summarize what was done
"""


class Agent:
    def __init__(
        self,
        provider: ChatProvider,
        tool_registry: ToolRegistry,
        session_manager: SessionManager,
        permission_manager: PermissionManager,
        config: AgentConfig | None = None,
    ) -> None:
        self._provider = provider
        self._tools = tool_registry
        self._sessions = session_manager
        self._permissions = permission_manager
        self._config = config or AgentConfig()
        self._rag: RagManager | None = None

    def set_rag(self, rag: RagManager | None) -> None:
        self._rag = rag

    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        self._sessions.add_message(ChatMessage(role="user", content=user_input))

        tool_errors = 0
        tool_rounds = 0
        all_usage: list[TokenUsage | None] = []

        while tool_rounds < self._config.max_tool_rounds:
            tool_rounds += 1

            system_msg = ChatMessage(
                role="system",
                content=self._build_system_prompt(),
            )

            rag_context = ""
            if self._rag:
                rag_context = await self._rag.retrieve_context(user_input)

            messages = [system_msg]
            if rag_context:
                messages.append(ChatMessage(role="system", content=rag_context))
            messages += self._sessions.all_messages()

            request = ChatRequest(
                messages=messages,
                tools=self._get_tool_definitions(),
                tool_choice="auto",
                temperature=0.0,
                max_tokens=16384,
            )

            events: list[AgentEvent] = []
            response = None
            reasoning_text = ""

            try:
                async for event in self._provider.astream(request):
                    if event.kind == "reasoning_delta":
                        reasoning_text += event.text or ""
                    elif event.kind == "text_delta":
                        if not events or events[-1].kind != "text":
                            events.append(AgentEvent(kind="text"))
                        events[-1].text += event.text or ""
                    elif event.kind == "tool_call_started":
                        yield AgentEvent(
                            kind="tool_call", tool_call_id=event.tool_call_id or "",
                            tool_name=event.tool_name or "",
                        )
                    elif event.kind == "message_completed":
                        response = event.response
                        if response.usage:
                            all_usage.append(response.usage)
            except Exception as exc:
                yield AgentEvent(kind="error", error=str(exc))
                return

            if response is None:
                yield AgentEvent(kind="error", error="no response from provider")
                return

            finish = response.finish_reason

            if reasoning_text:
                yield AgentEvent(kind="reasoning", text=reasoning_text)

            for ev in events:
                if ev.kind == "text" and ev.text:
                    yield AgentEvent(kind="text", text=ev.text)

            assistant_msg = ChatMessage(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls or None,
            )
            self._sessions.add_message(assistant_msg)

            if finish == "stop":
                yield AgentEvent(kind="done", finish_reason="stop", usage=self._combine_usage(all_usage))
                return

            if finish == "length":
                if response.content:
                    yield AgentEvent(kind="text", text=response.content + "\n\n[Response truncated due to length]")
                yield AgentEvent(kind="done", finish_reason="length", usage=self._combine_usage(all_usage))
                return

            if finish == "tool_calls" and response.tool_calls:
                ctx = ExecutionContext(
                    cwd=self._sessions.current.metadata.get("cwd", ".") if self._sessions.current else ".",
                    workspace_root=".",
                    request_approval=self._permissions.make_callback(),
                )

                for tc in response.tool_calls:
                    result = self._tools.call(tc.name, ctx=ctx, **tc.arguments)
                    yield AgentEvent(kind="tool_result", tool_call_id=tc.id, tool_name=tc.name, tool_result=result)
                    if result.startswith("Error"):
                        tool_errors += 1

                    self._sessions.add_message(ChatMessage(
                        role="tool",
                        content=result[:10000],
                        tool_call_id=tc.id,
                    ))

                if tool_errors >= self._config.max_tool_errors:
                    yield AgentEvent(kind="error", error=f"too many tool errors ({tool_errors})")
                    yield AgentEvent(kind="done", finish_reason="error", usage=self._combine_usage(all_usage))
                    return
                continue

            if finish == "content_filter":
                yield AgentEvent(kind="error", error="response filtered by content policy")
                yield AgentEvent(kind="done", finish_reason="content_filter", usage=self._combine_usage(all_usage))
                return

            yield AgentEvent(kind="error", error=f"unknown finish_reason: {finish}")
            yield AgentEvent(kind="done", finish_reason="error", usage=self._combine_usage(all_usage))
            return

        yield AgentEvent(kind="error", error=f"exceeded max tool rounds ({self._config.max_tool_rounds})")
        yield AgentEvent(kind="done", finish_reason="error", usage=self._combine_usage(all_usage))

    def _build_system_prompt(self) -> str:
        tools_desc = "\n".join(
            f"- {s.name}: {s.description}"
            for s in self._tools.list_specs()
        )
        return SYSTEM_PROMPT.format(tools_description=tools_desc)

    def _get_tool_definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=s.name,
                description=s.description,
                parameters=s.parameters,
            )
            for s in self._tools.list_specs()
        ]

    @staticmethod
    def _combine_usage(all_usage: list[TokenUsage | None]) -> dict | None:
        if not all_usage:
            return None
        inp = sum(u.input_tokens or 0 for u in all_usage if u is not None)
        out = sum(u.output_tokens or 0 for u in all_usage if u is not None)
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}
