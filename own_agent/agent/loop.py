from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from own_agent.agent.types import AgentConfig, AgentEvent
from own_agent.context.manager import ContextManager
from own_agent.context.rag.manager import RagManager
from own_agent.permissions.manager import PermissionManager
from own_agent.providers.base import ChatProvider
from own_agent.providers.errors import ProviderError, classify_exception, ProviderErrorKind
from own_agent.providers.types import (
    ChatMessage, ChatRequest, ChatStreamEvent, FinishReason, ToolCall, ToolChoice, ToolChoiceFunction,
    ToolDefinition, TokenUsage,
)
from own_agent.session.manager import SessionManager
from own_agent.skills.loader import Skill
from own_agent.tools.context import ExecutionContext
from own_agent.tools.registry import ToolRegistry
from own_agent.tools.types import ToolResult, ToolSpec

SUMMARY_PROMPT = """Summarize the following conversation between a user and an AI coding agent.
Focus on: what tasks were completed, what files were modified, what decisions were made.
Keep the summary concise (2-4 sentences).

Conversation:
{conversation}
"""

REPLAN_HINT = "(You have been working for several turns without updating your plan. Consider reviewing progress or creating a plan if the task is complex.)"
PLAN_HINT_INTERVAL = 5

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
7. For complex tasks, use the plan tool to break down the work before starting. Update your plan as you make progress.
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
        self._skills: list[Skill] = []
        self._tool_defs: list[ToolDefinition] | None = None
        self._cached_system_prompt: str | None = None
        self._ctx_mgr = ContextManager(
            max_tokens=self._config.context_window,
            llm_summarize=self._summarize,
        )
        self._rounds_without_plan = 0

    async def _summarize(self, messages: list[ChatMessage]) -> str:
        text = "\n".join(f"{m.role}: {m.content[:200]}" for m in messages if m.content)
        prompt = SUMMARY_PROMPT.format(conversation=text[:4000])
        try:
            response = await self._provider.acomplete(ChatRequest(
                messages=[ChatMessage(role="user", content=prompt)],
                max_tokens=512,
            ))
            return response.content
        except Exception:
            return ""

    def set_rag(self, rag: RagManager | None) -> None:
        self._rag = rag

    def set_provider(self, provider: ChatProvider) -> None:
        self._provider = provider

    def set_skills(self, skills: list[Skill]) -> None:
        self._skills = skills
        self._cached_system_prompt = None

    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        self._sessions.add_message(ChatMessage(role="user", content=user_input))
        self._ctx_mgr.check(self._sessions.all_messages())

        tool_errors = 0
        tool_rounds = 0
        all_usage: list[TokenUsage | None] = []

        while tool_rounds < self._config.max_tool_rounds:
            tool_rounds += 1

            system_msg = ChatMessage(
                role="system",
                content=self._build_system_prompt(),
            )

            rag_query = user_input
            if self._rag and tool_rounds > 1:
                recent = self._sessions.all_messages()[-4:]
                rag_query = " ".join(m.content for m in recent if m.content and m.role in ("user", "assistant")) or user_input
            rag_context = ""
            if self._rag:
                rag_context = await self._rag.retrieve_context(rag_query)

            messages = [system_msg]
            if rag_context:
                messages.append(ChatMessage(role="system", content=rag_context))
            plan_text = (
                self._sessions.current.metadata.get("plan", "")
                if self._sessions.current else ""
            )
            if plan_text:
                messages.append(ChatMessage(role="system", content=f"## Current Plan\n{plan_text}"))
            if self._rounds_without_plan >= PLAN_HINT_INTERVAL:
                messages.append(ChatMessage(role="system", content=REPLAN_HINT))
            chat_msgs = await self._ctx_mgr.compress(self._sessions.all_messages())
            messages += chat_msgs

            request = ChatRequest(
                messages=messages,
                tools=self._get_tool_definitions(),
                tool_choice="auto",
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )

            events: list[AgentEvent] = []
            response = None
            reasoning_text = ""
            _RETRYABLE = frozenset({ProviderErrorKind.TIMEOUT, ProviderErrorKind.RATE_LIMIT, ProviderErrorKind.NETWORK_ERROR, ProviderErrorKind.SERVER_ERROR})

            for attempt in range(1 + self._config.max_retries):
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
                    break  # success
                except Exception as exc:
                    if isinstance(exc, ProviderError):
                        can_retry = exc.retryable
                    else:
                        can_retry = classify_exception(exc) in _RETRYABLE
                    if not can_retry or attempt >= self._config.max_retries:
                        yield AgentEvent(kind="error", error=str(exc))
                        return
                    yield AgentEvent(kind="text", text=f"\n[retry {attempt + 1}/{self._config.max_retries}...]\n")
                    events = []
                    reasoning_text = ""
                    response = None
                    await asyncio.sleep(self._config.retry_delay)

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
            self._ctx_mgr.check(self._sessions.all_messages())

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
                    if tc.name == "plan":
                        plan_content = tc.arguments.get("plan_text", "")
                        if self._sessions.current:
                            self._sessions.current.metadata["plan"] = plan_content
                            self._sessions.save()
                        self._rounds_without_plan = 0
                    tool_result = await asyncio.to_thread(self._tools.call, tc.name, ctx=ctx, **tc.arguments)
                    if isinstance(tool_result, ToolResult) and not tool_result.success:
                        content = f"Error: {tool_result.error}"
                        tool_errors += 1
                    elif isinstance(tool_result, ToolResult):
                        content = tool_result.output
                    else:
                        content = str(tool_result)
                    yield AgentEvent(kind="tool_result", tool_call_id=tc.id, tool_name=tc.name, tool_result=content)
                    self._sessions.add_message(ChatMessage(
                        role="tool",
                        content=content[:10000],
                        tool_call_id=tc.id,
                    ))
                    self._ctx_mgr.check(self._sessions.all_messages())

                if not any(tc.name == "plan" for tc in response.tool_calls):
                    self._rounds_without_plan += 1

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
        if self._cached_system_prompt is None:
            tools_desc = "\n".join(
                f"- {s.name}: {s.description}"
                for s in self._tools.list_specs()
            )
            skills_block = ""
            if self._skills:
                parts = []
                for sk in self._skills:
                    header = f"## {sk.name}" + (f": {sk.description}" if sk.description else "")
                    parts.append(f"{header}\n{sk.content}")
                skills_block = "\n\n---\nLoaded skills:\n" + "\n\n".join(parts)
            self._cached_system_prompt = SYSTEM_PROMPT.format(tools_description=tools_desc) + skills_block
        return self._cached_system_prompt

    def _get_tool_definitions(self) -> list[ToolDefinition]:
        if self._tool_defs is None:
            self._tool_defs = [
                ToolDefinition(name=s.name, description=s.description, parameters=s.parameters)
                for s in self._tools.list_specs()
            ]
        return self._tool_defs

    @staticmethod
    def _combine_usage(all_usage: list[TokenUsage | None]) -> dict | None:
        if not all_usage:
            return None
        inp = sum(u.input_tokens or 0 for u in all_usage if u is not None)
        out = sum(u.output_tokens or 0 for u in all_usage if u is not None)
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}
