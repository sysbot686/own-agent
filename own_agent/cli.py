from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from own_agent.agent.loop import Agent
from own_agent.agent.types import AgentConfig
from own_agent.config import load_config
from own_agent.context.rag.manager import RagConfig, RagManager
from own_agent.permissions.manager import PermissionManager
from own_agent.permissions.types import PermissionMode
from own_agent.providers.presets import ProviderPreset, get_preset
from own_agent.session.manager import SessionManager
from own_agent.session.store import SessionStore
from own_agent.tools import ToolRegistry, register_all_tools

console = Console()


def _build_provider(config, model: str | None = None):
    pname = config.provider_name
    preset: ProviderPreset | None = get_preset(pname)
    if preset is None:
        console.print(f"[red]Unknown provider: {pname}[/]")
        sys.exit(1)

    api_key = config.get_provider_value(
        "api_key", env=preset.api_key_env,
    ) or ""
    base_url = config.get_provider_value(
        "base_url", env=preset.base_url_env, default=preset.default_base_url,
    )
    resolved_model = model or config.get_provider_value(
        "model", default=preset.default_model,
    ) or preset.default_model

    if pname == "anthropic":
        from own_agent.providers.anthropic import AnthropicProvider
        return AnthropicProvider(
            name=pname, model=resolved_model,
            api_key=api_key, base_url=base_url,
        )
    else:
        from own_agent.providers.openai_compatible import OpenAICompatibleProvider
        return OpenAICompatibleProvider(
            name=pname, model=resolved_model,
            api_key=api_key, base_url=base_url,
        )


def _get_permission_mode(config) -> PermissionMode:
    mode_str = config.get_config_value("permissions.mode", default="standard")
    try:
        return PermissionMode(mode_str)
    except ValueError:
        return PermissionMode.STANDARD


def _interactive_approval(action: str, details: str, _mode) -> bool:
    console.print()
    console.print(Panel(f"[yellow]Permission Request[/]\n[b]{action}[/]\n{details[:500]}", title="⚠️  Approval"))
    while True:
        response = input("  Allow? (y/N/a=always/?: ") or "n"
        r = response.strip().lower()
        if r in ("y", "yes"):
            return True
        if r in ("a", "always"):
            return True
        if r in ("n", "no", ""):
            return False
        console.print("  y=yes, N=no, a=always")


def print_welcome():
    console.print(Panel.fit(
        "[bold cyan]own-agent[/] — a local coding agent\n"
        "Type [bold]/help[/] for commands, [bold]/exit[/] to quit",
        title="Welcome",
    ))


def print_help():
    console.print(Panel.fit(
        "[bold]Commands[/]\n"
        "  /help        Show this help\n"
        "  /exit, /q    Quit\n"
        "  /new         Start a new session\n"
        "  /sessions    List sessions\n"
        "  /resume <id> Resume a session\n"
        "  /model <m>   Switch model\n"
        "  /clear       Clear screen\n"
        "  /status      Show session info\n"
        "  /reindex     Re-index project for RAG\n"
        "  /rag on/off  Enable/disable RAG\n"
    ))


def print_sessions(sessions: list[dict]):
    if not sessions:
        console.print("[dim]No sessions yet[/]")
        return
    table = Table(title="Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Title")
    table.add_column("Messages")
    table.add_column("Updated")
    for s in sessions:
        mid = s["id"][:8]
        msg_c = s.get("message_count", 0)
        updated = s.get("updated_at", "")[:19] if s.get("updated_at") else ""
        table.add_row(mid, s.get("title", "") or "(untitled)", str(msg_c), updated)
    console.print(table)


async def ainput(prompt: str = "") -> str:
    try:
        return await asyncio.to_thread(input, prompt)
    except asyncio.CancelledError:
        raise KeyboardInterrupt()


async def run_cli(model: str | None = None, prompt: str | None = None) -> None:
    config = load_config()

    provider = _build_provider(config, model=model)
    perm_mode = _get_permission_mode(config)
    perm_mgr = PermissionManager(mode=perm_mode, on_request=_interactive_approval)

    tool_registry = ToolRegistry()
    register_all_tools(tool_registry)

    session_store = SessionStore()
    session_mgr = SessionManager(session_store)
    session_mgr.new(title="main")

    rag_mgr = RagManager(project_root=".")
    rag_enabled = True
    try:
        await rag_mgr.index_project()
        if rag_mgr.total_chunks > 0:
            console.print(f"[dim]Indexed {rag_mgr.total_chunks} chunks for RAG[/]")
    except Exception as exc:
        console.print(f"[dim]RAG index failed: {exc}[/]")

    agent_config = AgentConfig()
    agent = Agent(
        provider=provider,
        tool_registry=tool_registry,
        session_manager=session_mgr,
        permission_manager=perm_mgr,
        config=agent_config,
    )
    agent.set_rag(rag_mgr if rag_enabled else None)

    print_welcome()

    if prompt:
        await _run_single_prompt(agent, prompt)
        return

    while True:
        try:
            user_input = await ainput("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/]")
            break

        text = user_input.strip()
        if not text:
            continue

        if text == "/exit" or text == "/q":
            break
        if text == "/help":
            print_help()
            continue
        if text == "/new":
            session_mgr.new()
            console.print("[green]New session started[/]")
            continue
        if text == "/sessions":
            print_sessions(session_mgr.list_sessions())
            continue
        if text.startswith("/resume "):
            sid = text[8:].strip()
            if session_mgr.resume(sid):
                console.print(f"[green]Resumed session {sid[:8]}[/]")
            else:
                console.print(f"[red]Session not found: {sid}[/]")
            continue
        if text == "/clear":
            os.system("cls" if sys.platform == "win32" else "clear")
            continue
        if text == "/status":
            cur = session_mgr.current
            if cur:
                console.print(f"[cyan]Session:[/] {cur.id[:8]} | [cyan]Messages:[/] {len(cur.messages)}")
            else:
                console.print("[dim]No active session[/]")
            continue
        if text.startswith("/model "):
            new_model = text[7:].strip()
            if new_model:
                provider = _build_provider(config, model=new_model)
                agent._provider = provider
                console.print(f"[green]Switched to model: {new_model}[/]")
            continue
        if text == "/reindex":
            try:
                await rag_mgr.index_project()
                console.print(f"[green]Re-indexed {rag_mgr.total_chunks} chunks[/]")
            except Exception as exc:
                console.print(f"[red]Re-index failed: {exc}[/]")
            continue
        if text == "/rag on":
            rag_enabled = True
            agent.set_rag(rag_mgr)
            console.print("[green]RAG enabled[/]")
            continue
        if text == "/rag off":
            rag_enabled = False
            agent.set_rag(None)
            console.print("[yellow]RAG disabled[/]")
            continue
        if text.startswith("/"):
            console.print(f"[red]Unknown command: {text}[/]")
            continue

        try:
            await _run_single_prompt(agent, text)
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[yellow]Interrupted[/]")


async def _run_single_prompt(agent: Agent, text: str) -> None:
    try:
        async for event in agent.run(text):
            if event.kind == "text":
                console.print(Markdown(event.text) if event.text.strip().startswith(("#", "-", "1.")) else event.text, end="")
            elif event.kind == "reasoning":
                console.print(f"[dim]🧠 {event.text}[/]")
            elif event.kind == "tool_call":
                console.print(f"[cyan]🔧 {event.tool_name}(...)[/]")
            elif event.kind == "tool_result":
                result = event.tool_result[:500]
                if event.tool_result.startswith("Error"):
                    console.print(f"  [red]→ {result}[/]")
                else:
                    console.print(f"  [green]→[/] [dim]{result}[/]")
            elif event.kind == "error":
                console.print(f"[red]Error: {event.error}[/]")
            elif event.kind == "done" and event.usage:
                u = event.usage
                console.print(f"\n[dim]▲ {u.get('input_tokens', '?')} in / {u.get('output_tokens', '?')} out / {u.get('total_tokens', '?')} total[/]")
    except Exception as exc:
        console.print(f"[red]Agent error: {exc}[/]")


def main() -> int:
    args = [a for a in sys.argv[1:] if a]
    model: str | None = None
    prompt: str | None = None
    i = 0
    while i < len(args):
        if args[i] in ("-m", "--model") and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] in ("-p", "--prompt") and i + 1 < len(args):
            prompt = args[i + 1]
            i += 2
        elif args[i] in ("-h", "--help"):
            print("Usage: ownagent [--model NAME] [--prompt TEXT]")
            print("  --model, -m     Model name (e.g. deepseek/deepseek-chat)")
            print("  --prompt, -p    Run a single prompt non-interactively")
            return 0
        else:
            i += 1

    try:
        asyncio.run(run_cli(model=model, prompt=prompt))
    except KeyboardInterrupt:
        pass
    return 0
