from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from own_agent.tools.types import ToolSpec


@dataclass
class McpServerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


class McpError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"[{code}] {message}")


class McpClient:
    def __init__(self, config: McpServerConfig, request_timeout: float = 30.0) -> None:
        self._config = config
        self._request_timeout = request_timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._capabilities: dict[str, Any] = {}

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            self._config.command,
            *self._config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**self._config.env} if self._config.env else None,
        )
        assert self._proc.stdout is not None
        assert self._proc.stdin is not None
        self._reader = self._proc.stdout
        self._writer = self._proc.stdin
        self._reader_task = asyncio.create_task(self._read_loop())

    async def initialize(self) -> dict[str, Any]:
        result = await self._request("initialize", {
            "protocolVersion": "2024-11-20",
            "capabilities": {},
            "clientInfo": {"name": "own-agent", "version": "0.1.0"},
        })
        self._capabilities = result.get("capabilities", {})
        return result

    async def list_tools(self) -> list[ToolSpec]:
        result = await self._request("tools/list", {})
        raw_tools: list[dict[str, Any]] = result.get("tools", [])
        specs: list[ToolSpec] = []
        for t in raw_tools:
            specs.append(ToolSpec(
                name=t.get("name", ""),
                description=t.get("description", ""),
                parameters=t.get("inputSchema", t.get("parameters", {})),
                categories=("mcp",),
            ))
        return specs

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._request("tools/call", {"name": name, "arguments": arguments})
        content: list[dict[str, Any]] = result.get("content", [])
        parts: list[str] = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "resource":
                parts.append(str(item.get("resource", "")))
        return "\n".join(parts)

    async def stop(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._writer is not None and not self._writer.is_closing():
            self._writer.close()
        if self._proc is not None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                self._proc.kill()
                await self._proc.wait()
        self._proc = None
        self._reader = None
        self._writer = None
        self._reader_task = None

        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

    @property
    def connected(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.connected:
            raise McpError(-32000, "not connected")

        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut

        try:
            msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}, ensure_ascii=False)
            assert self._writer is not None
            self._writer.write((msg + "\n").encode("utf-8"))
            await self._writer.drain()

            result = await asyncio.wait_for(fut, timeout=self._request_timeout)
            return result
        finally:
            self._pending.pop(req_id, None)

    async def _read_loop(self) -> None:
        assert self._reader is not None
        buffer = ""
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break
                buffer += line.decode("utf-8")
                while "\n" in buffer:
                    raw, buffer = buffer.split("\n", 1)
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(msg, dict) and "id" in msg:
                        req_id = msg["id"]
                        if isinstance(req_id, int) and req_id in self._pending:
                            fut = self._pending[req_id]
                            if not fut.done():
                                if "error" in msg:
                                    err = msg["error"]
                                    fut.set_exception(McpError(err.get("code", 0), err.get("message", ""), err.get("data")))
                                else:
                                    fut.set_result(msg.get("result", {}))
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.cancel()
            self._pending.clear()
