from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

from own_agent.tools.types import ToolSpec


@dataclass
class McpServerConfig:
    command: str
    args: list[str] = list
    env: dict[str, str] | None = None


class McpClient:
    def __init__(self, config: McpServerConfig) -> None:
        self._config = config
        self._proc: subprocess.Popen | None = None

    async def start(self) -> None:
        self._proc = subprocess.Popen(
            [self._config.command, *self._config.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    async def list_tools(self) -> list[ToolSpec]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return f"(MCP tool '{name}' not implemented)"

    async def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc = None

    @property
    def connected(self) -> bool:
        return self._proc is not None
