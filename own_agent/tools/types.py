from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    categories: tuple[str, ...] = ()
    permission: str | None = None


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: str | None = None
    output: str = ""


ToolFunc = Callable[..., str | ToolResult]
