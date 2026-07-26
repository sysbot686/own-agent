from __future__ import annotations

from typing import Any

from own_agent.tools.context import ExecutionContext
from own_agent.tools.types import ToolFunc, ToolResult, ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._impls: dict[str, ToolFunc] = {}

    def register(self, spec: ToolSpec, impl: ToolFunc) -> None:
        self._tools[spec.name] = spec
        self._impls[spec.name] = impl

    def get(self, name: str) -> tuple[ToolSpec, ToolFunc] | None:
        spec = self._tools.get(name)
        impl = self._impls.get(name)
        if spec is None or impl is None:
            return None
        return spec, impl

    def list_specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def has(self, name: str) -> bool:
        return name in self._tools

    def call(self, name: str, ctx: ExecutionContext | None = None, **kwargs: Any) -> ToolResult | str:
        pair = self.get(name)
        if pair is None:
            return ToolResult(success=False, error=f"unknown tool '{name}'")
        spec, impl = pair

        if spec.permission and ctx is not None and ctx.request_approval is not None:
            details = f"{name}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})"
            approved = ctx.request_approval(
                f"Tool: {name} ({spec.permission})",
                details,
            )
            if not approved:
                return ToolResult(success=False, error=f"permission denied for tool '{name}'")

        try:
            result = impl(ctx=ctx, **kwargs)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        if isinstance(result, ToolResult):
            return result
        output = str(result)
        if output.startswith("Error:"):
            return ToolResult(success=False, error=output)
        return ToolResult(success=True, output=output)
