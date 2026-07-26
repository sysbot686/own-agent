from own_agent.tools.context import ExecutionContext
from own_agent.tools.filesystem import (
    EDIT_SPEC, GREP_SPEC, GLOB_SPEC, LS_SPEC, VIEW_SPEC, WRITE_SPEC,
    edit, glob, grep, ls, view, write,
)
from own_agent.tools.registry import ToolRegistry
from own_agent.tools.shell import PYTHON_EXEC_SPEC, SHELL_SPEC, python_exec, shell
from own_agent.tools.think import SPEC as THINK_SPEC, think
from own_agent.tools.types import ToolResult, ToolSpec

__all__ = [
    "ToolSpec", "ToolResult", "ToolRegistry", "ExecutionContext",
    "think", "ls", "view", "write", "edit", "grep", "glob", "shell", "python_exec",
]


def register_all_tools(registry: ToolRegistry) -> None:
    registry.register(THINK_SPEC, think)
    registry.register(LS_SPEC, ls)
    registry.register(VIEW_SPEC, view)
    registry.register(WRITE_SPEC, write)
    registry.register(EDIT_SPEC, edit)
    registry.register(GREP_SPEC, grep)
    registry.register(GLOB_SPEC, glob)
    registry.register(SHELL_SPEC, shell)
    registry.register(PYTHON_EXEC_SPEC, python_exec)
