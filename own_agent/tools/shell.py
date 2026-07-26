from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import textwrap

from own_agent.tools.types import ToolSpec


def shell(
    command: str,
    description: str = "",
    timeout: int = 30,
    workdir: str | None = None,
    **kwargs,
) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s:\n{command[:200]}"
    except Exception as exc:
        return f"Error: {exc}"

    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        if output:
            output += "\n--- stderr ---\n"
        output += result.stderr
    if result.returncode != 0:
        output = f"Exit code: {result.returncode}\n{output}"
    if not output:
        output = f"(completed with exit code {result.returncode}, no output)"

    return output[:20000] + ("\n... (truncated)" if len(output) > 20000 else "")


SHELL_SPEC = ToolSpec(
    name="shell",
    description="Execute a shell command. Provide a clear description of what the command does for security review.",
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "description": {
                "type": "string",
                "description": "Short (5-10 words) explanation of what this command does. Required for permission approval.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 30).",
                "default": 30,
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for the command (default: current directory).",
            },
        },
        "required": ["command", "description"],
    },
    categories=("execution",),
    permission="shell",
)


def python_exec(code: str, **kwargs) -> str:
    imports = (
        "import ast, base64, collections, datetime, functools, glob, hashlib, itertools, "
        "json, math, os, pathlib, random, re, statistics, sys, textwrap, typing, urllib, uuid"
    )
    namespace: dict[str, object] = {}
    _buf = io.StringIO()
    _old = sys.stdout
    sys.stdout = _buf
    try:
        exec(f"{imports}\n{code}", namespace)
        _captured = _buf.getvalue()
    except Exception as exc:
        _buf.close()
        sys.stdout = _old
        return f"Error: {exc}"
    sys.stdout = _old
    _buf.close()

    result = namespace.get("_result")
    if result is not None:
        return str(result).rstrip()
    if _captured:
        return _captured.rstrip()
    return "(no result)"


PYTHON_EXEC_SPEC = ToolSpec(
    name="python_exec",
    description="Execute Python code in an isolated sandbox. Captures stdout; set `_result` for explicit return.",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Set `_result` to return a value.",
            },
        },
        "required": ["code"],
    },
    categories=("execution", "development"),
    permission="shell",
)
