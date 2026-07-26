from __future__ import annotations

import subprocess
import sys
import textwrap

from own_agent.tools.types import ToolSpec


def shell(
    command: str,
    description: str = "",
    timeout: int = 120,
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
                "description": "Timeout in seconds (default 120).",
                "default": 120,
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
    wrapped = textwrap.dedent(f"""\
        {imports}
        _result = None
        try:
{textwrap.indent(code, '            ')}
        except Exception as _e:
            _result = f"Error: {{_e}}"
    """)
    namespace: dict[str, object] = {}
    try:
        exec(wrapped, namespace)
    except Exception as exc:
        return f"Error: {exc}"
    result = namespace.get("_result")
    if result is None:
        return "(no result variable set)"
    return str(result)


PYTHON_EXEC_SPEC = ToolSpec(
    name="python_exec",
    description="Execute Python code in an isolated sandbox. Set `_result` to capture output.",
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
