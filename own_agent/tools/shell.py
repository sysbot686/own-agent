from __future__ import annotations

import io
import re
import subprocess
import sys

from own_agent.tools.types import ToolSpec

_DANGEROUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\brm\s+(-rf?|--recursive)\s+/?\s*$', re.I), "recursive delete from root"),
    (re.compile(r'\brm\s+(-rf?|--recursive)\s+~', re.I), "recursive delete from home"),
    (re.compile(r'\b(rm\s+-rf?\s+\*|rm\s+-rf?\s+\.)', re.I), "recursive delete of all files"),
    (re.compile(r'\b(rd|rmdir)\s+/[sSqQ]', re.I), "recursive directory delete"),
    (re.compile(r'\bdel\s+/[fFsSqQ]', re.I), "force file delete"),
    (re.compile(r'\bformat\b', re.I), "disk format"),
    (re.compile(r'\bdiskpart\b', re.I), "disk partition"),
    (re.compile(r'\bdd\s+.*of=', re.I), "raw disk write"),
    (re.compile(r'\bwget\s+.*\|', re.I), "piped remote download to shell"),
    (re.compile(r'\bcurl\s+.*\|', re.I), "piped remote download to shell"),
    (re.compile(r'\bsudo\s+rm\b', re.I), "sudo recursive delete"),
    (re.compile(r'\bsudo\s+dd\b', re.I), "sudo raw disk write"),
    (re.compile(r'>\s*\\\\\.\\(PHYSICALDRIVE|GLOBALROOT)', re.I), "raw disk write on Windows"),
    (re.compile(r'\bchmod\s+-R\s+777\s+/', re.I), "recursive permission change from root"),
    (re.compile(r'\bmkfs\.', re.I), "filesystem creation"),
    (re.compile(r'^\s*:\s*\(\s*\)\s*\{', re.I), "fork bomb"),
    (re.compile(r'\breboot\b', re.I), "system reboot"),
    (re.compile(r'\bshutdown\b', re.I), "system shutdown"),
    (re.compile(r'\bpoweroff\b', re.I), "system poweroff"),
    (re.compile(r'\binit\s+0\b', re.I), "system shutdown via init"),
    (re.compile(r'\btakeown\b', re.I), "take ownership of system files"),
    (re.compile(r'\brestrict\s+/[fF]', re.I), "Windows ACL reset"),
    (re.compile(r'\breg\s+delete\b', re.I), "registry key delete"),
]

_SAFE_MODULES = frozenset({
    "ast", "base64", "collections", "datetime", "functools",
    "glob", "hashlib", "itertools", "json", "math", "pathlib",
    "random", "re", "statistics", "string", "textwrap", "typing", "uuid",
    "io", "decimal", "fractions", "pprint", "copy", "enum",
})

def _safe_import(name: str, *args, **kwargs) -> object:
    base = name.split(".")[0]
    if base not in _SAFE_MODULES:
        raise ImportError(f"module '{name}' is not allowed in restricted mode")
    return __import__(name, *args, **kwargs)

_RESTRICTED_BUILTINS: dict[str, object] = {
    "abs": abs, "all": all, "any": any, "ascii": ascii, "__import__": _safe_import,
    "bin": bin, "bool": bool, "bytearray": bytearray, "bytes": bytes,
    "callable": callable, "chr": chr, "complex": complex,
    "dict": dict, "dir": dir, "divmod": divmod, "enumerate": enumerate,
    "filter": filter, "float": float, "format": format, "frozenset": frozenset,
    "getattr": getattr, "hasattr": hasattr, "hash": hash, "hex": hex,
    "id": id, "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "map": map, "max": max,
    "min": min, "next": next, "object": object, "oct": oct, "ord": ord,
    "pow": pow, "print": print, "range": range, "repr": repr,
    "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum,
    "super": super, "tuple": tuple, "type": type, "vars": vars,
    "zip": zip, "True": True, "False": False, "None": None,
    "hasattr": hasattr, "setattr": setattr, "delattr": delattr,
}

_PYTHON_SAFE_IMPORTS = (
    "import ast, base64, collections, datetime, functools, glob, hashlib, itertools, "
    "json, math, pathlib, random, re, statistics, textwrap, typing, uuid"
)


def _check_dangerous_command(command: str) -> str | None:
    for pattern, reason in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            return f"Blocked potentially destructive command ({reason}): {command[:200]}"
    return None


def shell(
    command: str,
    description: str = "",
    timeout: int = 30,
    workdir: str | None = None,
    **kwargs,
) -> str:
    blocked = _check_dangerous_command(command)
    if blocked:
        return blocked

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
    description="Execute a shell command. Dangerous commands (rm -rf /, format, etc.) are blocked automatically.",
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
    namespace: dict[str, object] = {}
    _buf = io.StringIO()
    _old = sys.stdout
    sys.stdout = _buf
    try:
        exec(f"{_PYTHON_SAFE_IMPORTS}\n{code}", {"__builtins__": _RESTRICTED_BUILTINS}, namespace)
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
    description="Execute Python code in a restricted environment (os/sys removed, basic builtins only). Captures stdout; set `_result` for explicit return.",
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
