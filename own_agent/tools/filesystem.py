from __future__ import annotations

import os
import re
from pathlib import Path

from own_agent.tools.context import ExecutionContext
from own_agent.tools.types import ToolResult, ToolSpec


def ls(path: str = "", **kwargs) -> str:
    target = Path(path) if path else Path.cwd()
    if not target.exists():
        return f"Error: path not found: {target}"
    if target.is_file():
        return str(target)
    entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines: list[str] = []
    for e in entries:
        suffix = "/" if e.is_dir() else ""
        lines.append(f"{e.name}{suffix}")
    return "\n".join(lines) if lines else "(empty directory)"


LS_SPEC = ToolSpec(
    name="ls",
    description="List files and directories at the given path.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (default: current directory).",
                "default": "",
            },
        },
    },
    categories=("filesystem",),
)


def view(file_path: str, offset: int = 1, limit: int = 200, **kwargs) -> str:
    target = Path(file_path)
    if not target.exists():
        return f"Error: file not found: {file_path}"
    if target.is_dir():
        return f"Error: {file_path} is a directory, not a file"

    try:
        text = target.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error reading file: {exc}"

    lines = text.splitlines()
    total = len(lines)
    start = max(0, offset - 1)
    end = start + limit
    selected = lines[start:end]

    out = f"File: {file_path} ({total} lines)"
    if start > 0 or end < total:
        out += f" [lines {start + 1}-{min(end, total)}]"
    out += "\n" + "-" * 40 + "\n"
    for i, line in enumerate(selected, start=start + 1):
        out += f"{i:6d}: {line}\n"
    if end < total:
        out += "... (use offset={} to see next chunk)\n".format(end + 1)
    return out


VIEW_SPEC = ToolSpec(
    name="view",
    description="Read a file with optional line range. Lines are numbered for reference.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to read.",
            },
            "offset": {
                "type": "integer",
                "description": "Starting line number (1-indexed, default 1).",
                "default": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum lines to read (default 200).",
                "default": 200,
            },
        },
        "required": ["file_path"],
    },
    categories=("filesystem",),
)


def write(file_path: str, content: str, ctx: ExecutionContext | None = None, **kwargs) -> str:
    target = Path(file_path)

    if target.exists() and not target.is_file():
        return f"Error: {file_path} exists but is not a file"

    if ctx is not None and ctx.request_approval is not None:
        exists = target.exists()
        details = f"{'Overwrite' if exists else 'Create'}: {file_path} ({len(content)} chars)"
        approved = ctx.request_approval("Tool: write", details)
        if not approved:
            return f"Error: permission denied for write({file_path})"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"Error writing file: {exc}"

    return f"Written {len(content)} chars to {file_path}"


WRITE_SPEC = ToolSpec(
    name="write",
    description="Write content to a file, creating parent directories if needed.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path of the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Full content to write to the file.",
            },
        },
        "required": ["file_path", "content"],
    },
    categories=("filesystem",),
    permission="write",
)


def edit(file_path: str, old_string: str, new_string: str, ctx: ExecutionContext | None = None, **kwargs) -> str:
    target = Path(file_path)
    if not target.exists():
        return f"Error: file not found: {file_path}"

    try:
        text = target.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error reading file: {exc}"

    count = text.count(old_string)
    if count == 0:
        return f"Error: old_string not found in {file_path}"
    if count > 1 and "replace_all" not in kwargs:
        return (
            f"Error: old_string appears {count} times in {file_path}. "
            "Use replace_all=True to replace all, or make old_string more specific."
        )

    if ctx is not None and ctx.request_approval is not None:
        details = f"Edit: {file_path}\n--- old ---\n{old_string[:200]}\n--- new ---\n{new_string[:200]}"
        approved = ctx.request_approval("Tool: edit", details)
        if not approved:
            return f"Error: permission denied for edit({file_path})"

    new_text = text.replace(old_string, new_string)
    try:
        target.write_text(new_text, encoding="utf-8")
    except Exception as exc:
        return f"Error writing file: {exc}"

    return f"Applied edit to {file_path} ({count} occurrence(s))"


EDIT_SPEC = ToolSpec(
    name="edit",
    description="Apply an exact string replacement in a file. Useful for targeted edits without rewriting the entire file.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "Text to be replaced (must match exactly).",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text.",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    },
    categories=("filesystem",),
    permission="write",
)


SKIP_DIRS_GREP = frozenset({
    ".venv", "venv", "env", ".git", "__pycache__",
    "node_modules", ".idea", ".vscode", ".tox",
    ".eggs", ".mypy_cache", ".pytest_cache",
})


def _should_skip(dirpath: str) -> bool:
    parts = Path(dirpath).parts
    return any(p in SKIP_DIRS_GREP or p.endswith(".egg-info") for p in parts)


def grep(pattern: str, path: str = ".", include: str = "", **kwargs) -> str:
    root = Path(path)
    if not root.exists():
        return f"Error: path not found: {path}"

    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        return f"Error in regex pattern: {exc}"

    matches: list[str] = []
    include_pat = re.compile(include) if include else None
    max_results = 200

    for dirpath, _dirnames, filenames in os.walk(root):
        if _should_skip(dirpath):
            continue
        for fname in filenames:
            if include_pat and not include_pat.search(fname):
                continue
            fpath = Path(dirpath) / fname
            try:
                for i, line in enumerate(fpath.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if compiled.search(line):
                        rel = fpath.relative_to(Path.cwd()) if fpath.is_relative_to(Path.cwd()) else fpath
                        matches.append(f"{rel}:{i}: {line.rstrip()[:200]}")
                        if len(matches) >= max_results:
                            break
            except Exception:
                continue
            if len(matches) >= max_results:
                break
        if len(matches) >= max_results:
            break

    if not matches:
        return "(no matches)"
    result = "\n".join(matches)
    if len(matches) >= max_results:
        result += f"\n... (truncated at {max_results} matches)"
    return result


GREP_SPEC = ToolSpec(
    name="grep",
    description="Search file contents with a regex pattern.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for.",
            },
            "path": {
                "type": "string",
                "description": "Root directory to search (default: current directory).",
                "default": ".",
            },
            "include": {
                "type": "string",
                "description": "Optional regex to filter filenames (e.g. \\.py$).",
            },
        },
        "required": ["pattern"],
    },
    categories=("filesystem", "search"),
)


def glob(pattern: str, path: str = ".", **kwargs) -> str:
    root = Path(path)
    if not root.exists():
        return f"Error: path not found: {path}"

    if "**" in pattern:
        all_files = list(root.rglob(pattern))
    else:
        all_files = list(root.glob(pattern))

    matches = sorted(
        str(p.relative_to(root) if p.is_relative_to(root) else p)
        for p in all_files
        if p.exists() and not _should_skip(str(p))
    )
    if not matches:
        return "(no matches)"
    return "\n".join(matches[:500]) + ("\n... (truncated)" if len(matches) > 500 else "")


GLOB_SPEC = ToolSpec(
    name="glob",
    description="Find files matching a glob pattern (e.g. **/*.py, src/**/*.ts).",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match (supports ** for recursive).",
            },
            "path": {
                "type": "string",
                "description": "Root directory for the search (default: current directory).",
                "default": ".",
            },
        },
        "required": ["pattern"],
    },
    categories=("filesystem", "search"),
)
