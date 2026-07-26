from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from own_agent.config import AppConfig


@dataclass
class ExecutionContext:
    config: AppConfig | None = None
    cwd: str = field(default="")
    workspace_root: str = field(default="")
    request_approval: Callable[[str, str], bool] | None = None

    def __post_init__(self):
        if not self.cwd:
            self.cwd = str(Path.cwd())
        if not self.workspace_root:
            self.workspace_root = self.cwd

    def resolve_path(self, path: str) -> Path | None:
        p = Path(path)
        if not p.is_absolute():
            p = Path(self.cwd) / p
        p = p.resolve()
        root = Path(self.workspace_root).resolve()
        try:
            p.relative_to(root)
            return p
        except ValueError:
            return None

    def abs_path(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = Path(self.cwd) / p
        return p.resolve()
