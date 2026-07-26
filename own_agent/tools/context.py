from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from own_agent.config import AppConfig


@dataclass
class ExecutionContext:
    config: AppConfig
    cwd: str = field(default="")
    request_approval: Callable[[str, str], bool] | None = None

    def __post_init__(self):
        if not self.cwd:
            self.cwd = str(Path.cwd())
