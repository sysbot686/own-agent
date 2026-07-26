from __future__ import annotations

from enum import Enum


class PermissionMode(str, Enum):
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    BYPASS = "bypass"
    LENIENT = "lenient"
