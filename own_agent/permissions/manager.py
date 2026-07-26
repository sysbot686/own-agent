from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from own_agent.permissions.types import PermissionMode

logger = logging.getLogger("own-agent.permissions")


@dataclass
class ApprovalResult:
    approved: bool
    reason: str = ""


class PermissionManager:
    def __init__(
        self,
        mode: PermissionMode = PermissionMode.STANDARD,
        on_request: Callable[[str, str, PermissionMode], bool] | None = None,
    ) -> None:
        self.mode = mode
        self._on_request = on_request

    @property
    def needs_approval(self) -> bool:
        return self.mode in (PermissionMode.STANDARD, PermissionMode.AGGRESSIVE)

    def request(self, action: str, details: str) -> ApprovalResult:
        if self.mode == PermissionMode.BYPASS:
            logger.info("BYPASS: %s — %s", action, details[:200])
            return ApprovalResult(approved=True, reason="bypass mode")

        sensitive = any(k in details.lower() for k in ("write", "edit", "shell", "python_exec"))

        if self.mode == PermissionMode.LENIENT:
            if not sensitive:
                return ApprovalResult(approved=True, reason="lenient: non-sensitive")
            if self._on_request:
                result = self._on_request(action, details, self.mode)
                return ApprovalResult(approved=result)

        if self.mode in (PermissionMode.STANDARD, PermissionMode.AGGRESSIVE):
            if self._on_request:
                result = self._on_request(action, details, self.mode)
                return ApprovalResult(approved=result)
            return ApprovalResult(approved=False, reason="no approval callback configured")

        return ApprovalResult(approved=False, reason=f"unknown mode: {self.mode}")

    def make_callback(self) -> Callable[[str, str], bool]:
        def _cb(action: str, details: str) -> bool:
            return self.request(action, details).approved
        return _cb
