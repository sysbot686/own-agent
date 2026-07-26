"""Provider error classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ProviderErrorKind(StrEnum):
    PROMPT_TOO_LONG = "prompt_too_long"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH_ERROR = "auth_error"
    CONFIG_ERROR = "config_error"
    UNSUPPORTED = "unsupported"
    SERVER_ERROR = "server_error"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ProviderError(Exception):
    kind: ProviderErrorKind
    message: str

    @property
    def retryable(self) -> bool:
        return self.kind in {
            ProviderErrorKind.TIMEOUT,
            ProviderErrorKind.RATE_LIMIT,
            ProviderErrorKind.NETWORK_ERROR,
            ProviderErrorKind.SERVER_ERROR,
        }


def classify_error(message: str, *, status_code: int | None = None) -> ProviderErrorKind:
    text = message.lower()
    if "context length" in text or "prompt too long" in text or "maximum context" in text:
        return ProviderErrorKind.PROMPT_TOO_LONG
    if status_code in {401, 403}:
        return ProviderErrorKind.AUTH_ERROR
    if status_code == 429:
        return ProviderErrorKind.RATE_LIMIT
    if status_code == 408:
        return ProviderErrorKind.TIMEOUT
    if status_code is not None and 500 <= status_code <= 599:
        return ProviderErrorKind.SERVER_ERROR
    if status_code is not None and 400 <= status_code <= 499:
        return ProviderErrorKind.API_ERROR
    if "rate limit" in text or "429" in text:
        return ProviderErrorKind.RATE_LIMIT
    if "api key" in text or "unauthorized" in text or "401" in text or "403" in text:
        return ProviderErrorKind.AUTH_ERROR
    if "timeout" in text or "timed out" in text:
        return ProviderErrorKind.TIMEOUT
    if "network" in text or "connection" in text:
        return ProviderErrorKind.NETWORK_ERROR
    if "500" in text or "502" in text or "503" in text or "504" in text or "server error" in text:
        return ProviderErrorKind.SERVER_ERROR
    if "api" in text or "server" in text:
        return ProviderErrorKind.API_ERROR
    return ProviderErrorKind.UNKNOWN


def classify_exception(exc: BaseException) -> ProviderErrorKind:
    status = None
    for name in ("status_code", "status", "http_status"):
        code = getattr(exc, name, None)
        if isinstance(code, int):
            status = code
            break
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    return classify_error(str(exc), status_code=status)
