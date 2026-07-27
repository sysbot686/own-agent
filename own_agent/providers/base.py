"""Abstract ChatProvider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from own_agent.providers.errors import ProviderError, ProviderErrorKind
from own_agent.providers.types import ChatRequest, ChatResponse, ChatStreamEvent


class ChatProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        ...

    @abstractmethod
    def complete(self, request: ChatRequest) -> ChatResponse:
        ...

    async def acomplete(self, request: ChatRequest) -> ChatResponse:
        import asyncio
        return await asyncio.to_thread(self.complete, request)

    async def astream(self, request: ChatRequest) -> AsyncIterator[ChatStreamEvent]:
        raise ProviderError(
            ProviderErrorKind.UNSUPPORTED,
            f"provider {self.name} does not support streaming",
        )
        yield  # pragma: no cover
