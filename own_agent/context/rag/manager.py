from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from own_agent.context.rag.chunker import CodeChunker
from own_agent.context.rag.indexer import Bm25Indexer
from own_agent.context.rag.retriever import Retriever


@dataclass
class RagConfig:
    enabled: bool = True
    max_chunks: int = 8
    chunk_size: int = 50
    chunk_overlap: int = 10
    index_patterns: tuple[str, ...] = ("**/*.py", "**/*.pyi", "**/*.rs", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx", "**/*.md", "**/*.toml", "**/*.json", "**/*.yaml", "**/*.yml")
    embedder: str = "bm25"


class RagManager:
    def __init__(self, config: RagConfig | None = None, project_root: str | None = None) -> None:
        self._config = config or RagConfig()
        self._project_root = project_root or str(Path.cwd())
        self._chunker = CodeChunker(
            chunk_size=self._config.chunk_size,
            overlap=self._config.chunk_overlap,
        )
        self._indexer = Bm25Indexer()
        self._retriever = Retriever(self._indexer)
        self._indexed = False

    @property
    def config(self) -> RagConfig:
        return self._config

    async def index_project(self) -> None:
        patterns = list(self._config.index_patterns)
        chunks = self._chunker.chunk_project(self._project_root, patterns)
        self._indexer.index(chunks)
        self._indexed = True

    async def retrieve_context(self, query: str) -> str:
        if not self._config.enabled or not self._indexed:
            return ""
        chunks = self._retriever.retrieve(query, top_k=self._config.max_chunks)
        return self._retriever.format_context(chunks)

    @property
    def is_indexed(self) -> bool:
        return self._indexed

    @property
    def total_chunks(self) -> int:
        return self._indexer.total_chunks
