from __future__ import annotations

from own_agent.context.rag.chunker import Chunk
from own_agent.context.rag.indexer import Bm25Indexer


class Retriever:
    def __init__(self, indexer: Bm25Indexer) -> None:
        self._indexer = indexer

    def retrieve(self, query: str, top_k: int = 8) -> list[Chunk]:
        results = self._indexer.search(query, top_k=top_k)
        return [c for c, _ in results]

    @staticmethod
    def format_context(chunks: list[Chunk]) -> str:
        if not chunks:
            return ""

        parts: list[str] = ["<project_context>"]
        seen = set()
        for c in chunks:
            key = (c.file_path, c.start_line)
            if key in seen:
                continue
            seen.add(key)
            parts.append(
                f"{c.file_path} (lines {c.start_line}-{c.end_line}):\n"
                f"```\n{c.content}\n```"
            )
        parts.append("</project_context>")
        return "\n\n".join(parts)
