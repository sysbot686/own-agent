from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from own_agent.context.rag.chunker import Chunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_]\w*|[^\s]", text.lower())


class Bm25Indexer:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: list[Chunk] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}
        self._tokenized: list[Counter] = []
        self._ready = False

    def index(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tokenized = []
        self._doc_lens = []

        df: Counter[str] = Counter()
        for chunk in self._chunks:
            tokens = Counter(_tokenize(chunk.content))
            self._tokenized.append(tokens)
            self._doc_lens.append(sum(tokens.values()))
            for term in tokens:
                df[term] += 1

        n = len(self._chunks)
        self._avgdl = sum(self._doc_lens) / max(n, 1)

        self._idf = {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }
        self._ready = True

    def search(self, query: str, top_k: int = 8) -> list[tuple[Chunk, float]]:
        if not self._ready or not self._chunks:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: list[float] = []
        for i, tokens in enumerate(self._tokenized):
            score = 0.0
            dl = self._doc_lens[i]
            for qt in query_tokens:
                if qt not in self._idf:
                    continue
                tf = tokens.get(qt, 0)
                idf = self._idf[qt]
                score += idf * (tf * (self._k1 + 1)) / (tf + self._k1 * (1 - self._b + self._b * dl / max(self._avgdl, 1)))
            scores.append(score)

        paired = list(zip(self._chunks, scores))
        paired.sort(key=lambda x: -x[1])
        return [(c, s) for c, s in paired[:top_k] if s > 0]

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)
