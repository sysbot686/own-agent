from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BINARY_EXTENSIONS = frozenset({
    ".pyc", ".pyo", ".exe", ".dll", ".so", ".dylib", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".woff", ".woff2", ".ttf", ".eot",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".whl", ".egg",
})
SKIP_DIRS = frozenset({
    ".venv", "venv", "env", ".git", "__pycache__",
    "node_modules", ".idea", ".vscode", ".tox",
    ".eggs", "*.egg-info", ".mypy_cache", ".pytest_cache",
})


@dataclass
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class CodeChunker:
    def __init__(self, chunk_size: int = 50, overlap: int = 10, max_file_size: int = 512 * 1024) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._max_file_size = max_file_size

    def chunk_project(self, root: str, patterns: list[str]) -> list[Chunk]:
        root_path = Path(root).resolve()
        all_chunks: list[Chunk] = []
        seen = set()

        for pattern in patterns:
            for f in root_path.rglob(pattern):
                if not f.is_file():
                    continue
                resolved = str(f.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                parts = f.relative_to(root_path).parts
                if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in parts):
                    continue
                if f.suffix.lower() in BINARY_EXTENSIONS:
                    continue
                if f.stat().st_size > self._max_file_size:
                    continue
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                all_chunks.extend(self._chunk_file(text, str(f.relative_to(root_path))))

        return all_chunks

    def _chunk_file(self, text: str, rel_path: str) -> list[Chunk]:
        lines = text.splitlines()
        total = len(lines)
        chunks: list[Chunk] = []
        start = 0
        while start < total:
            end = min(start + self._chunk_size, total)
            content = "\n".join(lines[start:end])
            chunks.append(Chunk(
                file_path=rel_path,
                start_line=start + 1,
                end_line=end,
                content=content,
                metadata={"total_lines": total},
            ))
            if end >= total:
                break
            start += self._chunk_size - self._overlap
        return chunks

    def chunk_text(self, text: str, source: str = "") -> list[Chunk]:
        return self._chunk_file(text, source)
