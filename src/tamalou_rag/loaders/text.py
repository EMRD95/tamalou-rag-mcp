"""Plain text loader — example of how to add a new format in <30 lines.

Drop this file in `loaders/`, add a section to config.yaml, done. The registry
auto-discovers it on next import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .base import Loader, Chunk


class TextLoader(Loader):
    name = "text"
    extensions = [".txt"]
    collection = "tamalou_memory"

    def chunks(self, path: Path, label: str) -> Iterator[Chunk]:
        chunk_chars = int(self.config.get("chunk_chars", 1500))
        overlap = int(self.config.get("overlap", 150))

        text = path.read_text(encoding="utf-8", errors="replace")
        i = 0
        idx = 0
        while i < len(text):
            yield Chunk(
                text=text[i : i + chunk_chars],
                source=label,
                metadata={"type": "text", "chunk_idx": idx},
            )
            i += chunk_chars - overlap
            idx += 1
