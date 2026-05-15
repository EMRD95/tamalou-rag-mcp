"""Loader interface — drop a subclass in this package and you have a new format."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Chunk:
    text: str
    source: str
    metadata: dict = field(default_factory=dict)
    # `metadata` ends up in ChromaDB as the entry's metadata.
    # Reserved keys this project uses:
    #   page          → 0-indexed page in a paginated source (PDF)
    #   pdf_filename  → file under exports/ for screenshot extraction
    #   first_ts/last_ts → timestamps for time-windowed chunks


class Loader(ABC):
    """Subclass once per format. The class name matters less than `name` + `extensions`."""

    name: str = ""              # short identifier used in config.yaml under loaders.<name>
    extensions: list[str] = []  # file suffixes this loader handles, e.g. [".pdf"]
    collection: str = ""        # ChromaDB collection to write to (overrides config)

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abstractmethod
    def chunks(self, path: Path, label: str) -> Iterator[Chunk]:
        """Yield Chunks for a single source file."""
        raise NotImplementedError

    def target_collection(self) -> str:
        return self.collection or self.config.get("collection") or self.name
