"""PDF loader — one chunk per page, screenshots come from exports/."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator

from .base import Loader, Chunk


class PdfLoader(Loader):
    name = "pdf"
    extensions = [".pdf"]
    collection = "guide_pages"

    def chunks(self, path: Path, label: str) -> Iterator[Chunk]:
        import fitz  # PyMuPDF

        max_chars = int(self.config.get("max_chars_per_page", 2000))
        # Keep a copy of the source PDF in exports/ for later screenshot rendering
        from ..core import load_config
        exports = Path(load_config()["paths"]["exports"])
        exports.mkdir(parents=True, exist_ok=True)
        dest = exports / path.name
        if path.resolve() != dest.resolve():
            shutil.copy2(path, dest)

        doc = fitz.open(str(path))
        total = len(doc)
        try:
            for i in range(total):
                text = doc[i].get_text() or "[no extractable text]"
                try:
                    page_label = doc[i].get_label() or str(i)
                except Exception:
                    page_label = str(i)
                yield Chunk(
                    text=text[:max_chars],
                    source=label,
                    metadata={
                        "type": "pdf_page",
                        "page": i,
                        "page_label": page_label,
                        "total_pages": total,
                        "pdf_filename": path.name,
                    },
                )
        finally:
            doc.close()
