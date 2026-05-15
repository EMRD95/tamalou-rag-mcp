"""Render a specific page of a stored PDF as PNG (for visual answers)."""
from __future__ import annotations

import re
from pathlib import Path

from .core import load_config


def _resolve_pdf(source: str, pdf_filename: str | None) -> Path | None:
    cfg = load_config()
    exports = Path(cfg["paths"]["exports"])
    if pdf_filename:
        candidate = exports / pdf_filename
        if candidate.exists():
            return candidate
    # Fuzzy fallback: sanitize source name
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", source)[:50]
    for ext in (".pdf", ".PDF"):
        cand = exports / (safe + ext)
        if cand.exists():
            return cand
    # Last resort: any PDF whose name contains the safe stem
    for p in exports.glob("*.pdf"):
        if safe.lower() in p.name.lower():
            return p
    return None


def render_pages(hits: list[dict], output: str | None = None, zoom: float = 2.0) -> str | None:
    """hits = [{source, page, pdf_filename?}, ...]. Returns output path or None."""
    import fitz

    cfg = load_config()
    output = output or cfg["paths"]["screenshot"]

    pixmaps = []
    for h in hits:
        pdf_path = _resolve_pdf(h.get("source", ""), h.get("pdf_filename"))
        if not pdf_path:
            continue
        page_idx = int(h.get("page", 0))
        doc = fitz.open(str(pdf_path))
        try:
            if page_idx >= len(doc):
                continue
            mat = fitz.Matrix(zoom, zoom)
            pixmaps.append(doc[page_idx].get_pixmap(matrix=mat))
        finally:
            doc.close()

    if not pixmaps:
        return None

    if len(pixmaps) == 1:
        pixmaps[0].save(output)
        return output

    # Vertical concat
    total_h = sum(p.height for p in pixmaps)
    max_w = max(p.width for p in pixmaps)
    combined = fitz.Pixmap(fitz.csRGB, max_w, total_h)
    combined.clear_with(255)
    y = 0
    for pix in pixmaps:
        for row in range(pix.height):
            src = row * pix.stride
            dst = (y + row) * combined.stride
            n = min(pix.stride, combined.stride)
            combined.samples[dst : dst + n] = pix.samples[src : src + n]
        y += pix.height
    combined.save(output)
    return output
