from __future__ import annotations

from pathlib import Path

import pytest


class FakeCollection:
    def __init__(self, records):
        self.records = records

    def get(self, where=None, include=None, limit=None):
        rows = self.records
        if where:
            rows = [r for r in rows if self._matches(r["metadata"], where)]
        if limit is not None:
            rows = rows[:limit]
        return {
            "ids": [r["id"] for r in rows],
            "documents": [r["text"] for r in rows],
            "metadatas": [r["metadata"] for r in rows],
        }

    def _matches(self, meta, where):
        if "$and" in where:
            return all(self._matches(meta, part) for part in where["$and"])
        if "$or" in where:
            return any(self._matches(meta, part) for part in where["$or"])
        for key, value in where.items():
            if isinstance(value, dict) and "$in" in value:
                if meta.get(key) not in value["$in"]:
                    return False
            elif meta.get(key) != value:
                return False
        return True


def test_pages_endpoint_returns_multiple_distinct_pages(monkeypatch):
    from tamalou_rag import server

    monkeypatch.setattr(server, "_collections", {
        "guide_pages": FakeCollection([
            {"id": "p83", "text": "text 83", "metadata": {"source": "Guide", "page": 83, "pdf_filename": "guide.pdf"}},
            {"id": "p84", "text": "text 84", "metadata": {"source": "Guide", "page": 84, "pdf_filename": "guide.pdf"}},
            {"id": "p85", "text": "text 85", "metadata": {"source": "Guide", "page": 85, "pdf_filename": "guide.pdf"}},
        ])
    })

    out = server.pages(pages="83,84,85", source="Guide")

    assert [item["page"] for item in out["items"]] == [83, 84, 85]
    assert [item["hits"][0]["text"] for item in out["items"]] == ["text 83", "text 84", "text 85"]


def test_mcp_pages_renders_unique_screenshots_per_page(monkeypatch):
    from tamalou_rag import mcp_server

    monkeypatch.setattr(mcp_server, "_http_pages", lambda **kwargs: {
        "items": [
            {"page": 83, "requested_page": "83", "hits": [{"source": "Guide", "text": "text 83", "metadata": {"page": 83, "pdf_filename": "guide.pdf"}}]},
            {"page": 84, "requested_page": "84", "hits": [{"source": "Guide", "text": "text 84", "metadata": {"page": 84, "pdf_filename": "guide.pdf"}}]},
            {"page": 85, "requested_page": "85", "hits": [{"source": "Guide", "text": "text 85", "metadata": {"page": 85, "pdf_filename": "guide.pdf"}}]},
        ]
    })

    rendered = []
    def fake_render(hits, output=None, zoom=2.0):
        rendered.append((hits[0]["page"], Path(output).name))
        return output
    monkeypatch.setattr(mcp_server, "render_pages", fake_render)

    out = mcp_server.tool_pages({"pages": [83, 84, 85], "source": "Guide"})

    assert [shot["page"] for shot in out["screenshots"]] == [83, 84, 85]
    assert len({shot["screenshot"] for shot in out["screenshots"]}) == 3
    assert rendered == [
        (83, "tamalou_page_Guide_83.png"),
        (84, "tamalou_page_Guide_84.png"),
        (85, "tamalou_page_Guide_85.png"),
    ]


def test_pdf_loader_stores_book_page_label_when_available(tmp_path, monkeypatch):
    import fitz
    from tamalou_rag import core
    from tamalou_rag.loaders.pdf import PdfLoader

    data_dir = tmp_path / "data"
    exports_dir = tmp_path / "exports"
    data_dir.mkdir()
    exports_dir.mkdir()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"""
paths:
  chroma_db: {tmp_path}/chroma_db
  data: {data_dir}
  exports: {exports_dir}
  screenshot: {tmp_path}/shot.png
embedding:
  model: ibm-granite/granite-embedding-97m-multilingual-r2
server:
  host: 127.0.0.1
  port: 18702
""")
    monkeypatch.setenv("TAMALOU_CONFIG", str(cfg))
    core.load_config.cache_clear()

    pdf_path = data_dir / "labels.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    doc.set_page_labels([{"startpage": 0, "style": "D", "firstpagenum": 83}])
    doc.save(str(pdf_path))
    doc.close()

    chunks = list(PdfLoader(config={}).chunks(pdf_path, label="Guide"))

    assert chunks[0].metadata["page_label"] == "83"
    assert chunks[1].metadata["page_label"] == "84"
