"""Smoke tests — verify the modular pieces wire up end-to-end."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_workspace(monkeypatch):
    """Isolated workspace with its own config, chroma_db, exports."""
    root = Path(tempfile.mkdtemp(prefix="tamalou-test-"))
    (root / "data").mkdir()
    (root / "exports").mkdir()
    (root / "chroma_db").mkdir()

    cfg = root / "config.yaml"
    cfg.write_text(f"""
paths:
  chroma_db: {root}/chroma_db
  data: {root}/data
  exports: {root}/exports
  screenshot: {root}/shot.png
embedding:
  model: ibm-granite/granite-embedding-97m-multilingual-r2
  batch_size: 32
server:
  host: 127.0.0.1
  port: 18702
loaders:
  pdf: {{enabled: true, max_chars_per_page: 2000, collection: guide_pages}}
  discord_md: {{enabled: true, collection: tamalou_memory}}
  text: {{enabled: true, collection: tamalou_memory}}
""")
    monkeypatch.setenv("TAMALOU_CONFIG", str(cfg))

    # Clear singletons so the new config takes effect
    from tamalou_rag import core
    core.load_config.cache_clear()
    core.get_embedding_model.cache_clear()
    core.get_client.cache_clear()

    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_loader_registry_discovers_builtins():
    from tamalou_rag.loaders import all_loaders
    names = set(all_loaders().keys())
    assert {"pdf", "discord_md", "text"}.issubset(names)


def test_loader_for_path_matches_extension():
    from tamalou_rag.loaders import loader_for_path
    assert loader_for_path(Path("foo.pdf")).name == "pdf"
    assert loader_for_path(Path("foo.md")).name == "discord_md"
    assert loader_for_path(Path("foo.txt")).name == "text"
    assert loader_for_path(Path("foo.xyz")) is None


def test_text_loader_chunks(tmp_workspace):
    from tamalou_rag.loaders import get_loader
    f = tmp_workspace / "data" / "sample.txt"
    f.write_text("a" * 5000)
    loader = get_loader("text")(config={"chunk_chars": 1000, "overlap": 100})
    chunks = list(loader.chunks(f, label="sample"))
    assert len(chunks) >= 4
    assert all(c.source == "sample" for c in chunks)


def test_add_drop_in_loader(tmp_workspace):
    """Drop a new loader file at runtime → registry picks it up."""
    pkg = Path(__file__).parent.parent / "src" / "tamalou_rag" / "loaders"
    new_loader = pkg / "_test_csv.py"
    new_loader.write_text("""
from pathlib import Path
from typing import Iterator
from .base import Loader, Chunk

class CsvLoader(Loader):
    name = "_test_csv"
    extensions = [".csv"]
    collection = "tamalou_memory"

    def chunks(self, path: Path, label: str) -> Iterator[Chunk]:
        for i, line in enumerate(path.read_text().splitlines()):
            yield Chunk(text=line, source=label, metadata={"row": i})
""")
    try:
        # Re-discover
        from tamalou_rag import loaders as _loaders
        _loaders._REGISTRY.clear()
        from tamalou_rag.loaders import all_loaders, loader_for_path
        assert "_test_csv" in all_loaders()
        assert loader_for_path(Path("data.csv")).name == "_test_csv"
    finally:
        new_loader.unlink(missing_ok=True)


def test_bundle_roundtrip(tmp_workspace):
    """Add a tiny PDF, export as bundle, wipe DB, import, search still works."""
    import fitz
    from tamalou_rag.add import add_file
    from tamalou_rag.bundle import export_filter, import_bundle
    from tamalou_rag.core import get_client, get_embedding_model

    # Create a one-page PDF
    pdf_path = tmp_workspace / "data" / "tiny.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "The quick brown fox jumps over the lazy dog.")
    doc.save(str(pdf_path))
    doc.close()

    # Ingest
    result = add_file(pdf_path, label="Tiny", loader_name="pdf")
    assert "error" not in result, result
    assert result["written"] >= 1

    # Export
    bundle = tmp_workspace / "bundle.tar.gz"
    manifest = export_filter(bundle, source="Tiny")
    assert "guide_pages" in manifest["collections"]
    assert bundle.exists()

    # Wipe and re-import
    client = get_client()
    for c in client.list_collections():
        client.delete_collection(c.name)

    imported = import_bundle(bundle)
    assert "error" not in imported
    assert imported["imported"]["guide_pages"] == 1

    # Idempotent re-import
    imported2 = import_bundle(bundle)
    assert imported2["imported"]["guide_pages"] == 0
    assert imported2["skipped"]["guide_pages"] == 1

    # Search hits the imported chunk
    coll = client.get_collection("guide_pages")
    emb = get_embedding_model().encode(["fox jumping"])[0].tolist()
    res = coll.query(query_embeddings=[emb], n_results=1)
    assert res["ids"][0]
    assert res["metadatas"][0][0].get("pdf_filename") == "tiny.pdf"


def test_bundle_rejects_model_mismatch(tmp_workspace):
    """Import refuses if the embedding model in the bundle differs."""
    import io, json, tarfile
    from tamalou_rag.bundle import import_bundle

    bundle = tmp_workspace / "bad.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        manifest = {"schema": 1, "model": "some/other-model", "collections": {}}
        data = json.dumps(manifest).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    result = import_bundle(bundle)
    assert "error" in result
    assert "model mismatch" in result["error"].lower()
