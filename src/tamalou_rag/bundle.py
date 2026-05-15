"""Bundle export/import — ship freshly-embedded chunks across machines.

Workflow:
    # On a GPU box
    tamalou-add bigbook.pdf "Big Book" --export /tmp/bigbook.tar.gz

    # SCP /tmp/bigbook.tar.gz to the prod server, then:
    tamalou-import /tmp/bigbook.tar.gz

Bundle layout (single .tar.gz):
    manifest.json       embedding model, schema version, per-collection counts
    <collection>.jsonl  one record per line: {"id", "text", "metadata"}
    <collection>.npy    embeddings array (row-aligned with jsonl)
    pdfs/<file>.pdf     optional: source PDFs needed for screenshot rendering
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

from .core import get_client, load_config


SCHEMA_VERSION = 1


def export_filter(
    output: Path,
    collection: str | None = None,
    source: str | None = None,
    ids: list[str] | None = None,
    include_pdfs: bool = True,
) -> dict:
    """Export entries matching a filter to a tar.gz bundle."""
    cfg = load_config()
    client = get_client()
    exports_dir = Path(cfg["paths"]["exports"])

    target_collections = [collection] if collection else [c.name for c in client.list_collections()]

    manifest = {
        "schema": SCHEMA_VERSION,
        "model": cfg["embedding"]["model"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collections": {},
    }
    pdf_filenames: set[str] = set()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as tar:
        for coll_name in target_collections:
            try:
                coll = client.get_collection(coll_name)
            except Exception:
                continue

            kwargs: dict = {"include": ["documents", "metadatas", "embeddings"]}
            if ids:
                kwargs["ids"] = ids
            elif source:
                kwargs["where"] = {"source": source}
            data = coll.get(**kwargs)

            n = len(data["ids"])
            if n == 0:
                continue

            # JSONL records
            jsonl_buf = io.BytesIO()
            for i in range(n):
                meta = data["metadatas"][i] or {}
                jsonl_buf.write(
                    (json.dumps({"id": data["ids"][i], "text": data["documents"][i], "metadata": meta}, ensure_ascii=False) + "\n").encode("utf-8")
                )
                if include_pdfs and meta.get("pdf_filename"):
                    pdf_filenames.add(meta["pdf_filename"])

            jsonl_bytes = jsonl_buf.getvalue()
            info = tarfile.TarInfo(name=f"{coll_name}.jsonl")
            info.size = len(jsonl_bytes)
            tar.addfile(info, io.BytesIO(jsonl_bytes))

            # Embeddings as .npy
            npy_buf = io.BytesIO()
            np.save(npy_buf, np.asarray(data["embeddings"], dtype=np.float32))
            npy_bytes = npy_buf.getvalue()
            info = tarfile.TarInfo(name=f"{coll_name}.npy")
            info.size = len(npy_bytes)
            tar.addfile(info, io.BytesIO(npy_bytes))

            manifest["collections"][coll_name] = {"count": n}

        # Bundle the source PDFs too so the receiving side can render screenshots
        for pdf_name in sorted(pdf_filenames):
            src = exports_dir / pdf_name
            if src.exists():
                tar.add(src, arcname=f"pdfs/{pdf_name}")

        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_bytes)
        tar.addfile(info, io.BytesIO(manifest_bytes))

    return manifest


def import_bundle(bundle: Path, skip_existing: bool = True) -> dict:
    """Import a .tar.gz bundle into the local ChromaDB. Idempotent on IDs."""
    cfg = load_config()
    client = get_client()
    exports_dir = Path(cfg["paths"]["exports"])
    exports_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(bundle, "r:gz") as tar:
        names = tar.getnames()

        # Read manifest
        try:
            mf_member = tar.getmember("manifest.json")
            manifest = json.loads(tar.extractfile(mf_member).read().decode("utf-8"))
        except KeyError:
            return {"error": "manifest.json missing — not a valid bundle"}

        if manifest.get("model") != cfg["embedding"]["model"]:
            return {
                "error": f"model mismatch: bundle uses {manifest['model']}, "
                f"local uses {cfg['embedding']['model']}. Embeddings would be incompatible."
            }

        result = {"imported": {}, "skipped": {}}

        for coll_name in manifest.get("collections", {}):
            jsonl_name = f"{coll_name}.jsonl"
            npy_name = f"{coll_name}.npy"
            if jsonl_name not in names or npy_name not in names:
                continue

            jsonl_text = tar.extractfile(jsonl_name).read().decode("utf-8")
            records = [json.loads(line) for line in jsonl_text.splitlines() if line.strip()]
            embeddings = np.load(io.BytesIO(tar.extractfile(npy_name).read()))

            coll = client.get_or_create_collection(coll_name, metadata={"hnsw:space": "cosine"})

            if skip_existing and records:
                existing = set(coll.get(ids=[r["id"] for r in records])["ids"])
                kept_idx = [i for i, r in enumerate(records) if r["id"] not in existing]
                result["skipped"][coll_name] = len(records) - len(kept_idx)
                records = [records[i] for i in kept_idx]
                embeddings = embeddings[kept_idx] if kept_idx else embeddings[:0]

            if records:
                for start in range(0, len(records), 500):
                    end = min(start + 500, len(records))
                    coll.add(
                        ids=[r["id"] for r in records[start:end]],
                        documents=[r["text"] for r in records[start:end]],
                        embeddings=embeddings[start:end].tolist(),
                        metadatas=[r["metadata"] for r in records[start:end]],
                    )

            result["imported"][coll_name] = len(records)

        # Restore source PDFs
        for member in tar.getmembers():
            if member.name.startswith("pdfs/") and member.isfile():
                target = exports_dir / Path(member.name).name
                if not target.exists():
                    with tar.extractfile(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)

    # Hot-reload local server if up
    try:
        import urllib.request
        host = cfg["server"].get("host", "127.0.0.1")
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = cfg["server"]["port"]
        urllib.request.urlopen(
            urllib.request.Request(f"http://{host}:{port}/reload", method="POST"),
            timeout=5,
        )
    except Exception:
        pass

    return result


def export_main():
    p = argparse.ArgumentParser(description="Export ChromaDB entries to a portable bundle.")
    p.add_argument("output", help="Path to write .tar.gz bundle")
    p.add_argument("--collection", help="Limit to one collection")
    p.add_argument("--source", help="Limit to entries with this source label (e.g. 'Big Book')")
    p.add_argument("--ids", help="Comma-separated list of IDs to export")
    p.add_argument("--no-pdfs", action="store_true", help="Don't bundle source PDFs")
    args = p.parse_args()

    ids = [s.strip() for s in args.ids.split(",")] if args.ids else None
    manifest = export_filter(
        Path(args.output).resolve(),
        collection=args.collection,
        source=args.source,
        ids=ids,
        include_pdfs=not args.no_pdfs,
    )
    print(json.dumps(manifest, indent=2))


def import_main():
    p = argparse.ArgumentParser(description="Import a bundle into the local ChromaDB.")
    p.add_argument("bundle", help="Path to .tar.gz bundle")
    p.add_argument("--allow-overwrite", action="store_true", help="Re-add IDs that already exist")
    args = p.parse_args()

    result = import_bundle(Path(args.bundle).resolve(), skip_existing=not args.allow_overwrite)
    print(json.dumps(result, indent=2))
    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    export_main()
