"""One-shot migration: add pdf_filename to existing PDF entries.

The original `build_db.py` didn't store `pdf_filename` in metadata. This script
fills it in so the screenshot module can find the right PDF without relying on
a hardcoded source-name → filename map.

Run once after pointing the new repo at your existing chroma_db:
    python -m tamalou_rag.migrate
    python -m tamalou_rag.migrate --legacy-map legacy_map.json

Optional `legacy_map.json` format (label → filename in exports/):
    {
      "My Book Label": "actual_filename.pdf",
      "Another Source": "another_file.pdf"
    }
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .core import get_client, load_config


def _load_legacy_map(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text())


def _infer_filename(source: str, exports: Path, legacy_map: dict[str, str]) -> str | None:
    if source in legacy_map and (exports / legacy_map[source]).exists():
        return legacy_map[source]
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", source)[:50]
    for ext in (".pdf", ".PDF"):
        if (exports / (safe + ext)).exists():
            return safe + ext
    for p in exports.glob("*.pdf"):
        if safe.lower() in p.name.lower():
            return p.name
    return None


def main():
    p = argparse.ArgumentParser(description="Backfill pdf_filename metadata.")
    p.add_argument("--collection", default="guide_pages")
    p.add_argument("--legacy-map", help="Optional JSON file mapping source labels to PDF filenames")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_config()
    exports = Path(cfg["paths"]["exports"])
    legacy_map = _load_legacy_map(Path(args.legacy_map) if args.legacy_map else None)
    coll = get_client().get_collection(args.collection)
    data = coll.get(include=["metadatas"])

    todo: list[tuple[str, dict]] = []
    by_source: dict[str, set[str]] = {}
    for doc_id, meta in zip(data["ids"], data["metadatas"]):
        meta = meta or {}
        if meta.get("pdf_filename"):
            continue
        if meta.get("type") != "pdf_page":
            continue
        source = meta.get("source", "")
        fn = _infer_filename(source, exports, legacy_map)
        if not fn:
            by_source.setdefault(source, set()).add("MISSING")
            continue
        new_meta = {**meta, "pdf_filename": fn}
        todo.append((doc_id, new_meta))
        by_source.setdefault(source, set()).add(fn)

    print(f"Collection: {args.collection}  ({len(data['ids'])} entries)")
    for src, fns in by_source.items():
        marker = "✗" if fns == {"MISSING"} else "→"
        print(f"  {marker} {src}: {', '.join(fns)}")
    print(f"To update: {len(todo)}")

    if args.dry_run or not todo:
        return

    # ChromaDB updates one ID at a time fine; batch for speed.
    batch = 500
    for start in range(0, len(todo), batch):
        chunk = todo[start : start + batch]
        coll.update(
            ids=[t[0] for t in chunk],
            metadatas=[t[1] for t in chunk],
        )
    print(f"✅ updated {len(todo)} entries")


if __name__ == "__main__":
    main()
