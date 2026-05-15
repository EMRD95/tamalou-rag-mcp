"""Add a single file incrementally — no rebuild."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import load_config
from .embedder import write_chunks
from .loaders import all_loaders, loader_for_path


def add_file(path: Path, label: str | None = None, loader_name: str | None = None) -> dict:
    if not path.exists():
        return {"error": f"file not found: {path}"}

    cfg = load_config()
    loaders_cfg = cfg.get("loaders", {})

    if loader_name:
        loader_cls = all_loaders().get(loader_name)
        if not loader_cls:
            return {"error": f"unknown loader '{loader_name}'. Available: {list(all_loaders())}"}
    else:
        loader_cls = loader_for_path(path)
        if not loader_cls:
            return {"error": f"no loader matches {path.name}. Pass --loader to force one."}

    loader = loader_cls(loaders_cfg.get(loader_cls.name, {}))
    target = loaders_cfg.get(loader_cls.name, {}).get("collection", loader_cls.collection or loader_cls.name)
    label = label or path.stem

    print(f"📄 {path.name} → {loader_cls.name} (collection={target}, label='{label}')")
    chunks = list(loader.chunks(path, label=label))
    for ch in chunks:
        ch.metadata["_collection"] = target

    if not chunks:
        return {"error": "loader produced no chunks"}

    return write_chunks(chunks, default_collection=target)


def main():
    p = argparse.ArgumentParser(description="Incrementally add a file to the RAG.")
    p.add_argument("path", help="File to ingest")
    p.add_argument("label", nargs="?", help="Display label (defaults to filename)")
    p.add_argument("--loader", help="Force a specific loader (default: auto-detect)")
    args = p.parse_args()

    result = add_file(Path(args.path).resolve(), args.label, args.loader)
    print(result)
    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
