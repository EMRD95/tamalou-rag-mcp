"""Full rebuild: scan data/ and embed everything from scratch."""
from __future__ import annotations

import sys
from pathlib import Path

from .core import get_client, load_config
from .embedder import write_chunks
from .loaders import all_loaders, loader_for_path


def rebuild():
    cfg = load_config()
    data_dir = Path(cfg["paths"]["data"])
    if not data_dir.exists():
        print(f"❌ data dir not found: {data_dir}")
        sys.exit(1)

    # Drop existing collections — full rebuild
    client = get_client()
    for coll in client.list_collections():
        print(f"🗑️  drop {coll.name}")
        client.delete_collection(coll.name)

    loaders_cfg = cfg.get("loaders", {})
    loaders = {n: cls for n, cls in all_loaders().items() if loaders_cfg.get(n, {}).get("enabled", True)}

    all_chunks = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        loader_cls = loader_for_path(path)
        if not loader_cls or loader_cls.name not in loaders:
            continue
        loader = loader_cls(loaders_cfg.get(loader_cls.name, {}))
        target = loaders_cfg.get(loader_cls.name, {}).get("collection", loader_cls.collection or loader_cls.name)
        label = path.stem
        print(f"📄 {path.name} → {loader_cls.name} (collection={target})")
        for ch in loader.chunks(path, label=label):
            ch.metadata["_collection"] = target
            all_chunks.append(ch)

    if not all_chunks:
        print("Nothing to embed.")
        return

    write_chunks(all_chunks, default_collection="default")


def main():
    rebuild()


if __name__ == "__main__":
    main()
