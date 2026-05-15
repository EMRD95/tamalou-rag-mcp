"""Embed and write Chunks to ChromaDB. Used by ingest.py and add.py."""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Iterable

from .core import get_embedding_model, get_or_create_collection, load_config
from .loaders import Chunk


def write_chunks(chunks: Iterable[Chunk], default_collection: str = "default") -> dict:
    """Embed chunks and add them to their target collections. Incremental — never deletes."""
    chunks = list(chunks)
    if not chunks:
        return {"written": 0}

    model = get_embedding_model()
    cfg = load_config()
    batch = int(cfg["embedding"].get("batch_size", 32))

    print(f"🧠 Embedding {len(chunks)} chunks...")
    t0 = time.time()
    embeddings = model.encode(
        [c.text for c in chunks],
        show_progress_bar=True,
        batch_size=batch,
    )
    print(f"   done in {time.time() - t0:.1f}s")

    by_collection: dict[str, list[tuple[Chunk, list[float]]]] = defaultdict(list)
    for chunk, emb in zip(chunks, embeddings):
        target = chunk.metadata.get("_collection") or default_collection
        by_collection[target].append((chunk, emb.tolist()))

    written: dict[str, int] = {}
    for coll_name, items in by_collection.items():
        coll = get_or_create_collection(coll_name)
        prefix = uuid.uuid4().hex[:8]
        ids, docs, embs, metas = [], [], [], []
        for idx, (chunk, emb) in enumerate(items):
            ids.append(f"{prefix}_{idx}")
            docs.append(chunk.text)
            embs.append(emb)
            meta = {"source": chunk.source, **chunk.metadata}
            meta.pop("_collection", None)
            metas.append(meta)

        for start in range(0, len(ids), 500):
            end = min(start + 500, len(ids))
            coll.add(
                ids=ids[start:end],
                documents=docs[start:end],
                embeddings=embs[start:end],
                metadatas=metas[start:end],
            )
        written[coll_name] = coll.count()
        print(f"💾 {coll_name}: {len(ids)} added, total {coll.count()}")

    # Tell a running server (if any) to reload its in-memory handles
    try:
        import urllib.request
        host = cfg["server"].get("host", "127.0.0.1")
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = cfg["server"]["port"]
        req = urllib.request.Request(f"http://{host}:{port}/reload", method="POST")
        urllib.request.urlopen(req, timeout=5).read()
        print("🔄 server reloaded")
    except Exception:
        pass  # server not running, fine

    return {"written": sum(c for c in written.values()), "collections": written}
