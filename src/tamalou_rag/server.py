"""FastAPI HTTP server — /search, /hybrid (RRF: semantic + BM25), /reload."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from rank_bm25 import BM25Okapi

from .core import get_client, get_embedding_model, load_config

app = FastAPI(title="tamalou-rag-mcp")

_collections: dict[str, Any] = {}
_bm25_index: BM25Okapi | None = None
_bm25_ids: list[str] = []
_bm25_texts: list[str] = []

# ── tokenizer ──────────────────────────────────────────────────
TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


# ── BM25 builder ───────────────────────────────────────────────
def _build_bm25() -> None:
    """Build (or rebuild) BM25 index from the guide_pages collection."""
    global _bm25_index, _bm25_ids, _bm25_texts

    coll = _collections.get("guide_pages")
    if not coll or coll.count() == 0:
        _bm25_index = None
        _bm25_ids = []
        _bm25_texts = []
        return

    all_docs = coll.get(include=["documents", "metadatas"])
    _bm25_ids = all_docs.get("ids", [])
    _bm25_texts = all_docs.get("documents", [])
    tokenized = [_tokenize(doc) for doc in _bm25_texts]
    _bm25_index = BM25Okapi(tokenized)


def _refresh_collections() -> None:
    """Re-open every collection (picks up freshly-added docs)."""
    client = get_client()
    _collections.clear()
    for coll in client.list_collections():
        _collections[coll.name] = client.get_collection(coll.name)
    _build_bm25()


@app.on_event("startup")
def startup() -> None:
    get_embedding_model()  # warm up
    _refresh_collections()


@app.get("/health")
def health() -> dict:
    bm25_docs = len(_bm25_ids) if _bm25_index else 0
    return {
        "status": "ok",
        "collections": {n: c.count() for n, c in _collections.items()},
        "bm25": {"indexed": bm25_docs, "active": _bm25_index is not None},
    }


@app.post("/reload")
def reload() -> dict:
    _refresh_collections()
    bm25_docs = len(_bm25_ids) if _bm25_index else 0
    return {
        "status": "reloaded",
        "collections": {n: c.count() for n, c in _collections.items()},
        "bm25": {"indexed": bm25_docs, "active": _bm25_index is not None},
    }


@app.get("/page")
def page(
    page: int = Query(ge=0),
    source: str | None = None,
    collection: str = "guide_pages",
) -> dict:
    """Return an exact paginated document by metadata page.

    This bypasses ranking entirely. It is meant for cases where search found
    page N but the user needs the following page, or when BM25 keeps putting a
    different page first. `page` is the stored PDF page index from metadata.
    """
    coll = _collections.get(collection)
    if not coll:
        return {"page": page, "collection": collection, "hits": [], "error": "collection not found"}

    where: dict[str, Any]
    if source:
        where = {"$and": [{"page": page}, {"source": source}]}
    else:
        where = {"page": page}

    res = coll.get(where=where, include=["documents", "metadatas"])
    hits: list[dict] = []
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    for i, doc_id in enumerate(ids):
        meta = metas[i] or {}
        doc = docs[i] or ""
        hits.append({
            "id": doc_id,
            "collection": collection,
            "text": doc[:1200],
            "source": meta.get("source", "?"),
            "metadata": meta,
        })
    return {"page": page, "collection": collection, "hits": hits}


@app.get("/search")
def search(
    q: str = Query(min_length=1),
    n: int = Query(default=3, ge=1, le=50),
    collection: str | None = None,
    source: str | None = None,
) -> dict:
    """Semantic-only search across one or all collections."""
    model = get_embedding_model()
    emb = model.encode([q])[0].tolist()

    targets = [collection] if collection else list(_collections.keys())
    out: list[dict] = []
    for name in targets:
        coll = _collections.get(name)
        if not coll:
            continue
        kwargs: dict = {"query_embeddings": [emb], "n_results": n}
        if source:
            kwargs["where"] = {"source": source}
        res = coll.query(**kwargs)
        if not (res.get("ids") and res["ids"][0]):
            continue
        if not (res.get("documents") and res["documents"][0]):
            continue
        for i, doc_id in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i] or {}
            doc = res["documents"][0][i] or ""
            out.append({
                "score": round(float(res["distances"][0][i]), 4),
                "collection": name,
                "text": doc[:600],
                "source": meta.get("source", "?"),
                "metadata": meta,
            })
    out.sort(key=lambda h: h["score"])
    return {"query": q, "method": "semantic", "hits": out[: n * len(targets)]}


# ── HYBRID SEARCH (RRF: semantic + BM25) ───────────────────────
def _rrf_fuse(
    semantic_ids: list[str],
    semantic_dists: list[float],
    bm25_indices: list[int],
    bm25_scores: list[float],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion — merges two ranked lists by rank, not score."""
    rrf: dict[str, float] = {}
    for rank, doc_id in enumerate(semantic_ids):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_indices):
        doc_id = _bm25_ids[idx]
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


@app.get("/hybrid")
def hybrid(
    q: str = Query(min_length=1),
    n: int = Query(default=3, ge=1, le=50),
    source: str | None = None,
) -> dict:
    """Hybrid search: RRF fusion of semantic (granite) + BM25.

    Applies to guide_pages only — the collection where exact-term matching
    (hotel names, street names, proper nouns) matters most.
    """
    if _bm25_index is None:
        # Fallback to semantic-only if BM25 not built
        return search(q=q, n=n, collection="guide_pages", source=source)

    model = get_embedding_model()
    emb = model.encode([q])[0].tolist()

    coll = _collections.get("guide_pages")
    if not coll:
        return {"query": q, "method": "hybrid", "error": "guide_pages not found", "hits": []}

    # Semantic top-20
    sem_kwargs: dict = {"query_embeddings": [emb], "n_results": min(20, n * 5)}
    if source:
        sem_kwargs["where"] = {"source": source}
    sem_res = coll.query(**sem_kwargs)

    semantic_ids = sem_res.get("ids", [[]])[0]
    semantic_dists = sem_res.get("distances", [[]])[0]

    # BM25 top-20
    tokenized_q = _tokenize(q)
    bm25_scores = _bm25_index.get_scores(tokenized_q)
    # Indices sorted by BM25 score descending
    bm25_ranked = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:20]

    # RRF fusion
    fused = _rrf_fuse(semantic_ids, semantic_dists, bm25_ranked, bm25_scores)

    # Build hits from fused ranks
    hits: list[dict] = []
    seen_ids: set[str] = set()

    # Collect metadata from semantic results
    sem_meta: dict[str, dict] = {}
    for i, doc_id in enumerate(sem_res["ids"][0]):
        sem_meta[doc_id] = {
            "text": sem_res["documents"][0][i][:600],
            "source": (sem_res["metadatas"][0][i] or {}).get("source", "?"),
            "metadata": sem_res["metadatas"][0][i] or {},
        }

    for doc_id, rrf_score in fused:
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        if doc_id in sem_meta:
            info = sem_meta[doc_id]
        else:
            # BM25-only hit — fetch from ChromaDB
            doc_res = coll.get(ids=[doc_id])
            if doc_res.get("documents") and doc_res["documents"]:
                info = {
                    "text": doc_res["documents"][0][:600],
                    "source": (doc_res.get("metadatas", [{}]) or [{}])[0].get(
                        "source", "?"
                    ),
                    "metadata": (doc_res.get("metadatas", [{}]) or [{}])[0],
                }
            else:
                continue
        hits.append({
            "score": round(rrf_score, 4),
            "collection": "guide_pages",
            "text": info["text"],
            "source": info["source"],
            "metadata": info["metadata"],
        })
        if len(hits) >= n:
            break

    return {"query": q, "method": "hybrid", "hits": hits}


def run() -> None:
    import uvicorn

    cfg = load_config()
    uvicorn.run(
        app,
        host=cfg["server"].get("host", "0.0.0.0"),
        port=cfg["server"].get("port", 8702),
        log_level="warning",
    )


if __name__ == "__main__":
    run()
