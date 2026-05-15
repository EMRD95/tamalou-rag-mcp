"""FastAPI HTTP server — /search per collection, /reload to refresh."""
from __future__ import annotations

from fastapi import FastAPI, Query

from .core import get_client, get_embedding_model, load_config

app = FastAPI(title="tamalou-rag-mcp")

_collections: dict = {}


def _refresh_collections():
    """Re-open every collection (picks up freshly-added docs)."""
    client = get_client()
    _collections.clear()
    for coll in client.list_collections():
        _collections[coll.name] = client.get_collection(coll.name)


@app.on_event("startup")
def startup():
    get_embedding_model()  # warm up
    _refresh_collections()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "collections": {n: c.count() for n, c in _collections.items()},
    }


@app.post("/reload")
def reload():
    _refresh_collections()
    return {
        "status": "reloaded",
        "collections": {n: c.count() for n, c in _collections.items()},
    }


@app.get("/search")
def search(
    q: str = Query(min_length=1),
    n: int = Query(default=3, ge=1, le=50),
    collection: str | None = None,
    source: str | None = None,
):
    """Search a single collection by name, or all if not specified."""
    model = get_embedding_model()
    emb = model.encode([q])[0].tolist()

    targets = [collection] if collection else list(_collections.keys())
    out = []
    for name in targets:
        coll = _collections.get(name)
        if not coll:
            continue
        kwargs = {"query_embeddings": [emb], "n_results": n}
        if source:
            kwargs["where"] = {"source": source}
        res = coll.query(**kwargs)
        if not (res.get("ids") and res["ids"][0]):
            continue
        for i, doc_id in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i] or {}
            out.append({
                "score": round(float(res["distances"][0][i]), 4),
                "collection": name,
                "text": res["documents"][0][i][:600],
                "source": meta.get("source", "?"),
                "metadata": meta,
            })
    out.sort(key=lambda h: h["score"])
    return {"query": q, "hits": out[: n * len(targets)]}


def run():
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
