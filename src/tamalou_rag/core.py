"""Shared singletons: config, ChromaDB client, embedding model."""
import os
from functools import lru_cache
from pathlib import Path

import yaml


def _resolve_path(p: str, root: Path) -> Path:
    return (Path(p) if Path(p).is_absolute() else (root / p)).resolve()


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Load config.yaml. Override path via TAMALOU_CONFIG env var."""
    cfg_path = os.getenv("TAMALOU_CONFIG")
    if cfg_path:
        path = Path(cfg_path).resolve()
    else:
        # Walk up from CWD looking for config.yaml, fall back to package default
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            candidate = parent / "config.yaml"
            if candidate.exists():
                path = candidate
                break
        else:
            path = Path(__file__).parent.parent.parent / "config.yaml"

    with open(path) as f:
        cfg = yaml.safe_load(f)

    root = path.parent
    paths = cfg.setdefault("paths", {})
    for k in ("chroma_db", "data", "exports"):
        if k in paths:
            paths[k] = str(_resolve_path(paths[k], root))
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(root)
    return cfg


@lru_cache(maxsize=1)
def get_embedding_model():
    """Lazy-load the embedding model. Singleton across the process."""
    from sentence_transformers import SentenceTransformer
    cfg = load_config()
    return SentenceTransformer(cfg["embedding"]["model"])


@lru_cache(maxsize=1)
def get_client():
    import chromadb
    cfg = load_config()
    return chromadb.PersistentClient(path=cfg["paths"]["chroma_db"])


def get_or_create_collection(name: str):
    return get_client().get_or_create_collection(
        name, metadata={"hnsw:space": "cosine"}
    )
