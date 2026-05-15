"""Loader registry — auto-discovers Loader subclasses dropped in this package."""
import importlib
import pkgutil
from typing import Iterator

from .base import Loader, Chunk

_REGISTRY: dict[str, type[Loader]] = {}


def _discover():
    """Import every module in this package so subclasses register themselves."""
    pkg_path = __path__  # type: ignore[name-defined]
    for _, modname, _ in pkgutil.iter_modules(pkg_path):
        if modname == "base":
            continue
        importlib.import_module(f"{__name__}.{modname}")

    for cls in Loader.__subclasses__():
        if cls.name and cls.name not in _REGISTRY:
            _REGISTRY[cls.name] = cls


def all_loaders() -> dict[str, type[Loader]]:
    if not _REGISTRY:
        _discover()
    return dict(_REGISTRY)


def get_loader(name: str) -> type[Loader]:
    loaders = all_loaders()
    if name not in loaders:
        raise KeyError(f"Unknown loader '{name}'. Available: {list(loaders)}")
    return loaders[name]


def loader_for_path(path) -> type[Loader] | None:
    """Find the first loader whose extensions match this file."""
    from pathlib import Path
    p = Path(path)
    for cls in all_loaders().values():
        if any(p.name.lower().endswith(ext) for ext in cls.extensions):
            return cls
    return None


__all__ = ["Loader", "Chunk", "all_loaders", "get_loader", "loader_for_path"]
