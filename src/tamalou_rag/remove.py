"""Remove all chunks for a given source label or pdf_filename.

By default lists what would be deleted and asks for confirmation. Optionally
deletes the source PDF from exports/ as well, and triggers a server reload.

Examples:
    tamalou-remove --source "Malaysia Statistics"          # interactive
    tamalou-remove --source "Malaysia Statistics" --yes    # no prompt
    tamalou-remove --filename Malaysia_Statistics.pdf      # match by filename
    tamalou-remove --source "Old Book" --dry-run           # preview only
    tamalou-remove --source "Old Book" --keep-pdf          # don't touch exports/
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from .core import get_client, load_config


def _trigger_reload(host: str, port: int) -> bool:
    """Best-effort POST /reload so a running server picks up the deletion."""
    try:
        import urllib.request

        req = urllib.request.Request(
            f"http://{host}:{port}/reload", method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _find_matches(coll, source: str | None, filename: str | None):
    """Return (ids, metas) of entries matching either source or pdf_filename."""
    if source:
        where = {"source": source}
    elif filename:
        where = {"pdf_filename": filename}
    else:
        raise ValueError("Need --source or --filename")
    data = coll.get(where=where, include=["metadatas"])
    return data["ids"], data["metadatas"] or []


def _summarize(ids: list[str], metas: list[dict]) -> str:
    if not ids:
        return "  (no matches)"
    by_source: dict[str, int] = defaultdict(int)
    pages_seen: dict[str, set] = defaultdict(set)
    for m in metas:
        m = m or {}
        src = m.get("source", "?")
        by_source[src] += 1
        if "page" in m:
            pages_seen[src].add(m["page"])
    lines = []
    for src, count in by_source.items():
        if pages_seen[src]:
            lines.append(f"  - {src}: {count} chunks (pages {min(pages_seen[src])}-{max(pages_seen[src])})")
        else:
            lines.append(f"  - {src}: {count} chunks")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Remove documents from the index.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--source", help="Source label (e.g. 'Malaysia Statistics')")
    g.add_argument("--filename", help="PDF filename in metadata (e.g. 'Foo.pdf')")
    p.add_argument("--collection", help="Limit to this collection (default: all)")
    p.add_argument("--keep-pdf", action="store_true", help="Don't delete the source PDF from exports/")
    p.add_argument("--dry-run", action="store_true", help="Show what would be deleted, don't delete")
    p.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = p.parse_args()

    cfg = load_config()
    client = get_client()
    collections = (
        [client.get_collection(args.collection)]
        if args.collection
        else client.list_collections()
    )

    total = 0
    plan: list[tuple[str, list[str], list[dict]]] = []
    print(f"=== Searching for matches ===")
    for coll in collections:
        ids, metas = _find_matches(coll, args.source, args.filename)
        if not ids:
            continue
        print(f"\n[{coll.name}]  {len(ids)} matching chunks")
        print(_summarize(ids, metas))
        plan.append((coll.name, ids, metas))
        total += len(ids)

    if total == 0:
        print("Nothing to remove.")
        return

    pdf_to_unlink: Path | None = None
    if not args.keep_pdf:
        # Find the underlying PDF (if any)
        exports = Path(cfg["paths"]["exports"])
        for _, _, metas in plan:
            for m in metas or []:
                if m and m.get("pdf_filename"):
                    candidate = exports / m["pdf_filename"]
                    if candidate.exists():
                        pdf_to_unlink = candidate
                        break
            if pdf_to_unlink:
                break
        if pdf_to_unlink:
            print(f"\nWill also delete file: {pdf_to_unlink}")

    print(f"\nTotal: {total} chunks across {len(plan)} collection(s).")

    if args.dry_run:
        print("Dry-run, nothing changed.")
        return

    if not args.yes:
        try:
            ans = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans != "y":
            print("Aborted.")
            sys.exit(1)

    for coll_name, ids, _ in plan:
        coll = client.get_collection(coll_name)
        coll.delete(ids=ids)
        print(f"  ✓ deleted {len(ids)} from {coll_name}")

    if pdf_to_unlink:
        pdf_to_unlink.unlink()
        print(f"  ✓ removed {pdf_to_unlink}")

    server_cfg = cfg.get("server", {})
    host = server_cfg.get("host", "localhost")
    if host == "0.0.0.0":
        host = "localhost"
    port = int(server_cfg.get("port", 8702))
    if _trigger_reload(host, port):
        print(f"  ✓ server reloaded ({host}:{port})")
    else:
        print(f"  (server at {host}:{port} not reachable, restart manually if needed)")


if __name__ == "__main__":
    main()
