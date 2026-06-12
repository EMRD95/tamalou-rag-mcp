"""MCP stdio server — exposes `search` and `add` as tools.

Search returns hits + an auto screenshot for any hit that has page metadata.
Add ingests a file path (or remote URL) into the right collection.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from .add import add_file
from .core import load_config
from .screenshot import render_pages


def _log(msg: str) -> None:
    print(f"[tamalou-mcp] {msg}", file=sys.stderr, flush=True)


def _send(rid, result=None, error=None):
    resp = {"jsonrpc": "2.0", "id": rid}
    if error:
        resp["error"] = error
    else:
        resp["result"] = result
    sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _server_base_url() -> str:
    cfg = load_config()
    host = cfg["server"].get("host", "127.0.0.1")
    if host == "0.0.0.0":
        host = "127.0.0.1"
    return f"http://{host}:{cfg['server']['port']}"


def _http_search(q: str, n: int, source: str | None = None) -> dict:
    import urllib.parse

    # Use hybrid (RRF: semantic + BM25) for guide_pages — the collection
    # where exact-term matching on proper nouns matters most.
    url = f"{_server_base_url()}/hybrid?q={urllib.parse.quote(q)}&n={n}"
    if source:
        url += f"&source={urllib.parse.quote(source)}"
    try:
        hybrid_data = json.loads(urllib.request.urlopen(url, timeout=10).read())
        # Fall back to semantic-only if hybrid returns an error
        if "error" not in hybrid_data:
            return hybrid_data
    except Exception:
        pass

    # Semantic fallback
    url2 = f"{_server_base_url()}/search?q={urllib.parse.quote(q)}&n={n}"
    if source:
        url2 += f"&source={urllib.parse.quote(source)}"
    return json.loads(urllib.request.urlopen(url2, timeout=10).read())


def _http_page(page: int, source: str | None = None, collection: str = "guide_pages") -> dict:
    import urllib.parse

    url = f"{_server_base_url()}/page?page={page}&collection={urllib.parse.quote(collection)}"
    if source:
        url += f"&source={urllib.parse.quote(source)}"
    return json.loads(urllib.request.urlopen(url, timeout=10).read())


def tool_search(params: dict) -> dict:
    query = params.get("query", "")
    n = int(params.get("n", 3))
    source = params.get("source")
    data = _http_search(query, n, source)
    hits = data.get("hits", [])

    # Find the best paginated hit (top one with a page number)
    best_page_hit = next(
        (h for h in hits if h.get("metadata", {}).get("page") is not None),
        None,
    )

    out = {"query": query, "hits": hits[:n]}
    if best_page_hit:
        page_meta = best_page_hit["metadata"]
        path = render_pages([{
            "source": best_page_hit.get("source"),
            "page": page_meta.get("page"),
            "pdf_filename": page_meta.get("pdf_filename"),
        }])
        if path:
            out["screenshot"] = f"MEDIA:{path}"
            out["screenshot_matches"] = {
                "source": best_page_hit.get("source"),
                "page": page_meta.get("page"),
                "text_excerpt": best_page_hit.get("text", "")[:300],
            }
            out["_instruction"] = (
                "The screenshot in `screenshot` shows the page described in "
                "`screenshot_matches`. If you quote a passage alongside the "
                "screenshot, ONLY quote from `screenshot_matches.text_excerpt` "
                "or the matching `hits[]` entry — never from a different hit, "
                "or the image and the quote will not match. Include the MEDIA: "
                "tag verbatim in your reply (no markdown image syntax)."
            )
    return out


def tool_page(params: dict) -> dict:
    page = int(params.get("page", 0))
    source = params.get("source")
    collection = params.get("collection", "guide_pages")
    data = _http_page(page, source, collection)
    hits = data.get("hits", [])

    out = {"page": page, "hits": hits}
    best_page_hit = next(
        (h for h in hits if h.get("metadata", {}).get("page") is not None),
        None,
    )
    if best_page_hit:
        page_meta = best_page_hit["metadata"]
        path = render_pages([{
            "source": best_page_hit.get("source"),
            "page": page_meta.get("page"),
            "pdf_filename": page_meta.get("pdf_filename"),
        }])
        if path:
            out["screenshot"] = f"MEDIA:{path}"
            out["screenshot_matches"] = {
                "source": best_page_hit.get("source"),
                "page": page_meta.get("page"),
                "text_excerpt": best_page_hit.get("text", "")[:300],
            }
            out["_instruction"] = (
                "Exact page lookup: this screenshot is rendered from metadata.page "
                "without using search ranking. Quote only screenshot_matches.text_excerpt "
                "or the matching hit."
            )
    return out


def tool_add(params: dict) -> dict:
    src = params.get("source", "")
    label = params.get("label") or None
    loader = params.get("loader")

    # Remote URL → download first
    if src.startswith(("http://", "https://")):
        cfg = load_config()
        cache = Path(cfg["paths"]["exports"]) / "_downloads"
        cache.mkdir(parents=True, exist_ok=True)
        local = cache / Path(src.split("?")[0]).name
        urllib.request.urlretrieve(src, local)
        path = local
    else:
        path = Path(src)

    return add_file(path, label, loader)


def tool_health(_params: dict) -> dict:
    """Quick check that the underlying RAG HTTP server is up + collection sizes."""
    cfg = load_config()
    host = cfg.get("server", {}).get("host", "localhost")
    if host == "0.0.0.0":
        host = "localhost"
    port = int(cfg.get("server", {}).get("port", 8702))
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3) as resp:
            data = json.loads(resp.read().decode())
            data["server"] = f"{host}:{port}"
            return data
    except Exception as e:
        return {"status": "down", "server": f"{host}:{port}", "error": str(e)}


TOOLS = [
    {
        "name": "search",
        "description": "Semantic search across all indexed sources (PDFs, Discord exports, etc.). Returns hits with text + auto screenshot for paginated sources.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "n": {"type": "integer", "default": 3},
                "source": {"type": "string", "description": "Filter by source name (e.g. 'Guide du Routard', 'Rough Guides', 'Delicious Malaysia')"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "page",
        "description": "Render and return one exact PDF page by metadata page number, bypassing search ranking. Use this when you need the next page or a known page and search keeps returning another top hit.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "description": "Stored PDF page index from metadata.page (0-based)."},
                "source": {"type": "string", "description": "Optional source filter, e.g. 'Guide du Routard'."},
                "collection": {"type": "string", "default": "guide_pages"},
            },
            "required": ["page"],
        },
    },
    {
        "name": "add",
        "description": "Incrementally index a new source file (local path or http URL). Auto-detects format. The new content becomes searchable immediately, no restart needed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "File path or http(s) URL"},
                "label": {"type": "string", "description": "Display name for the source"},
                "loader": {"type": "string", "description": "Force a specific loader (pdf, discord_md, ...). Optional."},
            },
            "required": ["source"],
        },
    },
    {
        "name": "health",
        "description": "Check the RAG server status and document counts per collection. No arguments.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def main():
    _log("started")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = req.get("method")
        rid = req.get("id")
        params = req.get("params", {})

        if method == "initialize":
            _send(rid, {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "tamalou-rag", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            })
        elif method == "tools/list":
            _send(rid, {"tools": TOOLS})
        elif method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments", {})
            try:
                if name == "search":
                    result = tool_search(args)
                elif name == "page":
                    result = tool_page(args)
                elif name == "add":
                    result = tool_add(args)
                elif name == "health":
                    result = tool_health(args)
                else:
                    _send(rid, error={"code": -32601, "message": f"Unknown: {name}"})
                    continue
                _send(rid, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]})
            except Exception as e:
                _log(f"error in {name}: {e}")
                _send(rid, {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}], "isError": True})
        elif method == "notifications/initialized":
            pass


if __name__ == "__main__":
    main()
