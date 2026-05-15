"""Discord export `.md` loader — splits a channel transcript into time-bursts."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from .base import Loader, Chunk


# A reasonable default for DiscordChatExporter / Tyrrrz-style markdown exports:
#   ## **author** – 2026-04-28T01:06:00+00:00
#   message body
LINE_RE = re.compile(r"^##\s*\*\*(?P<author>[^*]+)\*\*\s*[–-]\s*(?P<ts>[\dT:+\-Z .]+)\s*$")


def _parse(path: Path) -> list[dict]:
    msgs: list[dict] = []
    cur: dict | None = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.match(line.rstrip())
            if m:
                if cur:
                    msgs.append(cur)
                ts = m.group("ts").strip()
                try:
                    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    parsed = None
                cur = {"author": m.group("author").strip(), "ts": parsed, "raw_ts": ts, "body": []}
            elif cur is not None:
                cur["body"].append(line.rstrip())
    if cur:
        msgs.append(cur)
    return msgs


class DiscordMarkdownLoader(Loader):
    name = "discord_md"
    extensions = [".md"]
    collection = "tamalou_memory"

    def chunks(self, path: Path, label: str) -> Iterator[Chunk]:
        burst_minutes = int(self.config.get("burst_minutes", 30))
        burst_size = int(self.config.get("burst_size", 25))
        overlap = int(self.config.get("overlap", 5))
        max_chars = int(self.config.get("max_chars", 2000))

        msgs = _parse(path)
        if not msgs:
            return

        i = 0
        while i < len(msgs):
            burst = [msgs[i]]
            j = i + 1
            while j < len(msgs) and len(burst) < burst_size:
                if burst[-1]["ts"] and msgs[j]["ts"]:
                    if msgs[j]["ts"] - burst[-1]["ts"] > timedelta(minutes=burst_minutes):
                        break
                burst.append(msgs[j])
                j += 1

            text_parts = [
                f"{m['author']} ({m['raw_ts']}): {' '.join(m['body']).strip()}"
                for m in burst
            ]
            yield Chunk(
                text="\n".join(text_parts)[:max_chars],
                source=label,
                metadata={
                    "type": "conversation",
                    "msg_count": len(burst),
                    "first_ts": burst[0]["raw_ts"],
                    "last_ts": burst[-1]["raw_ts"],
                },
            )
            i += max(1, len(burst) - overlap)
