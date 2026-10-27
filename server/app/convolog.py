"""A text log of every conversation: what visitors said and what Boolian said back, one JSON object per line
in a file per day (conversations/2026-10-31.jsonl). No audio is ever kept.

Each record has `ts` (local time), `conn` (one browser connection) and `convo` (one visitor group: a new
id after every reset, goodbye or give-up), plus a `kind`:

    connect      a presenter connected
    visitor      text, via ("voice" | "typed"), rejected (why it was ignored, if it was)
    ghost        text as spoken (including [fart]-style noises), emotion, action, source, latency_ms
    cancelled    a reply was cut off (reason)
    reset        the visitor was forgotten (reason)

`scripts/show_conversations.py` prints them as readable transcripts.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


def new_id() -> str:
    return f"{datetime.now():%H%M%S}-{secrets.token_hex(2)}"


class ConversationLog:
    def __init__(self, directory: str | Path | None) -> None:
        self.dir = Path(directory) if directory else None
        self._warned = False

    def write(self, record: dict) -> None:
        """Append one record. Never raises: a full disk mustn't stop the show."""
        if self.dir is None:
            return
        now = datetime.now()
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / f"{now:%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": now.isoformat(timespec="seconds"), **record}, ensure_ascii=False) + "\n")
        except OSError as e:
            if not self._warned:
                log.warning("can't write the conversation log in %s: %s", self.dir, e)
                self._warned = True
