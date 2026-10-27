"""Per-turn latency timings, logged as JSON lines and summarised as p50/p95."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field

log = logging.getLogger("spectre.metrics")

STAGES = ("stt_ms", "first_token_ms", "first_sentence_ms", "first_tts_chunk_ms", "first_audio_sent_ms", "audible_ms", "total_ms")


@dataclass
class TurnTiming:
    generation: int
    kind: str  # "voice" | "text" | "say"
    t0: float = field(default_factory=time.monotonic)  # end of visitor speech / request received
    stt_ms: float | None = None
    first_token_ms: float | None = None
    first_sentence_ms: float | None = None
    first_tts_chunk_ms: float | None = None
    first_audio_sent_ms: float | None = None
    audible_ms: float | None = None
    total_ms: float | None = None
    outcome: str = "ok"

    def mark(self, stage: str) -> None:
        if getattr(self, stage) is None:
            setattr(self, stage, round((time.monotonic() - self.t0) * 1000, 1))

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("t0")
        return d


def _pct(values: list[float], p: float) -> float:
    s = sorted(values)
    k = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[k]


class Metrics:
    def __init__(self, keep: int = 500) -> None:
        self.turns: deque[TurnTiming] = deque(maxlen=keep)

    def record(self, timing: TurnTiming) -> None:
        self.turns.append(timing)
        log.info(json.dumps({"event": "turn", **timing.as_dict()}))

    def summary(self) -> dict:
        out: dict = {"turns": len(self.turns)}
        for stage in STAGES:
            vals = [v for t in self.turns if (v := getattr(t, stage)) is not None]
            if vals:
                out[stage] = {"p50": _pct(vals, 50), "p95": _pct(vals, 95), "n": len(vals)}
        return out
