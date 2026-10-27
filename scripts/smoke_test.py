#!/usr/bin/env python3
"""Smoke + latency test against a running Spectre server, no browser needed.

    python scripts/smoke_test.py ws://VM:8765/ws --turns 10
    python scripts/smoke_test.py ws://VM:8765/ws --wav hello.wav --turns 5   # exercises STT too

Prints per-turn timings and p50/p95, and exits non-zero if any turn fails.
Needs the server's dev dependencies (websockets, numpy): run it with server/.venv/bin/python.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import struct
import sys
import time
import urllib.request
import wave

import numpy as np
import websockets

QUESTIONS = [
    "Hello ghost! What's your name?",
    "Are you scary?",
    "I'm dressed as a witch tonight!",
    "How old are you?",
    "What's your favourite sweet?",
    "Can you do a trick for us?",
    "Do you have any ghost friends?",
    "Is it cold being a ghost?",
]


def load_wav_16k(path: str) -> bytes:
    with wave.open(path) as w:
        if w.getsampwidth() != 2:
            sys.exit("WAV must be 16-bit PCM")
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
        rate = w.getframerate()
    if rate != 16000:
        n = int(len(audio) * 16000 / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    return audio.astype("<i2").tobytes()


def mic_frame(seq: int, pcm: bytes) -> bytes:
    return struct.pack(">4sBBHII", b"GHST", 1, 0, len(pcm), seq, seq * 20) + pcm


async def run_turn(ws, request: list, label: str) -> dict:
    t0 = time.monotonic()
    for item in request:
        await (ws.send(item) if isinstance(item, bytes) else ws.send(json.dumps(item)))
    t0 = time.monotonic()  # measure from end of "speech", like the real thing
    result = {"label": label, "first_audio_ms": None, "audio_s": 0.0, "expression": None, "reply": "", "transcript": None}
    gen = None
    rate = 24000
    while True:
        msg = await asyncio.wait_for(ws.recv(), timeout=40)
        if isinstance(msg, bytes):
            if result["first_audio_ms"] is None:
                result["first_audio_ms"] = (time.monotonic() - t0) * 1000
            rate = struct.unpack_from(">I", msg, 16)[0]
            result["audio_s"] += (len(msg) - 20) / 2 / rate
            continue
        ev = json.loads(msg)
        kind = ev.get("type")
        if kind == "expression" and result["expression"] is None:
            result["expression"] = f"{ev['emotion']}:{ev['action']}"
        elif kind == "transcript":
            result["transcript"] = ev.get("text")
            if ev.get("rejected"):
                result["rejected"] = ev["rejected"]
        elif kind == "speech_start":
            gen = ev["generation_id"]
        elif kind == "speech_end":
            await ws.send(json.dumps({"type": "playback", "generation_id": gen, "event": "started"}))
            await ws.send(json.dumps({"type": "playback", "generation_id": gen, "event": "ended"}))
        elif kind == "reply":
            result["reply"] = ev.get("text", "")
            return result
        elif kind == "state" and ev.get("state") in ("idle", "error") and gen is None:
            result["state"] = ev["state"]
            return result


def pct(vals: list[float], p: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="ws://localhost:8765/ws")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--wav", help="16-bit WAV of someone speaking, sent as microphone audio")
    args = ap.parse_args()

    http = args.url.replace("ws://", "http://").replace("wss://", "https://").rsplit("/ws", 1)[0]
    try:
        print("ready:", urllib.request.urlopen(f"{http}/health/ready", timeout=5).read().decode())
    except Exception as e:  # 503 when a backend is down: still worth trying the turns
        print("ready check:", e)

    pcm = load_wav_16k(args.wav) if args.wav else None
    failures = 0
    latencies = []
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "role": "smoke-test"}))
        print("welcome:", json.loads(await ws.recv()))

        r = await run_turn(ws, [{"type": "say", "text": "Testing, testing. Is this thing haunted?", "emotion": "curious", "action": "tilt"}], "say")
        print(f"[say] first audio {r['first_audio_ms']:.0f} ms, {r['audio_s']:.1f}s audio" if r["first_audio_ms"] else f"[say] FAILED {r}")
        failures += r["first_audio_ms"] is None

        for i in range(args.turns):
            if pcm:
                frames = [mic_frame(n, pcm[o : o + 640]) for n, o in enumerate(range(0, len(pcm) - 639, 640))]
                req = [{"type": "listen_start"}, *frames, {"type": "listen_end"}]
                label = "voice"
            else:
                req = [{"type": "text", "text": QUESTIONS[i % len(QUESTIONS)]}]
                label = "text"
            r = await run_turn(ws, req, label)
            ok = r["first_audio_ms"] is not None
            failures += not ok
            if ok:
                latencies.append(r["first_audio_ms"])
            heard = f" heard={r['transcript']!r}" if pcm else ""
            print(f"[{label} {i + 1}]{heard} {r['expression']} first audio "
                  f"{r['first_audio_ms'] or float('nan'):.0f} ms, {r['audio_s']:.1f}s: {r['reply'] or r}")
            await asyncio.sleep(0.3)
        await ws.send(json.dumps({"type": "reset"}))

    if latencies:
        print(f"\nrequest → first audio byte: p50 {pct(latencies, 50):.0f} ms, p95 {pct(latencies, 95):.0f} ms, "
              f"mean {statistics.mean(latencies):.0f} ms over {len(latencies)} turns")
        print("(the browser adds ~120 ms jitter buffer + output latency; see the operator panel for true audible latency)")
    try:
        print("server metrics:", urllib.request.urlopen(f"{http}/metrics", timeout=5).read().decode())
    except Exception:
        pass
    print("FAILED turns:", failures) if failures else print("all turns OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
