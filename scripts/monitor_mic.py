#!/usr/bin/env python3
"""Listen live to the ESP32's microphone, as the server relays it, with a level meter. Use headphones, or it howls.

    python scripts/monitor_mic.py ws://SPECTRE_IP:8765/ws [--gain 4]

The meter shows, each half second: the overall level, the hum band (under 300 Hz), the speech band (300 Hz-3 kHz)
and the hiss band (3-8 kHz), so you can see what moving a wire or swapping the power supply changes.
Needs websockets, numpy and sounddevice (pip install sounddevice); Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue

import numpy as np
import sounddevice as sd
import websockets

RATE = 16000
BANDS = (("hum", 0, 300), ("speech", 300, 3000), ("hiss", 3000, 8000))


def band_db(block: np.ndarray) -> list[float]:
    spec = np.abs(np.fft.rfft(block * np.hanning(len(block)))) ** 2
    f = np.fft.rfftfreq(len(block), 1 / RATE)
    scale = (np.hanning(len(block)) ** 2).sum() * len(block) / 2
    return [10 * np.log10(spec[(f >= lo) & (f < hi)].sum() / scale + 1e-12) for _, lo, hi in BANDS]


async def monitor(url: str, gain: float) -> None:
    out: queue.Queue[np.ndarray] = queue.Queue(maxsize=50)
    pending = np.zeros(0, dtype=np.float32)

    def play(outdata, frames, _time, _status):
        nonlocal pending
        while len(pending) < frames:
            try:
                pending = np.concatenate([pending, out.get_nowait()])
            except queue.Empty:
                break
        n = min(frames, len(pending))
        outdata[:n, 0] = pending[:n]
        outdata[n:, 0] = 0
        pending = pending[n:]

    meter = np.zeros(0, dtype=np.float32)
    with sd.OutputStream(samplerate=RATE, channels=1, dtype="float32", callback=play, latency="low"):
        async with websockets.connect(url, max_size=None) as ws:
            for m in ({"type": "hello", "role": "presenter", "client": "monitor_mic"}, {"type": "attract", "on": False},
                      {"type": "mic_source", "source": "device"}):
                await ws.send(json.dumps({"v": 1, **m}))
            print(f"{'level':>8} {'hum':>7} {'speech':>7} {'hiss':>7}   (dBFS; Ctrl-C to stop)")
            while True:
                msg = await ws.recv()
                if isinstance(msg, str):
                    e = json.loads(msg)
                    if e.get("type") == "mic_device" and not e.get("connected"):
                        print("ESP32 not connected to the server")
                    continue
                if msg[:4] != b"GHST":
                    continue
                pcm = np.frombuffer(msg[16:], "<i2").astype(np.float32) / 32768
                if out.qsize() > 12:  # more than ~0.25 s behind: drop, rather than drift further back
                    continue
                out.put(np.clip(pcm * gain, -1, 1))
                meter = np.concatenate([meter, pcm])
                if len(meter) >= RATE // 2:
                    level = 20 * np.log10(np.sqrt(np.mean(meter ** 2)) + 1e-9)
                    hum, speech, hiss = band_db(meter[: RATE // 2])
                    bar = "#" * max(0, int((level + 70) / 2))
                    print(f"{level:8.1f} {hum:7.1f} {speech:7.1f} {hiss:7.1f}   {bar}", flush=True)
                    meter = np.zeros(0, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="the server's presenter socket, ws://SPECTRE_IP:8765/ws[?token=…]")
    ap.add_argument("--gain", type=float, default=4, help="playback boost, so quiet noises are audible (default 4)")
    args = ap.parse_args()
    try:
        asyncio.run(monitor(args.url, args.gain))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
