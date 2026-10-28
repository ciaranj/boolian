#!/usr/bin/env python3
"""Record the ESP32's microphone, as the server relays it, to a WAV you can listen to.

    python scripts/listen_mic.py ws://SPECTRE_IP:8765/ws --seconds 10 --out esp32.wav && afplay esp32.wav

It connects like a browser would, with attract mode off so he doesn't call out to it, and prints the level once a
second. Needs the server's dev dependencies (websockets): run it with server/.venv/bin/python.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct
import sys
import wave

import websockets

MIC_HEADER = struct.Struct(">4sBBHII")


def dbfs(pcm: bytes) -> float:
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    rms = math.sqrt(sum(s * s for s in samples) / max(1, len(samples)))
    return 20 * math.log10(rms / 32768 + 1e-9)


async def record(url: str, seconds: float, out: str) -> int:
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"v": 1, "type": "hello", "role": "presenter", "client": "listen_mic"}))
        await ws.send(json.dumps({"v": 1, "type": "attract", "on": False}))  # also stops motion greetings to us
        await ws.send(json.dumps({"v": 1, "type": "mic_source", "source": "device"}))
        pcm = bytearray()
        second = bytearray()
        gaps = 0
        expected = None
        want = int(seconds * 16000 * 2)
        try:
            async with asyncio.timeout(seconds + 5):
                while len(pcm) < want:
                    msg = await ws.recv()
                    if isinstance(msg, str):
                        e = json.loads(msg)
                        if e.get("type") == "mic_device":
                            state = f"“{e.get('name')}”, Wi-Fi {e.get('rssi', '?')} dBm" if e.get("connected") else "not connected"
                            print(f"ESP32: {state}", file=sys.stderr)
                        continue
                    magic, _v, _f, _n, seq, _ms = MIC_HEADER.unpack_from(msg)
                    if magic != b"GHST":
                        continue
                    if expected is not None and seq != expected:
                        gaps += 1
                    expected = (seq + 1) & 0xFFFFFFFF
                    frame = msg[MIC_HEADER.size :]
                    pcm += frame
                    second += frame
                    if len(second) >= 32000:
                        print(f"{len(pcm) / 32000:4.0f} s  {dbfs(bytes(second)):6.1f} dBFS", file=sys.stderr)
                        second.clear()
        except TimeoutError:
            pass
    if not pcm:
        print("no audio came through: is the ESP32 connected to this server?", file=sys.stderr)
        return 1
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes(pcm))
    print(f"wrote {len(pcm) / 32000:.1f} s to {out} ({gaps} gaps), overall {dbfs(bytes(pcm)):.1f} dBFS", file=sys.stderr)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="the server's presenter socket, ws://SPECTRE_IP:8765/ws[?token=…]")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--out", default="esp32.wav")
    args = ap.parse_args()
    sys.exit(asyncio.run(record(args.url, args.seconds, args.out)))


if __name__ == "__main__":
    main()
