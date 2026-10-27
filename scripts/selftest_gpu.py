"""GPU self-test for the Spectre container: Kokoro speaks a line, Whisper transcribes it back, the LLM answers.

Run via `scripts/spectre-container.sh selftest` (which sets up the environment and library paths).
`selftest --compare` instead times each voice model on the same lines and saves WAVs to listen to.
"""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import os
import subprocess
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))

from app.config import load_config  # noqa: E402
from app.llm import OpenAICompatLLM  # noqa: E402
from app.persona import strip_sounds  # noqa: E402
from app.stt import WhisperSTT  # noqa: E402
from app.tts import make_tts  # noqa: E402

LINE = "Hello there! I'm Boolian, the friendliest ghost in the whole town."
COMPARE_LINES = [
    LINE,
    "Ooh, a witch! Shall I tell you a secret? [whispering] I'm scared of the dark. [chuckle]",
]


def vram() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return "?"


async def main() -> int:
    cfg = load_config()
    failures = []
    print(f"VRAM before: {vram()}")

    tts = make_tts(cfg)
    print(f"\n[TTS] backend={cfg.tts_backend} voice={cfg.tts_voice}")
    if hasattr(tts, "load"):
        t = time.monotonic()
        await tts.load()
        print(f"  loaded in {time.monotonic() - t:.1f}s")
    pcm = b""
    for attempt in range(2):  # first call includes warm-up costs
        t = time.monotonic()
        pcm = b"".join([c async for c in tts.synth(LINE)])
        secs = len(pcm) / 2 / tts.sample_rate
        print(f"  run {attempt + 1}: {secs:.1f}s of audio in {(time.monotonic() - t) * 1000:.0f} ms")
    if not pcm:
        failures.append("TTS produced no audio")

    print(f"\n[STT] model={cfg.stt_model} device={cfg.stt_device}")
    stt = WhisperSTT(cfg)
    t = time.monotonic()
    await stt.load()
    print(f"  loaded in {time.monotonic() - t:.1f}s")
    if pcm:
        a = np.frombuffer(pcm, "<i2").astype(np.float32)
        n = int(len(a) * 16000 / tts.sample_rate)
        pcm16 = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype("<i2").tobytes()
        t = time.monotonic()
        tr = await stt.transcribe(pcm16)
        print(f"  heard {tr.text!r} in {(time.monotonic() - t) * 1000:.0f} ms (rejected={tr.rejected})")
        if "boolian" not in tr.text.lower() and "ghost" not in tr.text.lower():
            failures.append("Whisper didn't understand Kokoro's speech")

    print(f"\n[LLM] {cfg.llm_url}")
    llm = OpenAICompatLLM(cfg)
    if not await llm.healthy():
        failures.append(f"LLM not reachable at {cfg.llm_url}")
        print("  not reachable (is llama-server running with --host 0.0.0.0?)")
    else:
        t = time.monotonic()
        first, text = None, ""
        async for tok in llm.stream([{"role": "user", "content": "In one short sentence, say hello like a friendly ghost."}]):
            first = first or time.monotonic()
            text += tok
        print(f"  first token {(first - t) * 1000:.0f} ms, total {(time.monotonic() - t) * 1000:.0f} ms: {text.strip()!r}")
    await llm.aclose()

    print(f"\nVRAM after: {vram()}")
    if failures:
        print("\nFAILED:\n  - " + "\n  - ".join(failures))
        return 1
    print("\nAll good. Start the server with: scripts/spectre-container.sh run")
    return 0


async def compare() -> int:
    cfg = load_config()
    saved = []
    for backend in ("kokoro-local", "chatterbox"):
        print(f"\n[{backend}]")
        tts = make_tts(dataclasses.replace(cfg, tts_backend=backend))
        try:
            t = time.monotonic()
            await tts.load()
            print(f"  loaded in {time.monotonic() - t:.1f}s, VRAM now {vram()}")
        except Exception as e:  # e.g. chatterbox not installed
            print(f"  couldn't load: {e!r}")
            continue
        for i, line in enumerate(COMPARE_LINES, 1):
            text = line if tts.sound_tags else strip_sounds(line)
            for attempt in (1, 2):  # the first call includes warm-up costs
                t = time.monotonic()
                first, pcm = None, b""
                async for chunk in tts.synth(text):
                    first = first or time.monotonic()
                    pcm += chunk
                secs = len(pcm) / 2 / tts.sample_rate
                total = time.monotonic() - t
                print(f"  line {i} run {attempt}: first audio {(first - t) * 1000:.0f} ms, "
                      f"{secs:.1f}s of audio in {total * 1000:.0f} ms ({secs / total:.1f}x real time)")
            path = f"/tmp/boolian-{backend}-{i}.wav"
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(tts.sample_rate)
                w.writeframes(pcm)
            saved.append(path)
        del tts
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    print("\n\"first audio\" is what delays his reply: Kokoro streams, Chatterbox generates each sentence whole.")
    print("Listen on the Mac with:\n  scp 'root@boolian-spectre.home.arpa:/tmp/boolian-*.wav' . && open boolian-*.wav")
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(compare() if "--compare" in sys.argv[1:] else main()))
