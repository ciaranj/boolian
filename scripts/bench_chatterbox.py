"""Chatterbox-Turbo speed check: Chatterbox's own token loop vs ours (app/chatterbox_fast.py),
each on its own and while the LLM is busy generating (the two containers share the GPU).

Run via `scripts/spectre-container.sh bench-chatterbox`.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
os.environ.setdefault("TQDM_DISABLE", "1")

import torch  # noqa: E402

from app import chatterbox_fast  # noqa: E402
from app.config import load_config  # noqa: E402
from app.llm import OpenAICompatLLM  # noqa: E402

LINES = [
    "A witch!",
    "Hello there! I'm Boolian, the friendliest ghost in the whole town.",
    "I have been drifting around these halls for a hundred and seventy three years. [chuckle] Time flies!",
]


def check_sampler() -> bool:
    """Our filtering must keep exactly the tokens HF's processors keep, with the same scores."""
    from transformers import (LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
                              TopKLogitsWarper, TopPLogitsWarper)

    procs = LogitsProcessorList([TemperatureLogitsWarper(0.8), TopKLogitsWarper(1000), TopPLogitsWarper(0.95),
                                 RepetitionPenaltyLogitsProcessor(1.2)])
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for trial in range(50):
        logits = torch.randn(6563, generator=g) * (1 + trial / 10)  # from flat to very peaked
        history = torch.randint(0, 6561, (1, 40), generator=g)
        seen = torch.zeros(6563, dtype=torch.bool)
        seen[history[0]] = True
        ref = procs(history, logits.clone().unsqueeze(0))[0]
        vals, idx = chatterbox_fast.filtered(logits, seen, 0.8, 1000, 0.95, 1.2)
        ours = torch.full_like(ref, float("-inf"))
        ours[idx] = vals
        if not torch.equal(torch.isfinite(ref), torch.isfinite(ours)):
            print(f"  trial {trial}: kept {int(torch.isfinite(ours).sum())} tokens, HF kept {int(torch.isfinite(ref).sum())}")
            return False
        finite = torch.isfinite(ref)
        worst = max(worst, float((ref[finite] - ours[finite]).abs().max()))
    print(f"  identical token sets in 50 trials, largest score difference {worst:.2e}")
    return worst < 1e-4


def save(path: str, samples, sr: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


async def timed(label: str, fn, model, busy_llm: OpenAICompatLLM | None, tag: str) -> float:
    """Synthesise every line; with busy_llm, keep the LLM generating the whole time."""
    stop = asyncio.Event()

    async def keep_llm_busy():
        prompt = [{"role": "user", "content": "Tell me a very long story about a lighthouse keeper."}]
        while not stop.is_set():
            async for _ in busy_llm.stream(prompt, max_tokens=400):
                if stop.is_set():
                    break

    busy = asyncio.create_task(keep_llm_busy()) if busy_llm else None
    if busy:
        await asyncio.sleep(0.5)  # let the LLM get going
    total_audio = total_wall = 0.0
    try:
        for i, line in enumerate(LINES, 1):
            t = time.monotonic()
            samples = await asyncio.to_thread(fn, model, line)
            wall = time.monotonic() - t
            secs = len(samples) / model.sr
            total_audio += secs
            total_wall += wall
            print(f"  {label}: {secs:4.1f}s of audio in {wall * 1000:5.0f} ms  ({line[:40]!r})")
            save(f"/tmp/boolian-bench-{tag}-{i}.wav", samples, model.sr)
    finally:
        stop.set()
        if busy:
            busy.cancel()
    rate = total_audio / total_wall
    print(f"  => {rate:.1f}x real time")
    return rate


def original(model, text):
    return model.generate(text).squeeze(0).numpy()


async def main() -> int:
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    cfg = load_config()
    print("[sampler] ours vs Hugging Face's logits processors")
    if not check_sampler():
        print("FAILED: the fast sampler doesn't match; not benchmarking it")
        return 1

    print("\n[loading]")
    model = ChatterboxTurboTTS.from_pretrained("cuda")
    if cfg.tts_voice_ref:
        model.prepare_conditionals(cfg.tts_voice_ref)
    llm = OpenAICompatLLM(cfg)
    has_llm = await llm.healthy()
    if not has_llm:
        print(f"  LLM not reachable at {cfg.llm_url}: skipping the busy-GPU runs")

    results = {}
    for warm in LINES[:2]:
        original(model, warm)
    print("\n[original loop]")
    results["original"] = await timed("original", original, model, None, "original")
    if has_llm:
        print("\n[original loop, LLM busy]")
        results["original+llm"] = await timed("original", original, model, llm, "original-busy")

    chatterbox_fast.prepare(model)  # converts the transformer to bf16; the original loop can't run after this
    for warm in LINES[:2]:
        chatterbox_fast.generate(model, warm)
    print(f"\n[fast loop] transformer dtype {model.t3.tfmr.dtype}")
    results["fast"] = await timed("fast", chatterbox_fast.generate, model, None, "fast")
    if has_llm:
        print("\n[fast loop, LLM busy]")
        results["fast+llm"] = await timed("fast", chatterbox_fast.generate, model, llm, "fast-busy")
    await llm.aclose()

    print("\nSummary (x real time, higher is better):")
    for k, v in results.items():
        print(f"  {k:14} {v:5.1f}x")
    print("\nListen for any quality difference on the Mac:\n"
          "  scp 'root@boolian-spectre.home.arpa:/tmp/boolian-bench-*.wav' . && open boolian-bench-*.wav")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
