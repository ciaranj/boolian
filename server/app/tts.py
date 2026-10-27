"""TTS backends. Output is always mono signed 16-bit little-endian PCM at `sample_rate`."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncIterator

import httpx
import numpy as np

from .config import Config

log = logging.getLogger(__name__)


class KokoroTTS:
    """Kokoro-FastAPI's OpenAI-compatible /audio/speech endpoint, streaming raw PCM."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.sample_rate = cfg.tts_sample_rate
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=3.0))

    async def healthy(self) -> bool:
        root = self.cfg.tts_url.removesuffix("/v1")
        try:
            r = await self.client.get(f"{root}/health", timeout=2.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def synth(self, text: str) -> AsyncIterator[bytes]:
        body = {
            "model": "kokoro",
            "input": text,
            "voice": self.cfg.tts_voice,
            "response_format": "pcm",
            "speed": self.cfg.tts_speed,
            "stream": True,
        }
        carry = b""
        async with self.client.stream("POST", f"{self.cfg.tts_url}/audio/speech", json=body) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                chunk = carry + chunk
                even = len(chunk) - (len(chunk) % 2)
                carry = chunk[even:]
                if even:
                    yield chunk[:even]

    async def aclose(self) -> None:
        await self.client.aclose()


class LocalKokoroTTS:
    """Kokoro running inside this process (pip `kokoro`), so no separate TTS service is needed."""

    sample_rate = 24_000

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.pipeline = None
        self.lock = asyncio.Lock()  # one synthesis at a time on the GPU

    async def load(self) -> None:
        from kokoro import KPipeline

        lang = self.cfg.tts_voice[0]  # voice prefix: a = American, b = British, ...
        device = None if self.cfg.tts_device == "auto" else self.cfg.tts_device
        log.info("loading Kokoro (lang=%s, voice=%s, device=%s)", lang, self.cfg.tts_voice, device or "auto")
        self.pipeline = await asyncio.to_thread(KPipeline, lang_code=lang, repo_id="hexgrad/Kokoro-82M", device=device)
        await asyncio.to_thread(self._run, "Hello.")  # warm-up: loads the voice and compiles kernels
        log.info("Kokoro ready")

    async def healthy(self) -> bool:
        return self.pipeline is not None

    def _run(self, text: str) -> list[bytes]:
        out = []
        for _graphemes, _phonemes, audio in self.pipeline(text, voice=self.cfg.tts_voice, speed=self.cfg.tts_speed):
            if audio is None:
                continue
            samples = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            out.append((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        return out

    async def synth(self, text: str) -> AsyncIterator[bytes]:
        if self.pipeline is None:
            raise RuntimeError("Kokoro still loading")
        async with self.lock:
            chunks = await asyncio.to_thread(self._run, text)
        for pcm in chunks:
            yield pcm

    async def aclose(self) -> None:
        pass


class MockTTS:
    """Ghostly babble: one warbling vowel per syllable, so the mouth has something to chew on."""

    sample_rate = 24_000

    async def healthy(self) -> bool:
        return True

    async def synth(self, text: str) -> AsyncIterator[bytes]:
        await asyncio.sleep(0.08)
        sr = self.sample_rate
        syllables = max(2, round(len(text) / 4.5))
        pieces = []
        rng = np.random.default_rng(len(text))
        for i in range(syllables):
            n = int(sr * rng.uniform(0.11, 0.2))
            t = np.arange(n) / sr
            f0 = 150 + 40 * math.sin(i * 0.9) + rng.uniform(-10, 10)
            tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3))
            env = np.sin(np.pi * np.arange(n) / n) ** 0.6
            pieces.append(0.22 * tone * env)
            if rng.random() < 0.18:
                pieces.append(np.zeros(int(sr * 0.12)))
        audio = np.concatenate(pieces + [np.zeros(int(sr * 0.1))])
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        step = sr // 5 * 2  # 200 ms chunks
        for i in range(0, len(pcm), step):
            yield pcm[i : i + step]
            await asyncio.sleep(0.01)

    async def aclose(self) -> None:
        pass


def make_tts(cfg: Config) -> KokoroTTS | LocalKokoroTTS | MockTTS:
    if cfg.tts_backend == "mock":
        return MockTTS()
    if cfg.tts_backend == "kokoro-local":
        return LocalKokoroTTS(cfg)
    return KokoroTTS(cfg)
