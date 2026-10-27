"""Speech-to-text backends plus the hallucination filter.

Whisper confidently invents text on noise and silence ("Thanks for watching!"), so every
transcript is judged before the ghost is allowed to answer it.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import re
from dataclasses import dataclass

import numpy as np

from .config import Config
from .protocol import MIC_SAMPLE_RATE

log = logging.getLogger(__name__)

MIN_DURATION_S = 0.35

_PHANTOM_EXACT = {
    "you", "thank you", "thanks", "bye", "okay", "ok", "so", "uh", "um", "hmm", "mm", "oh",
    "thank you for watching", "thanks for watching", "thank you very much", "please subscribe",
}
_PHANTOM_SUBSTR = (
    "thanks for watching", "thank you for watching", "subscribe", "subtitles", "amara.org",
    "transcribed by", "transcript by", "copyright", "www.", ".com",
)


@dataclass(frozen=True)
class Transcript:
    text: str
    duration_s: float
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    # None = usable; "low_confidence" = ask to repeat; anything else = ignore silently.
    rejected: str | None = None


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", "", text.lower()).strip()


def judge(text: str, duration_s: float, avg_logprob: float, no_speech_prob: float) -> str | None:
    norm = _normalise(text)
    if duration_s < MIN_DURATION_S:
        return "too_short"
    if not norm:
        return "empty"
    if norm in _PHANTOM_EXACT or any(p in text.lower() for p in _PHANTOM_SUBSTR):
        return "phantom"
    if no_speech_prob > 0.6 and avg_logprob < -0.7:
        return "no_speech"
    if avg_logprob < -1.1:
        return "low_confidence"
    return None


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


class WhisperSTT:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    async def load(self) -> None:
        from faster_whisper import WhisperModel

        log.info("loading faster-whisper %s on %s (%s)", self.cfg.stt_model, self.cfg.stt_device, self.cfg.stt_compute_type)
        self.model = await asyncio.to_thread(
            WhisperModel, self.cfg.stt_model, device=self.cfg.stt_device, compute_type=self.cfg.stt_compute_type
        )
        await asyncio.to_thread(self._run, np.zeros(MIC_SAMPLE_RATE, dtype=np.float32))  # warm-up
        log.info("faster-whisper ready")

    async def transcribe(self, pcm: bytes) -> Transcript:
        audio = pcm16_to_float(pcm)
        return await asyncio.to_thread(self._run, audio)

    def _run(self, audio: np.ndarray) -> Transcript:
        duration = len(audio) / MIC_SAMPLE_RATE
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.stt_language,
            beam_size=1,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        if segs:
            avg_logprob = sum(s.avg_logprob for s in segs) / len(segs)
            no_speech = max(s.no_speech_prob for s in segs)
        else:
            avg_logprob, no_speech = -10.0, 1.0
        return Transcript(text, duration, avg_logprob, no_speech, judge(text, duration, avg_logprob, no_speech))


class MockSTT:
    """Pretends to hear a rotating set of visitor phrases."""

    PHRASES = itertools.cycle([
        "Hello ghost!",
        "What's your name?",
        "Are you scary?",
        "I'm dressed as a pumpkin!",
        "How old are you?",
        "Can you do a trick?",
    ])

    ready = True

    async def load(self) -> None:
        pass

    async def transcribe(self, pcm: bytes) -> Transcript:
        duration = len(pcm) / 2 / MIC_SAMPLE_RATE
        await asyncio.sleep(0.15)
        text = next(self.PHRASES)
        return Transcript(text, duration, -0.2, 0.01, judge(text, duration, -0.2, 0.01))


def make_stt(cfg: Config) -> WhisperSTT | MockSTT:
    return MockSTT() if cfg.stt_backend == "mock" else WhisperSTT(cfg)
