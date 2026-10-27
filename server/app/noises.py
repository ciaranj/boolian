"""Rude noises (raspberries and farts) the voice can't make.

Real recordings from assets/noises/ when there are any (CC0; see CREDITS.md), resampled and levelled to sit
with his voice. Otherwise they're synthesised from scratch: both are lip/flap buzzes: a jittery pulse train (one pulse per flap) through a couple of resonances, plus
a little breathy noise. Raspberries are higher, brighter and spittier; farts are lower, wobblier and duller.
Several variants of each are made once per sample rate and picked at random, so they don't all sound the same.
"""

from __future__ import annotations

import logging
import random
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
ASSETS = Path(__file__).resolve().parent.parent / "assets" / "noises"
KINDS = ("raspberry", "fart")
VARIANTS = 5

_cache: dict[tuple[str, int], list[np.ndarray]] = {}


def _resonator(sr: int, freq: float, decay_s: float, length_s: float) -> np.ndarray:
    t = np.arange(int(sr * length_s)) / sr
    return np.exp(-t / decay_s) * np.sin(2 * np.pi * freq * t)


def _filter(x: np.ndarray, sr: int, resonances: list[tuple[float, float, float]]) -> np.ndarray:
    """Through resonators given as (freq, decay seconds, gain), summed."""
    return sum(g * np.convolve(x, _resonator(sr, f, d, d * 7))[: len(x)] for f, d, g in resonances)


def _envelope(n: int, sr: int, attack_s: float, release_s: float) -> np.ndarray:
    env = np.ones(n)
    a, r = max(1, int(sr * attack_s)), max(1, int(sr * release_s))
    env[:a] = np.linspace(0, 1, a)
    env[-r:] *= np.linspace(1, 0, r) ** 1.5
    return env


def _buzz(sr: int, dur: float, f0, jitter: float, amp_wobble, rng: np.random.Generator, flap_noise: float) -> np.ndarray:
    """Impulse per flap at the (time-varying) rate f0(t), each with a tiny burst of noise like lips parting."""
    n = int(sr * dur)
    out = np.zeros(n)
    burst = int(sr * 0.0015)
    t = 0.0
    while t < dur:
        i = int(t * sr)
        a = amp_wobble(t) * (1 + 0.25 * rng.standard_normal())
        out[i] += a
        if flap_noise:
            j = min(n, i + burst)
            out[i:j] += a * flap_noise * rng.standard_normal(j - i) * np.linspace(1, 0, j - i)
        t += 1 / max(20.0, f0(t) * (1 + jitter * rng.standard_normal()))
    return out


def _raspberry(sr: int, rng: np.random.Generator) -> np.ndarray:
    dur = rng.uniform(0.55, 0.95)
    base = rng.uniform(85, 115)
    vib = rng.uniform(5, 8)
    f0 = lambda t: base * (1 + 0.08 * np.sin(2 * np.pi * vib * t)) * (1 - 0.25 * t / dur)
    wob = lambda t: 0.8 + 0.2 * np.sin(2 * np.pi * 11 * t)
    x = _buzz(sr, dur, f0, 0.12, wob, rng, flap_noise=0.9)
    y = _filter(x, sr, [(rng.uniform(280, 340), 0.003, 1.0), (rng.uniform(700, 850), 0.002, 0.8), (rng.uniform(1400, 1700), 0.0012, 0.5)])
    spit = rng.standard_normal(len(x)) * 0.02
    spit = np.diff(spit, prepend=0.0)  # high-passed: a little spray
    return (y + spit * np.abs(y).max()) * _envelope(len(x), sr, 0.015, 0.08)


def _fart(sr: int, rng: np.random.Generator) -> np.ndarray:
    style = rng.choice(["toot", "rumble", "squeak"], p=[0.4, 0.35, 0.25])
    if style == "toot":
        dur, a0, a1 = rng.uniform(0.3, 0.5), rng.uniform(90, 120), rng.uniform(110, 140)
    elif style == "rumble":
        dur, a0, a1 = rng.uniform(0.8, 1.3), rng.uniform(55, 75), rng.uniform(45, 60)
    else:
        dur, a0, a1 = rng.uniform(0.45, 0.8), rng.uniform(170, 220), rng.uniform(100, 130)
    drift = rng.uniform(0, 2 * np.pi)
    f0 = lambda t: (a0 + (a1 - a0) * t / dur) * (1 + 0.12 * np.sin(2 * np.pi * 3.3 * t + drift))
    flutter = rng.uniform(4, 7)
    wob = lambda t: 0.6 + 0.4 * abs(np.sin(2 * np.pi * flutter * t + drift))
    x = _buzz(sr, dur, f0, 0.22, wob, rng, flap_noise=0.35)
    # Most of the "parp" has to sit above 250 Hz, or small speakers turn it into a faint thud.
    y = _filter(x, sr, [(rng.uniform(140, 190), 0.006, 0.6), (rng.uniform(350, 450), 0.004, 1.0), (rng.uniform(800, 1000), 0.0015, 0.4)])
    y = np.convolve(y, np.ones(3) / 3)[: len(y)]  # a touch duller
    # A little "pff" of air at the start.
    pff = int(sr * 0.05)
    y[:pff] += rng.standard_normal(pff) * np.linspace(0.25, 0, pff) * np.abs(y).max()
    return y * _envelope(len(x), sr, 0.01, 0.06)


def _make(kind: str, sr: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = _raspberry(sr, rng) if kind == "raspberry" else _fart(sr, rng)
    y = y / (np.abs(y).max() + 1e-9) * 0.6  # about as loud as his voice
    return np.concatenate([y, np.zeros(int(sr * 0.08))])  # a breath before he carries on


def _resample(x: np.ndarray, src: int, dst: int, speed: float = 1.0) -> np.ndarray:
    """To `dst` Hz, played `speed` times faster (which also raises the pitch, like a tape)."""
    ratio = dst / (src * speed)
    if ratio < 1:  # low-pass first so downsampling doesn't alias
        cutoff = 0.45 * ratio
        k = np.arange(-32, 33)
        taps = np.sinc(2 * cutoff * k) * np.hanning(len(k))
        x = np.convolve(x, taps / taps.sum(), mode="same")
    t = np.arange(int(len(x) * ratio)) / ratio
    return np.interp(t, np.arange(len(x)), x)


def _level(y: np.ndarray) -> np.ndarray:
    """About as loud as his voice: -20 dBFS RMS over the loud part, peaks kept under 0.85."""
    loud = y[np.abs(y) > np.abs(y).max() * 0.05]
    rms = np.sqrt(np.mean(loud**2)) if len(loud) else 1.0
    y = y * (0.1 / (rms + 1e-9))
    return y * min(1.0, 0.85 / (np.abs(y).max() + 1e-9))


def _recordings(kind: str, sr: int) -> list[np.ndarray]:
    out = []
    for path in sorted(ASSETS.glob(f"{kind}-*.wav"), key=lambda p: int(p.stem.split("-")[1])):
        with wave.open(str(path)) as w:
            src = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, w.getnchannels()).mean(axis=1) / 32768
        y = _level(_resample(x, src, sr))
        out.append(np.concatenate([y, np.zeros(int(sr * 0.08))]))
    return out


def pcm(kind: str, sr: int) -> bytes:
    """A random variant of `kind` as s16le PCM at `sr`."""
    key = (kind, sr)
    if key not in _cache:
        _cache[key] = _recordings(kind, sr)
        if not _cache[key]:
            log.info("no %s recordings in %s; synthesising them", kind, ASSETS)
            base = 100 * (KINDS.index(kind) + 1)
            _cache[key] = [_make(kind, sr, base + i) for i in range(VARIANTS)]
    y = random.choice(_cache[key])
    return (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
