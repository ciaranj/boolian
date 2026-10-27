from dataclasses import replace

import numpy as np

from app.config import load_config
from app.tts import ChatterboxTTS, LocalKokoroTTS, make_tts


def local_cfg():
    return replace(load_config(), tts_backend="kokoro-local")


class FakeTensor:
    """Quacks like the torch tensor Kokoro yields."""

    def __init__(self, a):
        self.a = a

    def squeeze(self, dim):
        return FakeTensor(self.a.squeeze(dim))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def fake_pipeline(text, voice, speed):
    # Mirrors KPipeline.__call__: yields (graphemes, phonemes, audio) per chunk; audio may be None.
    yield text, "ph", FakeTensor(np.array([0.0, 0.5, -0.5, 2.0], dtype=np.float32))
    yield text, "ph", None


def test_factory_selects_local_kokoro():
    assert isinstance(make_tts(local_cfg()), LocalKokoroTTS)


async def test_local_kokoro_converts_and_clips():
    tts = LocalKokoroTTS(local_cfg())
    tts.pipeline = fake_pipeline
    chunks = [c async for c in tts.synth("Boo!")]
    assert len(chunks) == 1
    samples = np.frombuffer(chunks[0], "<i2")
    assert samples.tolist() == [0, 16383, -16383, 32767]
    assert await tts.healthy()


class FakeChatterbox:
    """Mirrors ChatterboxTurboTTS.generate(): the whole clip as a (1, n) float tensor."""

    sr = 24_000

    def __init__(self, n):
        self.n = n
        self.texts = []

    def generate(self, text):
        self.texts.append(text)
        return FakeTensor(np.full((1, self.n), 0.25, dtype=np.float32))


def test_factory_selects_chatterbox():
    tts = make_tts(replace(load_config(), tts_backend="chatterbox"))
    assert isinstance(tts, ChatterboxTTS) and tts.sound_tags


async def test_chatterbox_chunks_whole_clip_and_skips_empty_text():
    tts = ChatterboxTTS(replace(load_config(), tts_backend="chatterbox", tts_fast=False))
    tts.model = FakeChatterbox(n=24_000 // 4 * 2 + 10)  # two full chunks and a bit
    chunks = [c async for c in tts.synth("Boo! [chuckle] Got you.")]
    assert [len(c) // 2 for c in chunks] == [6000, 6000, 10]
    assert np.frombuffer(chunks[0], "<i2")[0] == 8191
    assert tts.model.texts == ["Boo! [chuckle] Got you."]
    assert [c async for c in tts.synth("  ")] == []


def fake_fast(monkeypatch, pieces: int, piece_s: float = 0.5, delay: float = 0.05):
    """Stand-in for app.chatterbox_fast (torch isn't installed here): emits pieces from the GPU thread."""
    import sys
    import threading
    import time
    import types

    import app

    mod = types.ModuleType("app.chatterbox_fast")
    mod.Cancelled = type("Cancelled", (Exception,), {})
    mod.calls = {"emitted": 0, "finished": threading.Event(), "saw_stop": False}

    def generate_streaming(model, text, emit, stop):
        try:
            for _ in range(pieces):
                time.sleep(delay)
                if stop.is_set():
                    mod.calls["saw_stop"] = True
                emit(np.full(int(24_000 * piece_s), 0.1, dtype=np.float32))
                mod.calls["emitted"] += 1
        finally:
            mod.calls["finished"].set()

    mod.generate_streaming = generate_streaming
    monkeypatch.setitem(sys.modules, "app.chatterbox_fast", mod)
    monkeypatch.setattr(app, "chatterbox_fast", mod, raising=False)
    return mod


def streaming_tts():
    tts = ChatterboxTTS(replace(load_config(), tts_backend="chatterbox", tts_fast=True, tts_stream=True))
    tts.model = object()
    return tts


async def test_chatterbox_streams_pieces_in_order(monkeypatch):
    mod = fake_fast(monkeypatch, pieces=3)
    chunks = [c async for c in streaming_tts().synth("Hello there, little pumpkin.")]
    assert mod.calls["emitted"] == 3
    assert [len(c) // 2 for c in chunks] == [6000, 6000] * 3  # each 0.5 s piece as two 250 ms messages


async def test_cancelled_stream_stops_the_thread_before_releasing_the_lock(monkeypatch):
    mod = fake_fast(monkeypatch, pieces=50)
    tts = streaming_tts()
    gen = tts.synth("A long sentence.")
    await gen.__anext__()  # first piece arrived
    await gen.aclose()  # the reply was cancelled
    assert mod.calls["finished"].is_set()  # the GPU thread was waited for...
    assert not tts.lock.locked()  # ...before the lock was released
    assert mod.calls["emitted"] < 50  # and it stopped at the next piece
