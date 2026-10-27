from dataclasses import replace

import numpy as np

from app.config import load_config
from app.tts import LocalKokoroTTS, make_tts


def local_cfg():
    return replace(load_config(), tts_backend="kokoro-local")


class FakeTensor:
    """Quacks like the torch tensor Kokoro yields."""

    def __init__(self, a):
        self.a = a

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
