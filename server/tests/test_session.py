import asyncio
import os

os.environ["SPECTRE_MOCK"] = "1"

from app.config import load_config  # noqa: E402
from app.llm import MockLLM  # noqa: E402
from app.metrics import Metrics  # noqa: E402
from app.protocol import OUT_HEADER, pack_mic_frame  # noqa: E402
from app.session import Conversation  # noqa: E402
from app.stt import MockSTT  # noqa: E402
from app.tts import MockTTS  # noqa: E402


class Recorder:
    def __init__(self):
        self.json: list[dict] = []
        self.audio: list[tuple[int, int]] = []

    async def send_json(self, m):
        self.json.append(m)

    async def send_bytes(self, b):
        _magic, _v, _f, _r, gen, seq, _sr = OUT_HEADER.unpack_from(b)
        self.audio.append((gen, seq))

    def types(self):
        return [m["type"] for m in self.json]


def make(llm=None):
    rec = Recorder()
    convo = Conversation(load_config(), MockSTT(), llm or MockLLM(), MockTTS(), Metrics(), rec.send_json, rec.send_bytes)
    return convo, rec


async def finish(convo):
    while convo.busy:
        await asyncio.sleep(0.01)


async def test_voice_turn_produces_expression_then_audio():
    convo, rec = make()
    await convo.handle_json({"type": "listen_start"})
    for i in range(50):  # 1 s of audio
        await convo.handle_audio(pack_mic_frame(i, i * 20, bytes(640)))
    await convo.handle_json({"type": "listen_end"})
    await finish(convo)
    types = rec.types()
    assert types.index("expression") < types.index("speech_start") < types.index("speech_end")
    assert "transcript" in types and rec.audio
    assert [m["role"] for m in convo.history] == ["user", "assistant"]
    assert convo.history[1]["content"].startswith("[")


async def test_short_utterance_ignored_silently():
    convo, rec = make()
    await convo.handle_json({"type": "listen_start"})
    await convo.handle_audio(pack_mic_frame(0, 0, bytes(640)))
    await convo.handle_json({"type": "listen_end"})
    await finish(convo)
    assert "speech_start" not in rec.types()
    assert rec.json[-1]["state"] == "idle"
    assert convo.history == type(convo.history)([], maxlen=convo.history.maxlen)


async def test_cancel_stops_stale_audio():
    convo, rec = make()
    await convo.handle_json({"type": "text", "text": "tell me a long story"})
    await asyncio.sleep(0.05)
    await convo.handle_json({"type": "cancel"})
    cancelled_gen = rec.json[-1]["generation_id"]
    sent_before = len(rec.audio)
    await asyncio.sleep(0.5)
    assert len(rec.audio) == sent_before
    await convo.handle_json({"type": "say", "text": "Hello again!", "emotion": "happy", "action": "bounce"})
    await finish(convo)
    assert all(gen != cancelled_gen for gen, _ in rec.audio[sent_before:])


async def test_llm_timeout_uses_fallback_line():
    class SlowLLM(MockLLM):
        async def stream(self, messages):
            await asyncio.sleep(60)
            yield "never"

    convo, rec = make(SlowLLM())
    object.__setattr__(convo.cfg, "llm_first_sentence_timeout_s", 0.2)
    await convo.handle_json({"type": "text", "text": "hi"})
    await finish(convo)
    assert "speech_start" in rec.types()
    assert rec.json[-1]["type"] == "reply" and rec.json[-1]["text"]


async def test_messages_alternate_after_greeting():
    convo, _ = make()
    await convo.handle_json({"type": "greet"})
    await finish(convo)
    convo.history.append({"role": "user", "content": "hi"})
    roles = [m["role"] for m in convo._build_messages()]
    assert roles == ["system", "user", "assistant", "user"]
