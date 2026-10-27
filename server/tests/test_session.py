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
        self.flags: list[int] = []

    async def send_json(self, m):
        self.json.append(m)

    async def send_bytes(self, b):
        _magic, _v, flags, _r, gen, seq, _sr = OUT_HEADER.unpack_from(b)
        self.audio.append((gen, seq))
        self.flags.append(flags)

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


class SlowVoice(MockTTS):
    """Stands in for Chatterbox: has reaction noises and wants an eager first chunk."""

    whole_sentences = True
    fillers = [b"\x01\x00" * 18_000, b"\x02\x00" * 6_000]  # 0.75 s and 0.25 s

    def __init__(self):
        self.texts = []

    async def synth(self, text):
        self.texts.append(text)
        async for pcm in super().synth(text):
            yield pcm


async def test_slow_voice_reacts_instantly_then_speaks_a_short_first_chunk():
    class Witch(MockLLM):
        async def stream(self, messages, max_tokens=None):
            for tok in ["[happy:bounce] ", "A ", "witch! ", "That is a marvellous costume. ", "Boo!"]:
                yield tok

    rec = Recorder()
    tts = SlowVoice()
    convo = Conversation(load_config(), MockSTT(), Witch(), tts, Metrics(), rec.send_json, rec.send_bytes)
    await convo.handle_json({"type": "text", "text": "I'm a witch"})
    await finish(convo)
    types = rec.types()
    # The reaction goes out before the LLM's tag even arrives, as the start of the same audio stream.
    assert types.index("speech_start") < types.index("expression") < types.index("speech_end")
    assert types.count("speech_start") == 1
    assert tts.texts[0] == "A witch!"  # under MIN_SENTENCE, but first chunks are eager for this voice
    assert [seq for _gen, seq in rec.audio] == list(range(len(rec.audio)))
    # The filler isn't part of what he "said".
    assert convo.history[-1]["content"] == "[happy:bounce] A witch! That is a marvellous costume. Boo!"


async def test_fast_voice_has_no_filler():
    convo, rec = make()
    await convo.handle_json({"type": "text", "text": "hello"})
    await finish(convo)
    assert rec.types().index("expression") < rec.types().index("speech_start")


async def test_quiet_visitor_is_nudged_twice_then_forgotten():
    convo, rec = make()
    object.__setattr__(convo.cfg, "nudge_s", 10)
    object.__setattr__(convo.cfg, "max_nudges", 2)
    await convo.handle_json({"type": "text", "text": "hello"})
    await finish(convo)
    history = list(convo.history)

    await convo.idle_tick()  # not quiet long enough yet
    assert not convo.busy
    for n in (1, 2):
        convo.last_activity -= 11
        await convo.idle_tick()
        await finish(convo)
        assert convo.nudges == n
    assert list(convo.history) == history  # the prompting isn't remembered
    assert rec.types().count("speech_start") == 3

    convo.last_activity -= 11
    await convo.idle_tick()  # still nobody: he gives up, says so, and forgets them
    await finish(convo)
    assert rec.types().count("speech_start") == 4
    assert not convo.history and convo.nudges == 0 and rec.json[-1]["reason"] == "nobody there"


async def test_idle_reset_without_nudges():
    convo, rec = make()
    object.__setattr__(convo.cfg, "nudge_s", 0)
    object.__setattr__(convo.cfg, "idle_reset_s", 30)
    await convo.handle_json({"type": "text", "text": "hello"})
    await finish(convo)
    convo.last_activity -= 31
    await convo.idle_tick()
    assert not convo.history and rec.json[-1]["reason"] == "idle"


async def test_answering_a_nudge_resets_the_count():
    convo, _rec = make()
    object.__setattr__(convo.cfg, "nudge_s", 10)
    await convo.handle_json({"type": "text", "text": "hello"})
    await finish(convo)
    convo.last_activity -= 11
    await convo.idle_tick()
    await finish(convo)
    await convo.handle_json({"type": "text", "text": "I'm a witch"})
    await finish(convo)
    assert convo.nudges == 0


async def test_goodbye_forgets_the_visitor_after_he_says_goodbye():
    convo, rec = make()
    await convo.handle_json({"type": "text", "text": "I'm a pumpkin"})
    await finish(convo)
    await convo.handle_json({"type": "text", "text": "OK bye Boolian!"})
    await finish(convo)
    assert not convo.history
    types = rec.types()
    assert types[-1] == "session_reset" and rec.json[-1]["reason"] == "goodbye"
    assert types.index("speech_end", len(types) - 6) < len(types) - 1  # he still said goodbye
    assert "cancel" not in types[-6:]  # and it wasn't cut off


async def test_rude_noise_is_one_flagged_chunk_in_the_speech():
    from app.protocol import OUT_NOISE_FLAGS

    convo, rec = make()
    await convo.handle_json({"type": "say", "text": "Watch this! [fart] Oops! [raspberry] Ha!", "emotion": "mischievous", "action": "none"})
    await finish(convo)
    flagged = [i for i, f in enumerate(rec.flags) if f]
    assert [rec.flags[i] for i in flagged] == [OUT_NOISE_FLAGS["fart"]]  # only the first noise in a reply plays
    assert 0 < flagged[0] < len(rec.flags) - 1  # with speech either side of it
    assert [s for _, s in rec.audio] == list(range(len(rec.audio)))
    assert "[fart]" in rec.json[-1]["text"]  # he remembers doing it
