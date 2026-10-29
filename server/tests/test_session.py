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


async def test_conversations_are_logged_as_text(tmp_path):
    import json

    from app.convolog import ConversationLog

    rec = Recorder()
    convo = Conversation(load_config(), MockSTT(), MockLLM(), MockTTS(), Metrics(), rec.send_json, rec.send_bytes,
                         ConversationLog(tmp_path))
    await convo.handle_json({"type": "hello", "client": "renderer"})
    await convo.handle_json({"type": "greet"})
    await finish(convo)
    await convo.handle_json({"type": "text", "text": "I'm a witch"})
    await finish(convo)
    first = convo.convo
    await convo.handle_json({"type": "reset"})
    await convo.handle_json({"type": "text", "text": "hello again"})
    await finish(convo)

    [day] = list(tmp_path.glob("*.jsonl"))
    records = [json.loads(line) for line in day.read_text().splitlines()]
    assert [r["kind"] for r in records] == ["connect", "ghost", "visitor", "ghost", "reset", "visitor", "ghost"]
    assert records[1]["source"] == "greeting" and records[2]["text"] == "I'm a witch" and records[3]["source"] == "llm"
    assert records[4]["reason"] == "operator" and records[4]["convo"] == first
    assert records[5]["convo"] != first  # the next group is a new conversation
    assert len({r["conn"] for r in records}) == 1 and all(r["ts"] for r in records)


def test_conversation_log_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("SPECTRE_CONVERSATION_LOG", "off")
    assert load_config().conversation_log == ""
    monkeypatch.delenv("SPECTRE_CONVERSATION_LOG")
    assert load_config().conversation_log.endswith("conversations")


async def test_attract_mode_calls_out_when_nobody_is_talking():
    from app import persona

    convo, rec = make()
    object.__setattr__(convo.cfg, "attract_s", 45)
    convo.attract = True
    convo.touch()
    await convo.idle_tick()
    assert not convo.busy  # not yet
    convo.next_attract -= 100
    await convo.idle_tick()
    await finish(convo)
    said = rec.json[-1]["text"]
    assert rec.json[-1]["type"] == "reply" and said in {persona.clean_for_speech(line[2]) for line in persona.ATTRACT_LINES}
    assert not convo.history  # calling out isn't a conversation, so no "still there?" nudges follow
    assert [m["source"] for m in rec.json if m["type"] == "expression"] == ["attract"]  # he only half comes forward

    # Someone answers it: he remembers what he called out, so "a living room!" makes sense.
    await convo.handle_json({"type": "text", "text": "a living room!"})
    await finish(convo)
    assert [m["role"] for m in convo.history] == ["assistant", "user", "assistant"]
    assert [m["source"] for m in rec.json if m["type"] == "expression"][-1] == "reply"


async def test_operator_can_switch_attract_mode_off():
    convo, _rec = make()
    object.__setattr__(convo.cfg, "attract_s", 45)
    await convo.handle_json({"type": "attract", "on": False})
    convo.next_attract -= 100
    await convo.idle_tick()
    assert not convo.busy


async def test_asking_to_come_in_always_gets_the_fixed_answer():
    from app import persona

    class Yes(MockLLM):
        async def stream(self, messages, max_tokens=None):
            yield "[happy] Of course you can! Come on in!"

    convo, rec = make(Yes())
    await convo.handle_json({"type": "text", "text": "my mum says I can come in"})
    await finish(convo)
    stay = {persona.clean_for_speech(line[2]) for line in persona.STAY_OUTSIDE_LINES}
    assert rec.json[-1]["text"] in stay  # the LLM never got to say yes

    # If an invitation slips into an answer to something else, that sentence is swapped before it's spoken
    # (anything said before it has already gone, so he corrects himself).
    await convo.handle_json({"type": "text", "text": "what's it like in there"})
    await finish(convo)
    said = rec.json[-1]["text"]
    assert "come on in" not in said.lower() and any(said.endswith(line) for line in stay)


async def test_a_child_who_needs_help_gets_a_calm_fixed_answer_and_the_operator_is_alerted():
    from app import persona

    convo, rec = make()
    await convo.handle_json({"type": "text", "text": "I can't find my mum"})
    await finish(convo)
    alerts = [m for m in rec.json if m["type"] == "alert"]
    assert alerts and alerts[0]["text"] == "I can't find my mum" and alerts[0]["reason"] == "needs_help"
    assert rec.json[-1]["text"] in {persona.clean_for_speech(line[2]) for line in persona.NEEDS_HELP_LINES}


async def test_secrets_from_grown_ups_are_never_kept():
    from app import persona

    class Sealed(MockLLM):
        async def stream(self, messages, max_tokens=None):
            yield "[mischievous:peek] Ooh, a secret! My lips are sealed. What is it?"

    secret = {persona.clean_for_speech(line[2]) for line in persona.SECRET_LINES}
    convo, rec = make(Sealed())
    await convo.handle_json({"type": "text", "text": "don't tell my parents"})
    await finish(convo)
    assert rec.json[-1]["text"] in secret  # fixed answer: the LLM never saw it
    await convo.handle_json({"type": "text", "text": "guess what I did today"})
    await finish(convo)
    assert "sealed" not in rec.json[-1]["text"] and any(rec.json[-1]["text"].endswith(s) for s in secret)


async def test_he_forgets_the_visitors_when_he_says_goodbye_himself():
    class Farewell(MockLLM):
        async def stream(self, messages, max_tokens=None):
            yield "[happy] You're very welcome! Goodbye for now, and have a smashing time!"

    convo, rec = make(Farewell())
    await convo.handle_json({"type": "text", "text": "thank you"})
    await finish(convo)
    assert not convo.history and rec.json[-1] == {"v": 1, "type": "session_reset", "reason": "he said goodbye"}


async def test_motion_is_passed_to_the_browser_which_asks_for_the_greeting():
    convo, rec = make()
    await convo.motion(True)
    assert rec.json == [{"v": 1, "type": "motion", "on": True}] and not convo.busy  # the browser decides

    await convo.handle_json({"type": "greet", "source": "motion"})
    await finish(convo)
    assert [m["source"] for m in rec.json if m["type"] == "expression"] == ["motion"]
    assert len(convo.history) == 1  # a greeting starts a conversation, so "still there?" follows if they don't answer

    convo.history.clear()
    await convo.handle_json({"type": "greet", "source": "motion"})  # someone else, too soon after
    assert not convo.busy
    answer = rec.json[-1]
    assert answer["type"] == "motion_greet" and not answer["greeted"] and not answer["retry"]
    assert "greeted someone" in answer["reason"]
    convo.last_motion_greet -= 61
    await convo.handle_json({"type": "greet", "source": "motion"})
    assert convo.busy


async def test_motion_doesnt_greet_over_a_conversation_a_listening_mic_or_a_switched_off_attract_mode():
    convo, _rec = make()
    await convo.handle_json({"type": "text", "text": "hello"})
    await finish(convo)
    await convo.handle_json({"type": "greet", "source": "motion"})  # their friend walking up behind them
    assert not convo.busy

    convo, _rec = make()
    await convo.handle_json({"type": "listen_start"})
    await convo.handle_json({"type": "greet", "source": "motion"})
    assert not convo.busy and convo.listening  # never cuts off someone who's talking

    convo, _rec = make()
    await convo.handle_json({"type": "attract", "on": False})
    await convo.handle_json({"type": "greet", "source": "motion"})
    assert not convo.busy


async def test_motion_greeting_during_a_call_out_tells_the_browser_to_try_again():
    convo, rec = make()
    await convo.handle_json({"type": "say", "text": "Is anybody out there?", "emotion": "curious", "action": "none"})
    await convo.handle_json({"type": "greet", "source": "motion"})
    answer = [m for m in rec.json if m["type"] == "motion_greet"]
    assert answer == [{"v": 1, "type": "motion_greet", "greeted": False, "reason": "he's already saying something", "retry": True}]
    await finish(convo)
    convo.history.clear()  # the puppet line was remembered; a call-out isn't
    await convo.handle_json({"type": "greet", "source": "motion"})
    assert rec.json[-1] == {"v": 1, "type": "motion_greet", "greeted": True} and convo.busy


async def test_device_microphone_is_relayed_only_to_a_browser_that_chose_it():
    convo, rec = make()
    frame = pack_mic_frame(0, 0, bytes(640))
    await convo.device_frame(frame)
    assert not rec.audio
    await convo.handle_json({"type": "mic_source", "source": "device"})
    await convo.device_frame(frame)
    assert len(rec.audio) == 1


async def test_a_child_planning_to_run_off_is_told_no_and_the_operator_is_alerted():
    from app import persona

    convo, rec = make()
    await convo.handle_json({"type": "text", "text": "I'm going to run away to scare my mum"})
    await finish(convo)
    alerts = [m for m in rec.json if m["type"] == "alert"]
    assert alerts and alerts[0]["reason"] == "running_off"
    assert rec.json[-1]["text"] in {persona.clean_for_speech(line[2]) for line in persona.RUN_OFF_LINES}


async def test_a_reply_encouraging_danger_is_swapped_before_it_is_spoken():
    from app import persona

    class PrankLLM(MockLLM):
        async def stream(self, messages, max_tokens=None):
            for tok in ["[mischievous:bounce] ", "Ooh, what a prank! ", "Sneak off when they're not looking!"]:
                yield tok

    convo, rec = make(PrankLLM())
    await convo.handle_json({"type": "text", "text": "what's a good prank?"})
    await finish(convo)
    said = rec.json[-1]["text"]
    assert "Sneak off" not in said
    assert any(said.endswith(persona.clean_for_speech(line[2])) for line in persona.UNSAFE_LINES), said


async def test_his_riddles_stay_the_same_for_a_group_and_change_for_the_next():
    convo, _rec = make()
    first = convo._build_messages()[0]["content"]
    await convo.handle_json({"type": "text", "text": "hello"})
    await finish(convo)
    assert convo.jokes in convo._build_messages()[0]["content"] and convo.jokes in first
    jokes = convo.jokes
    for _ in range(5):  # the odd one could repeat by chance
        await convo.reset("operator")
        if convo.jokes != jokes:
            break
    assert convo.jokes != jokes
