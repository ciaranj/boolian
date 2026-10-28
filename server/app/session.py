"""One conversation with one presenter (browser). Owns turn-taking, generations and history.

Every response gets a new generation id. Anything sent for an older generation is dropped,
so a cancelled reply can never leak audio into the next one.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime

from . import noises, persona
from .convolog import ConversationLog, new_id
from .config import Config
from .metrics import Metrics, TurnTiming
from .persona import Expression, ReplyParser, Sentence
from .protocol import MIC_SAMPLE_RATE, FrameError, pack_audio_chunk, parse_mic_frame

log = logging.getLogger(__name__)

SendJson = Callable[[dict], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]

MAX_SENTENCES = 4


class Conversation:
    def __init__(self, cfg: Config, stt, llm, tts, metrics: Metrics, send_json: SendJson, send_bytes: SendBytes,
                 convo_log: ConversationLog | None = None) -> None:
        self.cfg = cfg
        self.convo_log = convo_log or ConversationLog(None)
        self.conn = new_id()  # this browser connection
        self.convo = new_id()  # this visitor group; renewed whenever he forgets them
        self.stt, self.llm, self.tts = stt, llm, tts
        self.metrics = metrics
        self._send_json, self._send_bytes = send_json, send_bytes

        self.history: deque[dict] = deque(maxlen=cfg.history_turns * 2)
        self.generation = 0
        self.task: asyncio.Task | None = None
        self.timings: dict[int, TurnTiming] = {}
        self._last_filler = -1
        self.nudges = 0  # prompts said since the visitor last spoke
        self.attract = cfg.attract_s > 0  # the operator can switch attract mode off
        self.next_attract = 0.0
        self.last_attract: tuple[float, str] | None = None  # when, and what he called out

        self.listening = False
        self.audio = bytearray()
        self.expected_seq: int | None = None
        self.seq_gaps = 0
        self.bad_frames = 0
        self.last_activity = time.monotonic()

    # ---- inbound ---------------------------------------------------------------------------

    async def handle_json(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "hello":
            self._log("connect", client=str(msg.get("client", "")))
            await self._send_json({
                "v": 1, "type": "welcome", "ghost_name": self.cfg.ghost_name,
                "backends": {"stt": self.cfg.stt_backend, "llm": self.cfg.llm_backend, "tts": self.cfg.tts_backend},
                "input_format": {"codec": "pcm_s16le", "sample_rate": MIC_SAMPLE_RATE, "channels": 1, "frame_ms": 20},
            })
        elif kind == "listen_start":
            if self.busy:
                await self.cancel("barge_in")
            self.listening = True
            self.audio.clear()
            self.touch()
        elif kind == "listen_end":
            if not self.listening:
                return
            self.listening = False
            pcm = bytes(self.audio)
            self.audio.clear()
            self._start("voice", lambda gen, t: self._voice_turn(gen, t, pcm))
        elif kind == "listen_abort":
            self.listening = False
            self.audio.clear()
        elif kind == "text":
            text = str(msg.get("text", "")).strip()[:300]
            if text:
                if self.busy:
                    await self.cancel("new_input")
                self._start("text", lambda gen, t: self._text_turn(gen, t, text))
        elif kind == "say":
            text = str(msg.get("text", "")).strip()[:500]
            if text:
                if self.busy:
                    await self.cancel("puppet")
                emotion, action = str(msg.get("emotion", "neutral")), str(msg.get("action", "none"))
                self._start("say", lambda gen, t: self._say_turn(gen, t, emotion, action, text, source="puppet"))
        elif kind == "greet":
            if self.busy:
                await self.cancel("greet")
            emotion, action, text = persona.pick(persona.GREETINGS)
            self._start("say", lambda gen, t: self._say_turn(gen, t, emotion, action, text, source="greeting"))
        elif kind == "attract":
            self.attract = bool(msg.get("on")) and self.cfg.attract_s > 0
            self.touch()
        elif kind == "reset":
            await self.reset("operator")
        elif kind == "cancel":
            await self.cancel("operator")
        elif kind == "playback":
            self._on_playback(msg)
        elif kind == "ping":
            await self._send_json({"v": 1, "type": "pong", "t": msg.get("t")})
        else:
            log.debug("ignoring message type %r", kind)

    async def handle_audio(self, data: bytes) -> None:
        try:
            frame = parse_mic_frame(data)
        except FrameError as e:
            self.bad_frames += 1
            if self.bad_frames <= 5 or self.bad_frames % 100 == 0:
                log.warning("dropping malformed mic frame (%d so far): %s", self.bad_frames, e)
            return
        if self.expected_seq is not None and frame.seq != self.expected_seq:
            self.seq_gaps += 1
            log.info("mic sequence gap: expected %d got %d", self.expected_seq, frame.seq)
        self.expected_seq = (frame.seq + 1) & 0xFFFFFFFF
        if not self.listening:
            return
        if len(self.audio) < self.cfg.max_utterance_s * MIC_SAMPLE_RATE * 2:
            self.audio.extend(frame.pcm)

    # ---- control ---------------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    def touch(self) -> None:
        self.last_activity = time.monotonic()
        self.next_attract = self.last_activity + self.cfg.attract_s * random.uniform(0.6, 1.4)

    async def cancel(self, reason: str) -> None:
        old = self.generation
        if self.task and not self.task.done():
            self.task.cancel()
            self._log("cancelled", reason=reason)
        self.generation += 1  # anything still in flight for `old` is now stale
        if (t := self.timings.pop(old, None)) is not None:
            t.outcome = f"cancelled:{reason}"
            self.metrics.record(t)
        await self._send_json({"v": 1, "type": "cancel", "generation_id": old, "reason": reason})

    async def reset(self, reason: str) -> None:
        await self.cancel("reset")
        self.listening = False
        self.audio.clear()
        await self._forget(reason)

    async def _forget(self, reason: str) -> None:
        self.history.clear()
        self.nudges = 0
        self._log("reset", reason=reason)
        self.convo = new_id()
        await self._send_json({"v": 1, "type": "session_reset", "reason": reason})
        log.info("session reset (%s)", reason)

    async def idle_watchdog(self) -> None:
        while True:
            await asyncio.sleep(1)
            await self.idle_tick()

    async def idle_tick(self) -> None:
        """If he spoke and nobody answered, ask if they're still there (max_nudges times); if that gets no
        answer either, say goodbye and forget them, so the next group gets a fresh ghost."""
        if self.busy or self.listening:
            return
        if not self.history:
            # Nobody to talk to: call out now and then, to draw people over.
            if self.attract and time.monotonic() > self.next_attract:
                emotion, action, text = persona.pick(persona.ATTRACT_LINES)
                self.last_attract = (time.monotonic(), f"{persona.remembered_tag(Expression(emotion, action))} {text}")
                self._start("attract", lambda gen, t: self._say_turn(gen, t, emotion, action, text, remember=False, source="attract"))
            return
        quiet = time.monotonic() - self.last_activity
        if self.cfg.nudge_s and quiet > self.cfg.nudge_s:
            if self.nudges < self.cfg.max_nudges:
                self.nudges += 1
                emotion, action, text = persona.pick(persona.NUDGE_LINES)
                log.info("nobody answered; nudge %d", self.nudges)
                # Not remembered: the LLM should carry on from the conversation, not from the prompting.
                self._start("nudge", lambda gen, t: self._say_turn(gen, t, emotion, action, text, remember=False, source="nudge"))
            else:
                emotion, action, text = persona.pick(persona.LONELY_LINES)
                log.info("nobody answered %d nudges; giving up", self.nudges)
                self._start("nudge", lambda gen, t: self._give_up(gen, t, emotion, action, text))
        elif quiet > self.cfg.idle_reset_s:  # only reached with nudging off
            await self.reset("idle")

    async def _give_up(self, gen: int, timing: TurnTiming, emotion: str, action: str, text: str) -> None:
        await self._say_turn(gen, timing, emotion, action, text, remember=False, source="giving_up")
        if gen == self.generation:  # nobody spoke up while he was saying it
            await self._forget("nobody there")

    async def close(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()

    # ---- turns -----------------------------------------------------------------------------

    def _start(self, kind: str, body: Callable[[int, TurnTiming], Awaitable[None]]) -> None:
        self.generation += 1
        gen = self.generation
        timing = TurnTiming(gen, kind)
        self.timings[gen] = timing
        for old in [g for g in self.timings if g < gen - 20]:
            del self.timings[old]
        self.touch()
        self.task = asyncio.create_task(self._run(gen, timing, body))

    async def _run(self, gen: int, timing: TurnTiming, body: Callable[[int, TurnTiming], Awaitable[None]]) -> None:
        try:
            await body(gen, timing)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("turn %d failed", gen)
            timing.outcome = "error"
            await self._send(gen, {"type": "state", "state": "error"})
        timing.mark("total_ms")
        # Turns that produced audio are recorded when the browser reports playback ended.
        if timing.first_audio_sent_ms is None and self.timings.pop(gen, None) is not None:
            self.metrics.record(timing)

    async def _voice_turn(self, gen: int, timing: TurnTiming, pcm: bytes) -> None:
        await self._send(gen, {"type": "state", "state": "thinking"})
        tr = await self.stt.transcribe(pcm)
        timing.mark("stt_ms")
        await self._send(gen, {
            "type": "transcript", "text": tr.text, "rejected": tr.rejected,
            "duration_s": round(tr.duration_s, 2), "avg_logprob": round(tr.avg_logprob, 2),
            "no_speech_prob": round(tr.no_speech_prob, 2),
        })
        if tr.text.strip():  # ignored ones too ("rejected" says why): useful for tuning
            self._log("visitor", text=tr.text, via="voice", rejected=tr.rejected)
        if tr.rejected == "low_confidence":
            timing.outcome = "asked_repeat"
            emotion, action, line = persona.pick(persona.REPEAT_LINES)
            await self._say_turn(gen, timing, emotion, action, line, remember=False, source="asked_repeat")
            return
        if tr.rejected:
            timing.outcome = f"ignored:{tr.rejected}"
            await self._send(gen, {"type": "state", "state": "idle"})
            return
        await self._llm_turn(gen, timing, tr.text)

    async def _text_turn(self, gen: int, timing: TurnTiming, text: str) -> None:
        await self._send(gen, {"type": "state", "state": "thinking"})
        await self._send(gen, {"type": "transcript", "text": text, "rejected": None, "typed": True})
        self._log("visitor", text=text, via="typed", rejected=None)
        await self._llm_turn(gen, timing, text)

    async def _say_turn(self, gen: int, timing: TurnTiming, emotion: str, action: str, text: str, remember: bool = True,
                        source: str = "canned") -> None:
        tag = persona.parse_tag(f"[{emotion}:{action}]")
        parser = ReplyParser()
        events = parser.feed(f"[{tag.emotion}:{tag.action}] {text}") + parser.finish()
        queue: asyncio.Queue[Sentence | None] = asyncio.Queue()
        for ev in events:
            if isinstance(ev, Expression):
                await self._send_expression(gen, ev)
            else:
                queue.put_nowait(ev)
        queue.put_nowait(None)
        spoken = await self._speak(gen, timing, queue)
        if remember and spoken:
            self.history.append({"role": "assistant", "content": f"{persona.remembered_tag(tag)} " + " ".join(spoken)})
        self._log_reply(timing, " ".join(spoken), tag.emotion, tag.action, source)
        await self._send(gen, {"type": "reply", "text": " ".join(spoken)})

    async def _llm_turn(self, gen: int, timing: TurnTiming, user_text: str) -> None:
        self.nudges = 0
        if fixed := persona.fixed_reply(user_text):
            # Things the LLM has got wrong (inviting children in, keeping secrets from parents, joking with a
            # lost child): a fixed answer instead, and for a child who may need help, an alert for the operator.
            why, lines = fixed
            self.last_attract = None
            self.history.append({"role": "user", "content": user_text})
            if why == "needs_help":
                log.warning("a visitor may need help: %r", user_text)
                self._log("alert", reason=why, text=user_text)
                await self._send(gen, {"type": "alert", "reason": why, "text": user_text})
            emotion, action, line = persona.pick(lines)
            await self._say_turn(gen, timing, emotion, action, line, source=why)
            return
        # Answering what he just called out ("I've got riddles!") needs him to remember he said it.
        if not self.history and self.last_attract and time.monotonic() - self.last_attract[0] < 90:
            self.history.append({"role": "assistant", "content": self.last_attract[1]})
        self.last_attract = None
        self.history.append({"role": "user", "content": user_text})
        messages = self._build_messages()
        queue: asyncio.Queue[Sentence | None] = asyncio.Queue()
        reply: dict = {"tag": None}
        producer = asyncio.create_task(self._produce(gen, timing, messages, queue, reply))
        try:
            spoken = await self._speak(gen, timing, queue, first_timeout=self.cfg.llm_first_sentence_timeout_s,
                                       filler=self._pick_filler())
        finally:
            producer.cancel()
        if spoken:
            self.history.append({"role": "assistant", "content": f"{reply['tag'] or '[neutral]'} " + " ".join(spoken)})
        elif self.history and self.history[-1]["role"] == "user":
            self.history.pop()
        tag = persona.parse_tag(reply["tag"] or "[neutral]")
        self._log_reply(timing, " ".join(spoken), tag.emotion, tag.action, "llm" if timing.outcome == "ok" else timing.outcome)
        await self._send(gen, {"type": "reply", "text": " ".join(spoken)})
        if gen == self.generation and (persona.is_goodbye(user_text) or persona.is_farewell(" ".join(spoken))):
            # Either of them said goodbye, and his reply has finished; forget them without cutting that off.
            await self._forget("goodbye" if persona.is_goodbye(user_text) else "he said goodbye")

    def _log(self, kind: str, **fields) -> None:
        self.convo_log.write({"conn": self.conn, "convo": self.convo, "kind": kind, **fields})

    def _log_reply(self, timing: TurnTiming, text: str, emotion: str, action: str, source: str) -> None:
        if text:
            self._log("ghost", text=text, emotion=emotion, action=action, source=source,
                      latency_ms=timing.first_audio_sent_ms)

    def _build_messages(self) -> list[dict]:
        # Some chat templates (Gemma) insist on strict user/assistant alternation starting with user.
        turns: list[dict] = []
        for m in self.history:
            if turns and turns[-1]["role"] == m["role"]:
                turns[-1] = {"role": m["role"], "content": turns[-1]["content"] + " " + m["content"]}
            else:
                turns.append(dict(m))
        if turns and turns[0]["role"] == "assistant":
            turns.insert(0, {"role": "user", "content": "(Some visitors float up to you.)"})
        return [{"role": "system", "content": persona.system_prompt(self.cfg.ghost_name, sounds=self.tts.sound_tags, now=datetime.now())}, *turns]

    async def _produce(self, gen: int, timing: TurnTiming, messages: list[dict], queue: asyncio.Queue, reply: dict) -> None:
        parser = ReplyParser(eager_first=self.tts.whole_sentences)
        sentences = 0

        async def handle(events: list[Expression | Sentence]) -> bool:
            nonlocal sentences
            for ev in events:
                if isinstance(ev, Expression):
                    if reply["tag"] is None:
                        reply["tag"] = persona.remembered_tag(ev)
                    await self._send_expression(gen, ev)
                    continue
                swap = (("invite_redirect", persona.STAY_OUTSIDE_LINES) if persona.is_invitation(ev.text) else
                        ("secret_redirect", persona.SECRET_LINES) if persona.is_secret_keeping(ev.text) else None)
                if swap:
                    log.warning("LLM output replaced (%s)", swap[0])
                    timing.outcome = swap[0]
                    emotion, action, line = persona.pick(swap[1])
                    await self._send_expression(gen, Expression(emotion, action))
                    await queue.put(Sentence(line))
                    sentences += 1
                    return False
                if persona.is_unsafe(ev.text):
                    log.warning("unsafe LLM output replaced")
                    timing.outcome = "safety_redirect"
                    emotion, action, line = persona.SAFE_REDIRECT
                    await self._send_expression(gen, Expression(emotion, action))
                    await queue.put(Sentence(line))
                    sentences += 1
                    return False
                timing.mark("first_sentence_ms")
                await queue.put(ev)
                sentences += 1
                if sentences >= MAX_SENTENCES:
                    return False
            return True

        try:
            async with asyncio.timeout(self.cfg.llm_total_timeout_s):
                async with contextlib.aclosing(self.llm.stream(messages)) as stream:
                    async for token in stream:
                        timing.mark("first_token_ms")
                        if not await handle(parser.feed(token)):
                            return
                await handle(parser.finish())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("LLM failed after %d sentences: %r", sentences, e)
            if sentences == 0:
                timing.outcome = "llm_fallback"
                emotion, action, line = persona.pick(persona.FALLBACK_LINES)
                await self._send_expression(gen, Expression(emotion, action))
                await queue.put(Sentence(line))
        finally:
            queue.put_nowait(None)

    def _pick_filler(self) -> bytes | None:
        fillers = self.tts.fillers
        if not fillers:
            return None
        choices = [i for i in range(len(fillers)) if i != self._last_filler] or [0]
        self._last_filler = random.choice(choices)
        return fillers[self._last_filler]

    async def _speak(self, gen: int, timing: TurnTiming, queue: asyncio.Queue, first_timeout: float | None = None,
                     filler: bytes | None = None) -> list[str]:
        """Synthesise queued sentences in order and stream them to the presenter.

        A filler (pre-made reaction noise) plays first, while the real reply is still being made."""
        spoken: list[str] = []
        seq = 0
        started = False
        stop_after = False
        rude = False  # one rude noise per reply is plenty

        async def send_audio(pcm: bytes, noise: str | None = None) -> bool:
            nonlocal seq, started
            if not started:
                started = await self._send(gen, {
                    "type": "speech_start",
                    "audio_format": {"codec": "pcm_s16le", "sample_rate": self.tts.sample_rate, "channels": 1},
                })
            if gen != self.generation:
                return False
            await self._send_bytes(pack_audio_chunk(gen, seq, self.tts.sample_rate, pcm, noise=noise))
            seq += 1
            return True

        if filler:
            step = self.tts.sample_rate // 4 * 2  # 250 ms per message, like the voices
            for i in range(0, len(filler), step):
                if not await send_audio(filler[i : i + step]):
                    break
        while not stop_after:
            if first_timeout is not None and not spoken:
                try:
                    item = await asyncio.wait_for(queue.get(), first_timeout)
                except TimeoutError:
                    log.warning("no sentence from LLM within %.1fs; using fallback", first_timeout)
                    timing.outcome = "llm_timeout"
                    emotion, action, line = persona.pick(persona.FALLBACK_LINES)
                    await self._send_expression(gen, Expression(emotion, action))
                    item, stop_after = Sentence(line), True
            else:
                item = await queue.get()
            if item is None:
                break
            if gen != self.generation:
                break
            try:
                for kind, part in persona.split_noises(item.text):
                    if kind == "noise":
                        # One flagged chunk, so the presenter animates exactly as it's heard.
                        if not rude:
                            rude = True
                            if not await send_audio(noises.pcm(part, self.tts.sample_rate), noise=part):
                                return spoken
                        continue
                    text = part if self.tts.sound_tags else persona.strip_sounds(part)
                    # aclosing: when a reply is cancelled mid-sentence, the voice must stop (and release the
                    # GPU) now, not whenever the abandoned generator happens to be garbage collected.
                    async with contextlib.aclosing(self.tts.synth(text)) as audio:
                        async for pcm in audio:
                            timing.mark("first_tts_chunk_ms")
                            if not await send_audio(pcm):
                                return spoken
                            timing.mark("first_audio_sent_ms")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("TTS failed for %r: %r", item.text[:40], e)
                continue
            spoken.append(item.text)
        if started:
            await self._send(gen, {"type": "speech_end", "chunks": seq})
        else:
            # Nothing audible came out (TTS down or empty reply): show the gentle error pose.
            await self._send(gen, {"type": "state", "state": "error"})
        return spoken

    # ---- outbound --------------------------------------------------------------------------

    async def _send(self, gen: int, msg: dict) -> bool:
        if gen != self.generation:
            return False
        await self._send_json({"v": 1, "generation_id": gen, **msg})
        return True

    async def _send_expression(self, gen: int, ev: Expression) -> None:
        await self._send(gen, {"type": "expression", "emotion": ev.emotion, "action": ev.action, "intensity": ev.intensity})

    def _on_playback(self, msg: dict) -> None:
        self.touch()
        gen = msg.get("generation_id")
        timing = self.timings.get(gen) if isinstance(gen, int) else None
        if timing is None:
            return
        event = msg.get("event")
        if event == "started":
            timing.mark("audible_ms")
        elif event in ("ended", "cancelled"):
            self.timings.pop(gen, None)
            self.metrics.record(timing)
