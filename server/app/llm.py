"""LLM backends. The real one talks to any OpenAI-compatible server (llama.cpp's llama-server)."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from collections.abc import AsyncIterator

import httpx

from .config import Config

log = logging.getLogger(__name__)


class OpenAICompatLLM:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(cfg.llm_total_timeout_s, connect=3.0))

    async def healthy(self) -> bool:
        try:
            r = await self.client.get(f"{self.cfg.llm_url}/models", timeout=2.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def warmup(self) -> None:
        """Make the server load the model now, not when the first trick-or-treater speaks."""
        try:
            async for _ in self.stream([{"role": "user", "content": "Say hi."}], max_tokens=1):
                pass
            log.info("LLM warm")
        except Exception as e:
            log.warning("LLM warm-up failed (is llama-server running?): %r", e)

    async def stream(self, messages: list[dict], max_tokens: int | None = None) -> AsyncIterator[str]:
        body = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens or self.cfg.llm_max_tokens,
            "temperature": self.cfg.llm_temperature,
        }
        async with self.client.stream("POST", f"{self.cfg.llm_url}/chat/completions", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if content := delta.get("content"):
                    yield content

    async def aclose(self) -> None:
        await self.client.aclose()


class MockLLM:
    REPLIES = itertools.cycle([
        "[happy:bounce] Hello hello! I'm Boolian. I've been floating about here for 173 years, and you're my favourite visitor tonight.",
        "[curious:tilt] My name? Oh, it's Boolian. Or was it Bartholomew? One forgets these things after a century or two.",
        "[mischievous:spin] Scary? Me? Watch this. Boo! ...Hmm, nobody ever screams. Did I do it right?",
        "[surprised:bounce] A pumpkin! How marvellous! Are you the kind with a candle inside?",
        "[confused] How old? Well, I stopped counting at about a hundred. Ghosts are terrible at birthdays.",
        "[mischievous:hide] A trick? Ta-da! I've vanished! ...Wait, can you still see me?",
    ])

    async def healthy(self) -> bool:
        return True

    async def warmup(self) -> None:
        pass

    async def stream(self, messages: list[dict], max_tokens: int | None = None) -> AsyncIterator[str]:
        await asyncio.sleep(0.25)
        for word in next(self.REPLIES).split(" "):
            await asyncio.sleep(0.02)
            yield word + " "

    async def aclose(self) -> None:
        pass


def make_llm(cfg: Config) -> OpenAICompatLLM | MockLLM:
    return MockLLM() if cfg.llm_backend == "mock" else OpenAICompatLLM(cfg)
