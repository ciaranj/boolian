"""Sending to one browser without letting it hold up anyone else.

A browser that has gone to sleep with the page open never closes its connection, and once its buffer fills a send
to it waits for minutes. The ESP32's frames are relayed to each browser in turn, so one such browser would stop the
rest hearing anything; instead, a send that takes too long gives up on that browser for good.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


class BrowserGone(ConnectionError):
    """The browser didn't take a send in time, and has been given up on."""


class DeadlineSender:
    def __init__(self, deadline_s: float, on_stuck: Callable[[], None]) -> None:
        self.deadline_s = deadline_s
        self.on_stuck = on_stuck  # called once, when a send first misses the deadline
        self.gone = False
        self._lock = asyncio.Lock()  # one message at a time, so frames never interleave

    async def __call__(self, send: Callable[[], Awaitable[None]]) -> None:
        if self.gone:
            raise BrowserGone()
        try:
            async with asyncio.timeout(self.deadline_s):
                async with self._lock:
                    await send()
        except TimeoutError:
            if not self.gone:
                self.gone = True
                self.on_stuck()
            raise BrowserGone() from None
