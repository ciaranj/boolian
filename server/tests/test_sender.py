import asyncio
import time

import pytest

from app.sender import BrowserGone, DeadlineSender


async def test_sends_go_through_in_order():
    got = []
    send = DeadlineSender(1.0, on_stuck=lambda: pytest.fail("not stuck"))

    async def put(x):
        await asyncio.sleep(0.01)
        got.append(x)

    await asyncio.gather(*(send(lambda x=x: put(x)) for x in range(5)))
    assert got == [0, 1, 2, 3, 4]


async def test_a_browser_that_stops_taking_messages_is_given_up_on():
    stuck = []
    send = DeadlineSender(0.1, on_stuck=lambda: stuck.append(1))
    t0 = time.monotonic()
    with pytest.raises(BrowserGone):
        await send(lambda: asyncio.sleep(60))  # asleep with the page open: its buffer is full
    assert time.monotonic() - t0 < 1
    assert stuck == [1]

    called = []

    async def record():
        called.append(1)

    with pytest.raises(BrowserGone):  # and from then on, without even trying
        await send(record)
    assert called == [] and stuck == [1]


async def test_sends_queued_behind_a_stuck_one_give_up_too():
    stuck = []
    send = DeadlineSender(0.1, on_stuck=lambda: stuck.append(1))
    results = await asyncio.gather(send(lambda: asyncio.sleep(60)), send(lambda: asyncio.sleep(0)),
                                   return_exceptions=True)
    assert all(isinstance(r, BrowserGone) for r in results)
    assert stuck == [1]
