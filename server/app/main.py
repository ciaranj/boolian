"""FastAPI entry point: presenter WebSocket, health and metrics."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from contextlib import asynccontextmanager

# Chatterbox draws a progress bar for every sentence, which would flood the server log. tqdm reads this
# when it's first imported, so it has to be set before any model library loads.
os.environ.setdefault("TQDM_DISABLE", "1")
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import load_config
from .llm import make_llm
from .metrics import Metrics
from .protocol import FrameError, parse_mic_frame
from .session import Conversation
from .convolog import ConversationLog
from .stt import make_stt
from .tts import make_tts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("spectre")

cfg = load_config()
stt = make_stt(cfg)
llm = make_llm(cfg)
tts = make_tts(cfg)
metrics = Metrics()
convo_log = ConversationLog(cfg.conversation_log)
presenters: set[Conversation] = set()
mic_status: dict = {"connected": False}  # the ESP32, as last reported to the presenters
current_mic: object | None = None  # only the latest ESP32 to connect is listened to


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("backends: stt=%s llm=%s tts=%s", cfg.stt_backend, cfg.llm_backend, cfg.tts_backend)
    log.info("conversation log: %s", cfg.conversation_log or "off")
    # Model loading takes a while; /health/ready reports when everything is up.
    tasks = [asyncio.create_task(stt.load()), asyncio.create_task(llm.warmup())]
    if hasattr(tts, "load"):  # in-process TTS loads its model too
        tasks.append(asyncio.create_task(tts.load()))
    yield
    for t in tasks:
        t.cancel()
    await llm.aclose()
    await tts.aclose()


app = FastAPI(title="Project Spectre", lifespan=lifespan)


@app.get("/health/live")
async def live():
    return {"ok": True}


@app.get("/health/ready")
async def ready():
    llm_ok, tts_ok = await asyncio.gather(llm.healthy(), tts.healthy())
    checks = {"stt": stt.ready, "llm": llm_ok, "tts": tts_ok}
    return JSONResponse({"ok": all(checks.values()), **checks}, status_code=200 if all(checks.values()) else 503)


@app.get("/metrics")
async def get_metrics():
    return metrics.summary()


@app.websocket("/ws")
async def presenter(ws: WebSocket):
    if cfg.auth_token and ws.query_params.get("token") != cfg.auth_token:
        await ws.close(code=4401)
        return
    await ws.accept()
    lock = asyncio.Lock()

    async def send_json(msg: dict) -> None:
        async with lock:
            await ws.send_text(json.dumps(msg))

    async def send_bytes(data: bytes) -> None:
        async with lock:
            await ws.send_bytes(data)

    convo = Conversation(cfg, stt, llm, tts, metrics, send_json, send_bytes, convo_log)
    watchdog = asyncio.create_task(convo.idle_watchdog())
    presenters.add(convo)
    log.info("presenter connected from %s", ws.client)
    try:
        await convo.device_status(mic_status)
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                await convo.handle_audio(data)
            elif (text := msg.get("text")) is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    log.warning("ignoring non-JSON text message")
                    continue
                if isinstance(payload, dict):
                    await convo.handle_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        presenters.discard(convo)
        watchdog.cancel()
        await convo.close()
        with contextlib.suppress(Exception):
            await ws.close()
        log.info("presenter disconnected")


async def _to_presenters(fn) -> None:
    for convo in list(presenters):
        try:
            await fn(convo)
        except Exception as e:  # a presenter that's going away mustn't stop the others hearing
            log.debug("couldn't reach a presenter: %s", e)


@app.websocket("/mic")
async def mic_device(ws: WebSocket):
    """The ESP32: microphone frames (GHST, as in docs/protocol-v1.md) and PIR motion, relayed to the presenters."""
    global mic_status, current_mic
    if cfg.auth_token and ws.query_params.get("token") != cfg.auth_token:
        await ws.close(code=4401)
        return
    await ws.accept()
    me = object()
    current_mic = me
    status = {"connected": True, "name": "esp32", "gaps": 0, "bad": 0}
    expected: int | None = None
    log.info("mic device connected from %s", ws.client)

    async def publish(**changes) -> None:
        global mic_status
        status.update(changes)
        mic_status = dict(status)
        await _to_presenters(lambda c: c.device_status(mic_status))

    await publish()
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), cfg.mic_idle_s)
            except asyncio.TimeoutError:
                # It streams 50 frames a second, so silence means the link has died without closing (a Wi-Fi blip,
                # a reboot). Hang up, so the panel says it's gone and the ESP32 reconnects.
                log.warning("mic device silent for %.0f s: dropping it", cfg.mic_idle_s)
                with contextlib.suppress(Exception):
                    await ws.close(code=1001)
                break
            if msg["type"] == "websocket.disconnect":
                break
            if current_mic is not me:
                continue
            if (data := msg.get("bytes")) is not None:
                try:
                    frame = parse_mic_frame(data)
                except FrameError as e:
                    status["bad"] += 1
                    if status["bad"] <= 5:
                        log.warning("dropping malformed frame from the mic device: %s", e)
                    continue
                if expected is not None and frame.seq != expected:
                    status["gaps"] += 1
                expected = (frame.seq + 1) & 0xFFFFFFFF
                await _to_presenters(lambda c: c.device_frame(data))
            elif (text := msg.get("text")) is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                kind = payload.get("type")
                if kind == "hello":
                    expected = None  # a new stream: its sequence starts again
                    await publish(name=str(payload.get("device", "esp32"))[:40])
                elif kind == "motion":
                    on = bool(payload.get("on"))
                    log.info("motion %s", "seen" if on else "stopped")
                    await _to_presenters(lambda c: c.motion(on))
                elif kind == "stats":
                    # Once every few seconds: signal, levels and the device's own drop counts, for the operator.
                    keep = ("rssi", "db", "peak_db", "clips", "dropped", "reconnects", "heap", "uptime_s")
                    await publish(**{k: payload[k] for k in keep if isinstance(payload.get(k), (int, float))})
    except WebSocketDisconnect:
        pass
    finally:
        if current_mic is me:
            current_mic = None
            mic_status = {"connected": False}
            await _to_presenters(lambda c: c.device_status(mic_status))
        log.info("mic device disconnected")


if cfg.static_dir and Path(cfg.static_dir).is_dir():
    app.mount("/", StaticFiles(directory=cfg.static_dir, html=True), name="renderer")


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    run()
