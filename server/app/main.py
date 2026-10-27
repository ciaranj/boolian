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
    log.info("presenter connected from %s", ws.client)
    try:
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
        watchdog.cancel()
        await convo.close()
        with contextlib.suppress(Exception):
            await ws.close()
        log.info("presenter disconnected")


if cfg.static_dir and Path(cfg.static_dir).is_dir():
    app.mount("/", StaticFiles(directory=cfg.static_dir, html=True), name="renderer")


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    run()
