import json
import os

os.environ["SPECTRE_MOCK"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.protocol import pack_mic_frame  # noqa: E402


def next_json(ws, kind):
    while True:
        msg = ws.receive()
        if msg.get("text") and (m := json.loads(msg["text"]))["type"] == kind:
            return m


def test_the_esp32s_microphone_and_motion_reach_the_presenter():
    with TestClient(app) as client, client.websocket_connect("/ws") as presenter:
        assert next_json(presenter, "mic_device")["connected"] is False
        presenter.send_text(json.dumps({"type": "hello", "role": "presenter"}))
        presenter.send_text(json.dumps({"type": "mic_source", "source": "device"}))
        next_json(presenter, "welcome")
        with client.websocket_connect("/mic") as mic:
            assert next_json(presenter, "mic_device")["connected"] is True
            mic.send_text(json.dumps({"type": "hello", "role": "mic", "device": "porch"}))
            assert next_json(presenter, "mic_device")["name"] == "porch"
            frame = pack_mic_frame(7, 140, bytes(640))
            mic.send_bytes(frame)
            assert presenter.receive_bytes() == frame
            mic.send_text(json.dumps({"type": "stats", "rssi": -52, "db": -61.5, "junk": "x"}))
            status = next_json(presenter, "mic_device")
            assert status["rssi"] == -52 and "junk" not in status
            mic.send_text(json.dumps({"type": "motion", "on": True}))
            assert next_json(presenter, "motion")["on"] is True
        assert next_json(presenter, "mic_device")["connected"] is False


def test_a_silent_esp32_is_dropped_and_shown_as_disconnected(monkeypatch):
    import dataclasses

    import app.main as main

    monkeypatch.setattr(main, "cfg", dataclasses.replace(main.cfg, mic_idle_s=0.2))
    with TestClient(app) as client, client.websocket_connect("/ws") as presenter:
        assert next_json(presenter, "mic_device")["connected"] is False
        presenter.send_text(json.dumps({"type": "hello", "role": "presenter"}))
        with client.websocket_connect("/mic") as mic:
            assert next_json(presenter, "mic_device")["connected"] is True
            # Sends nothing: a link that died without closing.
            assert next_json(presenter, "mic_device")["connected"] is False
            assert mic.receive()["type"] == "websocket.close"
