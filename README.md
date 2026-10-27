# Project Spectre: Boolian the Halloween ghost

<p align="center"><img src="docs/boolian.png" alt="Boolian, a smiling cyan cartoon ghost glowing on a dark background" width="420"></p>

A friendly cartoon ghost projected from a MacBook. He listens through a microphone, thinks with local AI
(Whisper → llama.cpp → Kokoro on a CUDA VM) and answers with a matching face, mouth and voice.
If the AI breaks, an operator can puppet him with hotkeys.

```
MacBook browser (stage window → projector)  ←── WebSocket ──→  VM: Spectre server ── Whisper (in-process)
  ghost renderer · mic capture · playback                                          ├─ llama-server :8080
MacBook browser (remote window → laptop screen)                                    └─ Kokoro-FastAPI :8880
  operator controls, linked to the stage window
```

## Try it now on the Mac (no GPU, no VM)

```bash
# terminal 1: server with fake STT/LLM/TTS
cd server && uv venv --python 3.12 && uv pip install -e '.[dev]'
SPECTRE_MOCK=1 .venv/bin/python -m app.main

# terminal 2: renderer
cd renderer && npm install && npm run dev
```

Open <http://localhost:5173/?operator>, click **wake**, then hold **Space** and talk, or type in "Visitor says".
The mock ghost babbles in a warbly voice and cycles through canned replies.

To use a real LLM on the Mac as well, any OpenAI-compatible server works. This was tested with Ollama's `llama3.2`:

```bash
SPECTRE_STT_MODEL=base.en SPECTRE_STT_DEVICE=cpu SPECTRE_STT_COMPUTE_TYPE=int8 \
SPECTRE_LLM_URL=http://localhost:11434/v1 SPECTRE_LLM_MODEL=llama3.2 SPECTRE_TTS_BACKEND=mock \
  .venv/bin/python -m app.main          # needs: uv pip install faster-whisper
```

## GPU setup without Docker (Proxmox / CUDA containers)

Two containers with the GPU passed through, each with a clone of this repo:

| Container | Runs | Commands |
|---|---|---|
| **llm** | llama-server on :8080 | `scripts/llm-container.sh setup`, then `run` (or `service`) |
| **spectre** | Spectre on :8765, with Whisper and Kokoro in-process | `scripts/spectre-container.sh setup`, then `selftest`, then `run` (or `service`) |

- `llm-container.sh setup` reuses an existing `llama-server`, or builds llama.cpp with CUDA. Building needs `nvcc`,
  so use a CUDA *devel* image. The model is set in `llm.env`: by default Gemma 4 12B (4-bit, ~7.5 GB), downloaded from
  Hugging Face on first run. Set `LLM_MODEL=/path/model.gguf` to use a local file.
- `spectre-container.sh setup` installs espeak-ng, uv, Python 3.12, PyTorch (CUDA), faster-whisper and Kokoro,
  then writes `spectre.env`. **Edit `SPECTRE_LLM_URL`** to point at the llm container.
- `selftest` has Kokoro say a line, Whisper transcribe it back, and the LLM answer a prompt.
  It reports timings and VRAM, and tells you exactly which part failed.
- `service` installs a systemd unit so it starts with the container (`journalctl -fu spectre`).

On the Mac, open `http://localhost:5173/?server=ws://SPECTRE_CONTAINER_IP:8765/ws`.

## VM setup with Docker (CUDA, 16 GB)

1. Install Docker plus the NVIDIA Container Toolkit, and check that `docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` works.
2. Download an LLM. Llama 3.2 3B Instruct follows Boolian's tag format reliably in testing; a 4B model such as
   Qwen3-4B-Instruct-2507 or Gemma 3 4B is worth trying if you have time.
   ```bash
   mkdir -p infra/models
   huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir infra/models
   ln -sf Llama-3.2-3B-Instruct-Q4_K_M.gguf infra/models/model.gguf
   ```
3. `cd infra && docker compose up -d --build`. The first start downloads Whisper `large-v3-turbo` (~1.6 GB).
4. Wait until `curl localhost:8765/health/ready` returns `"ok": true`.
5. Test from any machine: `server/.venv/bin/python scripts/smoke_test.py ws://VM_IP:8765/ws --turns 10`.
   Add `--wav some-speech.wav` to exercise Whisper as well. The script prints p50/p95 latency.

VRAM estimate: a 3B/4B Q4 model with 4k context uses about 3–4 GB, Whisper large-v3-turbo fp16 about 2 GB and Kokoro under 1 GB,
which leaves plenty of headroom. Check with `nvidia-smi` while a turn is running.

Pick Boolian's voice with `SPECTRE_TTS_VOICE` (Kokoro voices such as `bm_fable`, `bm_george`, `bm_lewis`, `am_puck`).
The browser adds reverb and a slight chorus on top, which you can adjust with the Reverb slider.

All settings are `SPECTRE_*` environment variables; see `server/app/config.py`.
`SPECTRE_AUTH_TOKEN` enables a shared token: append `?token=…` to the page URL.

## On the night: runbook

**Setup**
1. Start the VM stack and confirm `/health/ready` is OK.
2. On the MacBook: `cd renderer && npm run dev`. The page must be served from **localhost**, because browsers
   only allow the microphone on secure origins.
3. Make the projector an extended display. In System Settings → Sound, choose the output speaker (wired if possible).
4. Open `http://localhost:5173/?server=ws://VM_IP:8765/ws` (the URL is remembered afterwards). Drag the window
   to the projector, **click wake**, then press **F** for fullscreen.
5. Press **O** and click **Open remote control window**. Drag the new window to the laptop screen. You operate
   from there; the projector shows only the ghost. Press **O** in the stage window again to hide its panel.
6. Press **K** for the test pattern, then line up scale and position with the arrow keys and +/-. Press **K** again to exit.
7. If you use Bluetooth speakers, raise **Speaker delay** until the mouth matches the sound.
8. Hold Space and say "hello". Check the log and latency readout.

**Running**
- **Hold Space** while a child talks, then release. This is the most reliable mode. Auto-listen (**V**) works
  in a quiet room, but groups of excited trick-or-treaters will trigger it constantly.
- **G** greets the group, which helps because kids don't know they can talk to him. **R** forgets the visitor;
  this also happens automatically after 40 s of quiet.
- **1–9** play puppet lines, and the "Boolian says" box speaks any typed text in his voice. Both still work if the
  server is down, using the browser's own voice.
- **X** / **Esc** stops him talking. **M** mutes.
- If he starts answering himself: turn auto-listen off and use Space.
- If the server dies: he falls asleep (Zzz), puppet lines still work, and he wakes when the server returns.

**Shutdown**: close the browser windows, then run `docker compose down` on the VM.

## Controls

| Key | Action | Key | Action |
|---|---|---|---|
| Space (hold) | listen | V | auto-listen on/off |
| G | greet | R | forget visitor |
| X / Esc | stop talking | M | mute |
| 1–9 | puppet lines | O | operator panel |
| F | fullscreen | K | projection test pattern |
| arrows | move ghost | + / - | scale |

Debug and art mode: `?state=speaking&emotion=mischievous&mouth=0.6` freezes a pose without a server.
`?seed=N` makes idle behaviour repeatable.

## Layout

```
server/      FastAPI + asyncio orchestration (session.py is the heart), pytest suite
renderer/    Vite + TypeScript, Canvas 2D ghost (no runtime dependencies)
scripts/     smoke_test.py: end-to-end turns and latency against a running server
infra/       docker compose for the VM
docs/        protocol-v1.md
```

## Deliberate deviations from the plan

- **No Pipecat.** In half-duplex mode, a hand-written turn loop (`server/app/session.py`) is simpler and fully under our control.
  Barge-in is supported by the operator (Space while he talks); automatic barge-in needs echo cancellation.
- **Canvas 2D instead of PixiJS**: one procedural character doesn't need a scene graph. It runs at 60 fps with zero dependencies.
- **Browser mic, not ESP32 (yet).** The server already accepts the ESP32 frame format (`GHST`) on the same socket.
  ESP32 support needs a small ingress change so a separate device can feed the presenter's conversation.
- **Emotion via an inline `[emotion:action]` tag**, validated and stripped server-side, with a fallback to neutral.

## Licence

MIT. See [LICENSE](LICENSE).
