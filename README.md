# Project Spectre: Boolian the Halloween ghost

<p align="center"><img src="docs/boolian.png" alt="Boolian, a smiling cyan cartoon ghost glowing on a dark background" width="420"></p>

A friendly, cheeky British cartoon ghost, projected from a laptop's web browser. He listens through a microphone, thinks with
local AI (Whisper → Gemma 4 12B in llama.cpp → a cloned Chatterbox-Turbo voice, all on one 16 GB GPU) and answers
with a matching face, mouth, moves and voice. Everything runs on the local network; no cloud services.
If the AI breaks, an operator can puppet him with hotkeys.

```
Laptop browser (stage window → projector)   ←── WebSocket ──→  spectre container :8765
  ghost renderer · mic + Silero VAD · playback · voice effects     Spectre server, with Whisper and
Laptop browser (remote window → its screen)                         Chatterbox-Turbo in-process
  operator controls, linked to the stage window                          │
                                                                    llm container: llama-server :8080
```

**What he does:** answers in one to three short sentences, with an emotion and a move (bounce, tilt, spin, hide, peek)
chosen per reply. He makes a reaction noise the moment a visitor stops talking, so the wait for the real answer is
covered. He remembers the last 25 exchanges and knows the date and rough time of night. He asks open questions, and
blows the odd raspberry or farts (real recordings, with animations to match). If nobody answers, he asks twice
whether anyone's there, then forgets them; he also forgets them when they say goodbye, or when he does (visitors often just say "thank you" and go).

## Try it now on a laptop (no GPU, no VM)

```bash
# terminal 1: server with fake STT/LLM/TTS
cd server && uv venv --python 3.12 && uv pip install -e '.[dev]'
SPECTRE_MOCK=1 .venv/bin/python -m app.main

# terminal 2: renderer
cd renderer && npm install && npm run dev
```

Any recent Chrome or Edge works (Safari and Firefox are untested). Open <http://localhost:5173/?operator>, click **wake**, then hold **Space** and talk, or type in "Visitor says".
The mock ghost babbles in a warbly voice and cycles through canned replies.

Add `&renderer=3d` for the three.js version: a lit 3D body with bloom, coloured mood lighting and real spins, using
the same face and behaviour. `&q3d=low` drops the bloom for a weaker GPU. Without it you get the original 2D ghost.

He wears a ghostly ribbon bow on his head, in his colour scheme's pink, in both versions; untick **Bow** in the
operator panel (under Projection) to take it off.

To use a real LLM on the laptop as well, any OpenAI-compatible server works. This was tested with Ollama's `llama3.2`:

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
| **spectre** | Spectre on :8765, with Whisper and the voice in-process | `scripts/spectre-container.sh setup`, then `selftest`, then `run` (or `service`) |

- `llm-container.sh setup` reuses an existing `llama-server`, or builds llama.cpp with CUDA. Building needs `nvcc`,
  so use a CUDA *devel* image. The model is set in `llm.env`: by default Gemma 4 12B (4-bit, ~7.5 GB), downloaded from
  Hugging Face on first run. Set `LLM_MODEL=/path/model.gguf` to use a local file. Keep `LLM_CTX=4096`: 8k runs
  out of VRAM alongside the spectre container.
- `spectre-container.sh setup` installs espeak-ng, uv, Python 3.12, PyTorch (CUDA), faster-whisper and Kokoro,
  then writes `spectre.env`. **Edit `SPECTRE_LLM_URL`** to point at the llm container.
- `selftest` has Kokoro say a line, Whisper transcribe it back, and the LLM answer a prompt.
  It reports timings and VRAM, and tells you exactly which part failed.
- `service` installs a systemd unit so it starts with the container (`journalctl -fu spectre`).
- **Chatterbox-Turbo voice** (what Boolian uses; setup starts with Kokoro). It sounds far more natural, clones a
  10 s+ voice clip and performs `[laugh]`, `[chuckle]`, `[sigh]`, `[gasp]` and `[whispering]`, which the LLM is then
  told it may use. `add-chatterbox` installs it after backing up the working environment, `selftest --compare` times
  it against Kokoro and saves WAVs to `/tmp`, and `undo-chatterbox` restores the backup. To use it, set in
  `spectre.env`: `SPECTRE_TTS_BACKEND=chatterbox`, `SPECTRE_TTS_VOICE_REF=/path/voice.wav` and `SPECTRE_TTS_STREAM=1`.
  Our faster decoding (`SPECTRE_TTS_FAST`, on by default) uses CUDA graphs at full precision;
  streaming sends each sentence in pieces as it's generated, so he starts speaking about 1.3 s after a visitor stops.

Measured on the RTX 4060 Ti 16 GB (median): Whisper 176 ms, first LLM word 247 ms, first real speech 1.28 s, with a
reaction noise heard at about 0.4 s.

On the laptop, run the renderer and open `http://localhost:5173/?server=ws://SPECTRE_CONTAINER_IP:8765/ws`.

## VM setup with Docker (CUDA, 16 GB)

The older, simpler route: Kokoro's voice and a small LLM. It hasn't been tested with Chatterbox or Gemma 4 12B; the
containers above are what Boolian runs on.

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

With Kokoro, pick the voice with `SPECTRE_TTS_VOICE` (such as `bm_fable`, `bm_george`, `bm_lewis`, `am_puck`).

The browser adds reverb and a slight chorus on top, which you can adjust with the Reverb slider. Under it, four
more ghostly effects are off by default: *Waver* (a slow, drifting pitch wobble), *Flanger* (a whooshing sweep),
*Echo* (darkening repeats) and *Shadow voice* (a quiet second voice an octave below). Try them on the real speaker;
each one costs a little clarity.

**Conversation log.** The server keeps a text log of every conversation: what visitors said (including what it
ignored, and why) and what Boolian said back, with his emotion, move and response time. No audio is ever kept. It
goes to `conversations/YYYY-MM-DD.jsonl` at the top of the repo on the spectre container (gitignored);
`SPECTRE_CONVERSATION_LOG` sets another directory, or `off`. Read it with
`server/.venv/bin/python scripts/show_conversations.py [2026-10-31 | --all] [--summary]`: transcripts, then a
summary of the night (groups, lines heard and ignored and why, how long replies took, timeouts and fallbacks,
resets, moves, rude noises, busiest hours). It will hold children's first names
and chatter, so keep it on the container and delete it when you're done with it.

All settings are `SPECTRE_*` environment variables; see `server/app/config.py`.
`SPECTRE_AUTH_TOKEN` enables a shared token: append `?token=…` to the page URL.

## The ESP32 microphone and motion sensor

An ESP32-C5 with an INMP441 microphone and an AM312 PIR, on Wi-Fi, so the mic can sit near the visitors instead of
the laptop. It streams audio to the server's `/mic` socket, and a person walking up makes him greet them. The
firmware is in `firmware/` (ESP-IDF 6.1), and a 3D-printable case is in `firmware/enclosure/`.

### Wiring

| INMP441 | ESP32-C5 | | AM312 | ESP32-C5 |
|---|---|---|---|---|
| VDD | 3V3 | | VCC (+) | 3V3 |
| GND | GND | | GND (−) | GND |
| SCK | GPIO4 | | OUT | GPIO3 |
| WS | GPIO5 | | | |
| SD | GPIO6 | | | |
| L/R | GND | | | |

- Both run from 3.3 V; don't power the INMP441 from 5 V.
- Female-to-female jumpers straight onto the pins are enough; tape or a dab of hot glue stops them working loose.
- Link the mic's L/R to its own GND pin, which saves a wire. Use the board's spare 3V3 and GND pins before splicing;
  if you must split one, use a Wago lever connector or solder and heat-shrink, not a strip of breadboard.
- Keep the mic's five wires together in one loose bundle, under about 30 cm. Its clock runs at 1 MHz, and a spread-out
  loop of wire picks up the Wi-Fi radio. The PIR's wire can be a metre or two and go its own way.

### Setting it up

1. Install [ESP-IDF 6.1](https://docs.espressif.com/projects/esp-idf/en/latest/esp32c5/get-started/).
2. `cp firmware/main/secrets.example.h firmware/main/secrets.h` and fill in the Wi-Fi name and password, the server
   (`ws://SPECTRE_IP:8765/mic`, plus `?token=…` if `SPECTRE_AUTH_TOKEN` is set) and a name for the operator panel.
   **`secrets.h` is gitignored; never commit it.** The build stops if it's missing.
3. Build and flash. Either of the board's USB ports works: `/dev/cu.usbmodem…` is the C5's own USB,
   `/dev/cu.wchusbserial…` the serial chip.
   ```bash
   cd firmware
   source ~/.espressif/tools/activate_idf_v6.1.sh
   idf.py -p /dev/cu.usbmodem… build flash monitor
   ```
   If it ever won't join Wi-Fi after changing networks, `idf.py -p … erase-flash flash` starts it fresh.

**Mesh Wi-Fi.** Out of the box, ESP-IDF often fails to rejoin a mesh network after a reboot. The firmware uses the
settings from [Willow](https://github.com/HeyWillow/willow): 802.11k/v and MBO so the nodes can steer it, three tries
before the next node, and no Wi-Fi state kept in flash (`sdkconfig.defaults`, `wifi_start()` in `main.c`). The log
names the access point it joins and the reason when it drops.

### Checking it

The serial monitor prints a line a second: level and peak in dBFS, clipping, frames sent and dropped, queue, Wi-Fi
signal, reconnects, free memory, and the PIR state with its raw flips that second.

- **Microphone.** Speech close to it should reach about −30 dBFS. If it barely moves, raise `MIC_GAIN` in `main.c`;
  if it clips, lower it. To hear what it hears:
  `server/.venv/bin/python scripts/listen_mic.py ws://SPECTRE_IP:8765/ws --seconds 10 && afplay esp32.wav`.
  A faint hiss is the INMP441's own noise floor (about −60 dBFS), well below the room; Whisper and Silero ignore it.
- **Listening live.** `scripts/monitor_mic.py ws://SPECTRE_IP:8765/ws` plays what it hears through the Mac (use
  headphones, or it howls), with the hum, speech and hiss levels twice a second. It needs `pip install sounddevice`.
- **Power it from a charger or power bank, into the left (UART) USB-C socket.** From the Mac's USB, or the C5's own
  socket, it picked up a low hum about 12 dB louder, which buried quiet speech and wrecked Whisper's accuracy.
- **PIR.** Cover it with a mug (glass and plastic block body heat): it should say "motion stopped" and stay quiet.
  Uncover it and wave: "motion seen", held about 2 s after you stop. Give it a minute after power-on to settle,
  and expect it to keep firing while you sit in front of it. A stream of raw flips means a loose or noisy OUT wire;
  the firmware ignores anything shorter than 300 ms.

### The case

`firmware/enclosure/enclosure.scad` is a parametric OpenSCAD box, about 79 × 64 × 44 mm, with a push-on lid. The PIR
pokes through the front above a small sound hole, the USB cable comes out of the side, and there are screw tabs on the
back. The board sits on ledges at each end so the jumpers can hang beneath it. **Measure your boards first**: the
numbers marked `MEASURE` are typical sizes, not yours. Then:

```bash
cd firmware/enclosure
openscad -D 'part="box"' -o box.stl enclosure.scad && openscad -D 'part="lid"' -o lid.stl enclosure.scad
```

Print the box open side up and the lid flat side down, with no supports; PETG if it'll be outside. Glue the mic into
the ring behind the sound hole, sound port facing out and unobstructed, and hot-glue the PIR in from inside.
Place it 0.5–1 m from where visitors stand, as far from the speaker as you can, with the PIR looking down the path.

### Using it

In the operator panel, choose **ESP32 microphone (via the server)** under Microphone. The info line shows its name,
Wi-Fi signal and dropped frames, and the log says when it connects, disconnects or sees motion. Push-to-talk and
auto-listen work just as with the laptop's mic. If the server hears nothing from it for 5 s
(`SPECTRE_MIC_IDLE_S`), it hangs up and shows it as disconnected, so a link that died without closing can't pass
for a working one; the ESP32 reconnects by itself, and gives up on a dead link after 15 s without a pong.

**Motion:** when the PIR sees someone and nobody is talking to him, he greets them, at most once a minute
(`SPECTRE_MOTION_GREET_S`, 0 = never). Switching attract mode off (**A**) stops it too. The attract call-outs carry
on as before.

## On the night: runbook

**Setup**
1. Make sure both containers are up (their services start at boot) and `curl SPECTRE_IP:8765/health/ready` is OK.
2. On the laptop: `cd renderer && npm run dev`. The page must be served from **localhost**, because browsers
   only allow the microphone on secure origins.
3. Make the projector an extended display, and choose the output speaker in the system's sound settings (wired if
   possible).
4. Open `http://localhost:5173/?server=ws://SPECTRE_IP:8765/ws` (the server is remembered afterwards; add
   `&renderer=3d` for the 3D ghost). Drag the window to the projector, **click wake**, then press **F** for fullscreen.
5. Press **O** and click **Open remote control window**. Drag the new window to the laptop screen. You operate
   from there; the projector shows only the ghost. Press **O** in the stage window again to hide its panel.
6. Press **K** for the test pattern, then line up scale and position with the arrow keys and +/-. Press **K** again to exit.
   To crop the picture to a window, doorway or screen, use the **Mask** sliders under Projection: they black out
   each edge of the projector and stay put when you rescale or move the ghost.
7. If you use Bluetooth speakers, raise **Speaker delay** until the mouth matches the sound. Set Reverb and the
   voice effects (Waver, Flanger, Echo, Shadow voice) on the real speaker; each costs a little clarity.
8. Plug in the ESP32 and choose **ESP32 microphone (via the server)** under Microphone; the log should say it's
   connected. Walk up to it: "motion seen" appears in the log.
9. Hold Space and say "hello". Check the log and latency readout.

**Running**
- **Hold Space** while a child talks, then release. This is the most reliable mode. Auto-listen (**V**) uses the
  Silero voice model ([ricky0123/vad](https://github.com/ricky0123/vad)), so bangs, rustling and music don't set it
  off, but a group of chattering trick-or-treaters still counts as talking. In the operator panel, *Voice
  threshold* sets how sure it must be that it's hearing speech (lower is more sensitive), and *Pause before reply*
  sets how long it waits after they stop. *Detect speech by* switches back to the old loudness-only detector.
- **Attract mode** (on by default, **A** toggles it): while nobody is talking to him, he calls out every 30–60 s to
  draw people over ("I've got riddles for anyone brave enough!", the odd fart). If someone answers, he remembers what
  he called out. `SPECTRE_ATTRACT_S` sets how often (0 = never).
- He knows 22 Halloween riddles and 8 jokes and knock-knocks (`RIDDLES` and `JOKES` in `persona.py`), and a guess-the-creature game, and offers them now and then. Each group of visitors gets a random handful of them, so he doesn't keep telling the same one.
- **G** greets the group, which helps because kids don't know they can talk to him. **R** forgets the visitor.
  If nobody answers him, he asks if they're still there twice (12 s apart), then says goodbye and forgets them.
- **1–9** play puppet lines (the raspberry and fart lines below them are click-only), and the "Boolian says" box
  speaks any typed text in his voice; `[fart]` or `[raspberry]` in it works too. Both still work if the server is
  down, using the browser's own voice.
- **X** / **Esc** stops him talking. **M** mutes.
- **If a red alert appears in the operator panel**, a child said something like "I'm lost", "I can't find my mum",
  "someone hit me" or "I want to hurt myself". He has calmly told them to stay in the light and shout for their
  grown-up, and that the humans in the house are coming, so go and help. Other alerts say a child talked about
  running off from their grown-up ("I'm going to run away to scare my mum") or something else unsafe (going with a
  stranger, playing with matches): he has told them no, so have a word with their grown-up. Click to dismiss.
- Some things never reach the AI, because it has answered them badly in testing: being lost or hurt, hurting
  themselves, running off or hiding from their grown-ups (or pretending to be lost to scare them), going out alone,
  strangers, roads, fire, keeping secrets from grown-ups, asking to come in, visit or meet, asking to be his best
  friend, and asking how to become a ghost (it once said "just trip over a big pumpkin!"). They all get fixed, safe
  answers. The prompt also forbids going along with anything dangerous, even as a prank. Invitations, secret-keeping
  or encouraging danger that still slip into a reply are swapped out before they're spoken.
- If he starts answering himself: turn auto-listen off and use Space.
- If the server dies: he falls asleep (Zzz), puppet lines still work, and he wakes when the server returns.

**Shutdown**: close the browser windows. The containers can stay up, or `systemctl stop spectre` / `spectre-llm`.

## Controls

| Key | Action | Key | Action |
|---|---|---|---|
| Space (hold) | listen | V | auto-listen on/off |
| G | greet | R | forget visitor |
| X / Esc | stop talking | M | mute |
| 1–9 | puppet lines | O | operator panel |
| F | fullscreen | K | projection test pattern |
| A | attract mode on/off | | |
| arrows | move ghost | + / - | scale |

Debug and art mode: `?state=speaking&emotion=mischievous&mouth=0.6` freezes a pose without a server, and
`&rude=fart` or `&rude=raspberry` repeats that animation. `?seed=N` makes idle behaviour repeatable.

## Layout

```
server/          FastAPI + asyncio orchestration (session.py is the heart; persona.py is his personality), pytest suite
server/assets/   the fart and raspberry recordings (credits in noises/CREDITS.md)
firmware/        ESP-IDF project for the ESP32-C5: microphone and PIR
renderer/        Vite + TypeScript, Canvas 2D ghost, or three.js with ?renderer=3d
scripts/         llm-container.sh and spectre-container.sh (setup and services), smoke_test.py (end-to-end turns and
                 latency), listen_mic.py (record the ESP32 to a WAV), selftest_gpu.py, bench_chatterbox.py, make_noises.py / import_noises.py
infra/           docker compose for the older Docker route
docs/            protocol-v1.md
```

## Deliberate deviations from the plan

- **No Pipecat.** In half-duplex mode, a hand-written turn loop (`server/app/session.py`) is simpler and fully under our control.
  Barge-in is supported by the operator (Space while he talks); automatic barge-in needs echo cancellation.
- **Canvas 2D instead of PixiJS**: one procedural character doesn't need a scene graph. It runs at 60 fps with zero
  dependencies. The optional three.js renderer reuses the same character logic and face.
- **Chatterbox-Turbo instead of Kokoro, Gemma 4 12B instead of a 4B model, Whisper large-v3-turbo instead of
  base/small**: the 16 GB card fits all three, and the latency budget allowed the bigger, better-sounding models.
- **Silero VAD in the browser**, not on the server, because the browser is where the mic is for now.
- **Two Proxmox containers with systemd** instead of Docker Compose on a VM.
- **The ESP32's audio goes via the browser.** The server relays it to the browser, which runs Silero on it and sends it
  back while listening, exactly as with its own mic. Push-to-talk, pre-roll and the VAD settings all work unchanged.
- **Emotion via an inline `[emotion:action]` tag**, validated and stripped server-side, with a fallback to neutral.

## Licence

MIT. See [LICENSE](LICENSE).
