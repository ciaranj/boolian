# Spectre protocol v1

One WebSocket per presenter (browser), `ws://SERVER:8765/ws[?token=…]`.
Text frames are JSON control messages; binary frames are audio. Every JSON message carries `"v": 1`.

## Generations

Every response the ghost gives (LLM reply, puppet line, greeting, "pardon?") gets a new, increasing
`generation_id`. The server never sends anything for a generation once it has been cancelled or superseded, and the
browser rejects audio chunks for any generation `<=` the last one it cancelled. Late audio from a cancelled reply
therefore cannot play.

## Browser → server (JSON)

| type | fields | meaning |
|---|---|---|
| `hello` | `role` | sent on connect; server answers `welcome` |
| `listen_start` | | visitor started talking (PTT down or VAD start). Cancels any reply in progress. |
| `listen_end` | | commit the utterance: STT → LLM → TTS |
| `listen_abort` | | discard captured audio (too short, operator cancel) |
| `text` | `text` | typed visitor input (operator panel); skips STT |
| `say` | `text`, `emotion`, `action` | puppet line: speak exactly this, no LLM |
| `greet` | `source?` | random canned greeting. With `source: "motion"` it's the browser asking to greet someone the PIR saw: the server only does it with nobody talking to him, attract mode on, and at most once a minute |
| `reset` | | cancel and forget the conversation history |
| `cancel` | | stop the current reply |
| `playback` | `generation_id`, `event`: `started`/`ended`/`cancelled` | what the speakers are *actually* doing |
| `mic_source` | `source`: `browser` / `device` | `device` asks the server to relay the ESP32's microphone frames to this browser |
| `ping` | `t` | server replies `pong` |

## Browser → server (binary): microphone frames

16-byte big-endian header, then little-endian PCM. The ESP32 should use the same layout.

| bytes | type | field |
|---|---|---|
| 0–3 | ASCII | `GHST` |
| 4 | u8 | version = 1 |
| 5 | u8 | flags = 0 |
| 6–7 | u16 | payload length in bytes (640 for 20 ms) |
| 8–11 | u32 | sequence number (wraps; reset per connection) |
| 12–15 | u32 | capture time, ms (device clock, wraps) |
| 16– | s16le | 16 kHz mono PCM |

Frames are only buffered between `listen_start` and `listen_end`; others just update sequence-gap stats.
Malformed frames are dropped and logged.

## ESP32 → server: `ws://SERVER:8765/mic[?token=…]`

The microphone device has its own socket. It sends the same `GHST` binary frames, plus JSON:

| type | fields | meaning |
|---|---|---|
| `hello` | `role: "mic"`, `device` (a name for the operator panel), `sample_rate` | on every connect; the sequence number starts again |
| `motion` | `on` | the PIR changed |
| `stats` | `rssi`, `db`, `peak_db`, `clips`, `dropped`, `reconnects`, `heap`, `uptime_s` | every 5 s, for the operator |

The server relays the frames, unchanged, to every presenter that sent `mic_source: device`. The browser treats them as
its own microphone: it runs voice detection on them and sends them back on its own socket while it's listening. Only
the latest device to connect is listened to.

## Server → browser (JSON)

| type | fields |
|---|---|
| `welcome` | `ghost_name`, `backends`, `input_format` |
| `state` | `state`: `thinking` / `idle` / `error` |
| `transcript` | `text`, `rejected` (null, or `too_short`/`empty`/`phantom`/`no_speech`/`low_confidence`), `typed?` |
| `expression` | `emotion` (neutral, happy, curious, surprised, mischievous, confused), `action` (none, bounce, tilt, spin, hide, peek), `intensity`, `source` (`reply` from the LLM, or `attract`, `nudge`, `greeting`, `puppet`, `giving_up`… for canned lines) |
| `speech_start` | `audio_format`: `{codec: "pcm_s16le", sample_rate, channels: 1}` |
| `speech_end` | `chunks` |
| `reply` | `text` (what was actually spoken) |
| `cancel` | `generation_id`, `reason` |
| `session_reset` | `reason`: `operator` / `idle` |
| `mic_device` | `connected`, and while connected `name`, `gaps`, `bad`, plus the device's latest `stats` fields |
| `motion` | `on`: the ESP32's PIR. The browser answers with `greet` (`source: "motion"`) once he's idle and it hasn't heard anyone talking for 2 s, so he doesn't talk over someone who walked up already speaking. It keeps trying while the PIR is on and for 8 s after |
| `motion_greet` | the answer to `greet` with `source: "motion"`: `greeted`, and if not, `reason` (in words, for the operator's log) and `retry` (true if he's only busy for a moment, so the browser can ask again) |

Binary frames starting `GHST` are the ESP32's microphone, relayed (only after `mic_source: device`).

`expression` is always sent before the first audio of a reply, so the face changes as the ghost starts to speak.

## Server → browser (binary): audio chunks

20-byte big-endian header, then s16le mono PCM at `sample_rate`.

| bytes | type | field |
|---|---|---|
| 0–3 | ASCII | `GHAU` |
| 4 | u8 | version = 1 |
| 5 | u8 | flags: bit 0 = last chunk (currently unused); bit 1 = a raspberry, bit 2 = a fart. A rude noise is sent as one chunk, so the presenter can animate as it plays. |
| 6–7 | u16 | reserved |
| 8–11 | u32 | generation id |
| 12–15 | u32 | chunk sequence within the generation |
| 16–19 | u32 | sample rate |

## LLM output convention

The model is prompted to begin each reply with `[emotion]` or `[emotion:action]`. The server strips the tag before
TTS, validates it against the allowlist (unknown values become `neutral`/`none`), and defaults to `neutral` if the tag
is missing. Speech never depends on the tag being well-formed.
