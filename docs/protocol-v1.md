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
| `greet` | | random canned greeting |
| `reset` | | cancel and forget the conversation history |
| `cancel` | | stop the current reply |
| `playback` | `generation_id`, `event`: `started`/`ended`/`cancelled` | what the speakers are *actually* doing |
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

## Server → browser (JSON)

| type | fields |
|---|---|
| `welcome` | `ghost_name`, `backends`, `input_format` |
| `state` | `state`: `thinking` / `idle` / `error` |
| `transcript` | `text`, `rejected` (null, or `too_short`/`empty`/`phantom`/`no_speech`/`low_confidence`), `typed?` |
| `expression` | `emotion` (neutral, happy, curious, surprised, mischievous, confused), `action` (none, bounce, tilt, spin, hide, peek), `intensity` |
| `speech_start` | `audio_format`: `{codec: "pcm_s16le", sample_rate, channels: 1}` |
| `speech_end` | `chunks` |
| `reply` | `text` (what was actually spoken) |
| `cancel` | `generation_id`, `reason` |
| `session_reset` | `reason`: `operator` / `idle` |

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
