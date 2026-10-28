"""Runtime configuration, read from SPECTRE_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _conversation_log() -> str:
    """SPECTRE_CONVERSATION_LOG: a directory, or "off". Defaults to conversations/ at the top of the repo."""
    value = _env("CONVERSATION_LOG", str(Path(__file__).resolve().parents[2] / "conversations"))
    return "" if value.lower() in ("", "off", "0", "none") else value


def _env(name: str, default: str) -> str:
    return os.environ.get(f"SPECTRE_{name}", default)


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    auth_token: str
    conversation_log: str  # directory for the text log of every conversation; "" = don't keep one

    # "mock" swaps every backend for a fake so the whole loop runs on a laptop.
    stt_backend: str
    stt_model: str
    stt_device: str
    stt_compute_type: str
    stt_language: str

    llm_backend: str
    llm_url: str
    llm_model: str
    llm_max_tokens: int
    llm_temperature: float
    llm_first_sentence_timeout_s: float
    llm_total_timeout_s: float

    tts_backend: str
    tts_url: str
    tts_voice: str
    tts_speed: float
    tts_device: str
    tts_voice_ref: str
    tts_fast: bool
    tts_stream: bool
    tts_sample_rate: int

    ghost_name: str
    history_turns: int
    idle_reset_s: float
    nudge_s: float
    max_nudges: int
    attract_s: float
    motion_greet_s: float
    mic_idle_s: float
    max_utterance_s: float
    static_dir: str


def load_config() -> Config:
    mock = _env("MOCK", "0") == "1"
    return Config(
        host=_env("HOST", "0.0.0.0"),
        port=int(_env("PORT", "8765")),
        auth_token=_env("AUTH_TOKEN", ""),
        conversation_log=_conversation_log(),
        stt_backend="mock" if mock else _env("STT_BACKEND", "faster-whisper"),
        stt_model=_env("STT_MODEL", "large-v3-turbo"),
        stt_device=_env("STT_DEVICE", "cuda"),
        stt_compute_type=_env("STT_COMPUTE_TYPE", "float16"),
        stt_language=_env("STT_LANGUAGE", "en"),
        llm_backend="mock" if mock else _env("LLM_BACKEND", "openai"),
        llm_url=_env("LLM_URL", "http://127.0.0.1:8080/v1").rstrip("/"),
        llm_model=_env("LLM_MODEL", "local"),
        llm_max_tokens=int(_env("LLM_MAX_TOKENS", "110")),
        llm_temperature=float(_env("LLM_TEMPERATURE", "0.8")),
        llm_first_sentence_timeout_s=float(_env("LLM_FIRST_SENTENCE_TIMEOUT_S", "8")),
        llm_total_timeout_s=float(_env("LLM_TOTAL_TIMEOUT_S", "20")),
        tts_backend="mock" if mock else _env("TTS_BACKEND", "kokoro"),
        tts_url=_env("TTS_URL", "http://127.0.0.1:8880/v1").rstrip("/"),
        tts_voice=_env("TTS_VOICE", "bm_fable"),
        tts_speed=float(_env("TTS_SPEED", "1.0")),
        tts_device=_env("TTS_DEVICE", "auto"),  # in-process voices only: auto | cuda | cpu
        tts_voice_ref=_env("TTS_VOICE_REF", ""),  # chatterbox only: a 10 s+ WAV to clone; "" = built-in voice
        tts_fast=_env("TTS_FAST", "1") == "1",  # chatterbox only: our faster decoding (0 = the original)
        tts_stream=_env("TTS_STREAM", "0") == "1",  # chatterbox only, needs tts_fast: send sentences in pieces
        tts_sample_rate=int(_env("TTS_SAMPLE_RATE", "24000")),
        ghost_name=_env("GHOST_NAME", "Boolian"),
        history_turns=int(_env("HISTORY_TURNS", "25")),  # ~70 tokens each; with the ~1.6k-token prompt, must fit the 4k context
        idle_reset_s=float(_env("IDLE_RESET_S", "40")),  # only used with nudging off
        nudge_s=float(_env("NUDGE_S", "12")),  # quiet seconds after he speaks before he prompts again; 0 = never
        max_nudges=int(_env("MAX_NUDGES", "2")),
        attract_s=float(_env("ATTRACT_S", "45")),  # with nobody to talk to, call out about this often (±40%); 0 = never  # "still there?" prompts before he gives up and forgets them
        motion_greet_s=float(_env("MOTION_GREET_S", "60")),  # PIR: greet newcomers at most this often; 0 = never
        mic_idle_s=float(_env("MIC_IDLE_S", "5")),  # ESP32: drop its connection after this long with nothing from it
        max_utterance_s=float(_env("MAX_UTTERANCE_S", "15")),
        static_dir=_env("STATIC_DIR", ""),
    )
