"""Binary framing shared with the renderer and (later) the ESP32. See docs/protocol-v1.md."""

from __future__ import annotations

import struct
from dataclasses import dataclass

PROTOCOL_VERSION = 1

# Microphone -> server. Big-endian header, little-endian PCM payload.
MIC_MAGIC = b"GHST"
MIC_HEADER = struct.Struct(">4sBBHII")  # magic, version, flags, payload_len, seq, capture_ms
MIC_SAMPLE_RATE = 16_000

# Server -> renderer audio.
OUT_MAGIC = b"GHAU"
OUT_HEADER = struct.Struct(">4sBBHIII")  # magic, version, flags, reserved, generation, seq, sample_rate
OUT_FLAG_LAST = 0x01
# This chunk is a rude noise, not speech, so the presenter can play its animation in time with it.
OUT_NOISE_FLAGS = {"raspberry": 0x02, "fart": 0x04}


class FrameError(ValueError):
    pass


@dataclass(frozen=True)
class MicFrame:
    seq: int
    capture_ms: int
    pcm: bytes


def parse_mic_frame(data: bytes) -> MicFrame:
    if len(data) < MIC_HEADER.size:
        raise FrameError(f"frame too short: {len(data)} bytes")
    magic, version, _flags, length, seq, capture_ms = MIC_HEADER.unpack_from(data)
    if magic != MIC_MAGIC:
        raise FrameError(f"bad magic {magic!r}")
    if version != PROTOCOL_VERSION:
        raise FrameError(f"unsupported version {version}")
    pcm = data[MIC_HEADER.size :]
    if len(pcm) != length:
        raise FrameError(f"payload length {len(pcm)} != declared {length}")
    if length % 2:
        raise FrameError("odd PCM payload length")
    return MicFrame(seq=seq, capture_ms=capture_ms, pcm=pcm)


def pack_mic_frame(seq: int, capture_ms: int, pcm: bytes) -> bytes:
    return MIC_HEADER.pack(MIC_MAGIC, PROTOCOL_VERSION, 0, len(pcm), seq & 0xFFFFFFFF, capture_ms & 0xFFFFFFFF) + pcm


def pack_audio_chunk(generation: int, seq: int, sample_rate: int, pcm: bytes, last: bool = False, noise: str | None = None) -> bytes:
    flags = (OUT_FLAG_LAST if last else 0) | OUT_NOISE_FLAGS.get(noise or "", 0)
    return OUT_HEADER.pack(OUT_MAGIC, PROTOCOL_VERSION, flags, 0, generation, seq, sample_rate) + pcm
