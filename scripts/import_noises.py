"""Cut Boolian's rude noises out of the downloaded CC0 recordings into server/assets/noises/.

    python scripts/import_noises.py DOWNLOAD_DIR

DOWNLOAD_DIR holds the original files, renamed as below (raspberries.wav: the m4a converted with
`afconvert -f WAVE -d LEI16 -c 1 raspberries.m4a raspberries.wav`). Sources and licences are in
server/assets/noises/CREDITS.md. Clips are written mono at their original rate; app.noises
resamples and levels them at load time.
"""

import sys
import wave
from pathlib import Path

import numpy as np

# (output name, source file, start s, length s)
CLIPS = [
    ("raspberry-1", "raspberries.wav", 0.87, 0.46),
    ("raspberry-2", "raspberries.wav", 2.21, 0.97),
    ("raspberry-3", "raspberries.wav", 4.11, 0.85),
    ("raspberry-4", "raspberries.wav", 5.88, 0.93),
    ("raspberry-5", "raspberries.wav", 7.70, 2.23),
    ("raspberry-6", "raspberries.wav", 10.54, 0.56),
    ("raspberry-7", "raspberries.wav", 12.15, 0.74),
    ("raspberry-8", "raspberries.wav", 13.71, 0.36),
    ("raspberry-9", "raspberries.wav", 14.99, 1.31),
    ("fart-1", "fart-0111.wav", 0.0, 1.48),
    ("fart-2", "fart-0240.wav", 0.0, 0.34),
    ("fart-3", "fart-0241.wav", 0.0, 0.24),
    ("fart-4", "fart-1031.wav", 0.15, 0.95),
]


def read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        ch, sr = w.getnchannels(), w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, ch).mean(axis=1) / 32768
    return x, sr


def main(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, file, start, length in CLIPS:
        x, sr = read(src / file)
        clip = x[int(start * sr) : int((start + length) * sr)].copy()
        fade_in, fade_out = int(sr * 0.008), int(sr * 0.04)
        clip[:fade_in] *= np.linspace(0, 1, fade_in)
        clip[-fade_out:] *= np.linspace(1, 0, fade_out)
        with wave.open(str(out / f"{name}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.clip(clip, -1, 1) * 32767).astype("<i2").tobytes())
        print(f"{name}.wav  {len(clip) / sr:.2f}s  from {file} @ {start}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(__file__).resolve().parent.parent / "server" / "assets" / "noises")
