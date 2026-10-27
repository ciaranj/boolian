"""Write every variant of Boolian's rude noises to WAV files, to listen to them:  python scripts/make_noises.py OUT_DIR"""

import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
from app import noises  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "noises")
out.mkdir(parents=True, exist_ok=True)
sr = 24_000
for kind in noises.KINDS:
    noises.pcm(kind, sr)  # fills the cache
    for i, y in enumerate(noises._cache[(kind, sr)]):
        path = out / f"{kind}-{i + 1}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((y.clip(-1, 1) * 32767).astype("<i2").tobytes())
        print(f"{path}  {len(y) / sr:.2f}s")
