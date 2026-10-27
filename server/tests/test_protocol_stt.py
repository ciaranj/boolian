import pytest

from app.protocol import FrameError, pack_mic_frame, parse_mic_frame
from app.stt import judge


def test_mic_frame_roundtrip():
    pcm = bytes(range(256)) * 2 + bytes(128)
    frame = parse_mic_frame(pack_mic_frame(7, 1234, pcm))
    assert (frame.seq, frame.capture_ms, frame.pcm) == (7, 1234, pcm)


@pytest.mark.parametrize("data", [b"GHS", b"XXXX" + bytes(12) + bytes(640), pack_mic_frame(1, 1, bytes(640))[:-2]])
def test_mic_frame_rejects_garbage(data):
    with pytest.raises(FrameError):
        parse_mic_frame(data)


@pytest.mark.parametrize(
    "text,dur,lp,ns,expected",
    [
        ("Thanks for watching!", 2.0, -0.3, 0.1, "phantom"),
        ("you", 1.0, -0.3, 0.1, "phantom"),
        ("Hello ghost", 0.2, -0.3, 0.1, "too_short"),
        ("Hello ghost", 1.0, -1.0, 0.9, "no_speech"),
        ("blorp flarg", 1.0, -1.5, 0.1, "low_confidence"),
        ("What's your name?", 1.2, -0.3, 0.05, None),
    ],
)
def test_judge(text, dur, lp, ns, expected):
    assert judge(text, dur, lp, ns) == expected
