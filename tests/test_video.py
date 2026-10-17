import subprocess

import pytest

from sicurezza.video import iter_frames, probe


@pytest.fixture(scope="module")
def clip_30fps(tmp_path_factory):
    path = tmp_path_factory.mktemp("v") / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=4",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_conversione_a_25_fps(clip_30fps):
    frames = list(iter_frames(clip_30fps))
    assert len(frames) == 100  # 4 s × 25 fps
    assert frames[0].shape == (120, 160, 3)


def test_segmento(clip_30fps):
    assert len(list(iter_frames(clip_30fps, start=1.0, end=3.0))) == 50


def test_probe(clip_30fps):
    info = probe(clip_30fps)
    assert (info.width, info.height) == (160, 120)
    assert info.duration == pytest.approx(4.0, abs=0.1)
