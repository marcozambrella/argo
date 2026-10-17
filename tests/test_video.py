import subprocess

import pytest

from sicurezza.video import leggi_frame, leggi_info_video


@pytest.fixture(scope="module")
def clip_a_30_fps(tmp_path_factory):
    percorso = tmp_path_factory.mktemp("video") / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=4",
                    "-pix_fmt", "yuv420p", str(percorso)], check=True)
    return percorso


def test_conversione_a_25_fps(clip_a_30_fps):
    frame = list(leggi_frame(clip_a_30_fps))
    assert len(frame) == 100  # 4 s × 25 fps
    assert frame[0].shape == (120, 160, 3)


def test_lettura_di_un_segmento(clip_a_30_fps):
    assert len(list(leggi_frame(clip_a_30_fps, inizio_secondi=1.0, fine_secondi=3.0))) == 50


def test_info_video(clip_a_30_fps):
    info = leggi_info_video(clip_a_30_fps)
    assert (info.larghezza, info.altezza) == (160, 120)
    assert info.durata_secondi == pytest.approx(4.0, abs=0.1)
