import json
from pathlib import Path

import pytest
import torch

from sicurezza.inferenza import analizza_video
from sicurezza.modelli import crea_modello
from sicurezza.video import leggi_info_video

RADICE = Path(__file__).parent.parent
PESI_POSE = RADICE / "runs" / "yolo26n-pose.pt"
VIDEO_PERSONE = Path(__file__).parent / "dati" / "persone_2s.mp4"


@pytest.mark.skipif(not PESI_POSE.exists(), reason="pesi yolo26n-pose non scaricati in runs/")
def test_demo_su_video_vero_con_classificatore_non_addestrato(tmp_path):
    iperparametri = {"dimensione": 32, "blocchi": 1}
    torch.save({"nome_modello": "graph_transformer", "iperparametri_modello": iperparametri,
                "stato": crea_modello("graph_transformer", **iperparametri).state_dict()}, tmp_path / "c.pt")
    eventi = analizza_video(str(VIDEO_PERSONE), str(tmp_path / "c.pt"), tmp_path / "demo",
                            percorso_modello_pose=str(PESI_POSE))
    assert len(eventi) == 2                                  # 50 frame → finestre 0-39 e 10-49
    assert eventi[0]["inizio_secondi"] == 0 and eventi[1]["fine_secondi"] == 2.0
    assert abs(sum(eventi[0]["probabilita"].values()) - 1) < 1e-3
    info = leggi_info_video(str(tmp_path / "demo" / "video_annotato.mp4"))
    assert info.durata_secondi == pytest.approx(2.0, abs=0.1)
    assert json.loads((tmp_path / "demo" / "eventi.json").read_text()) == eventi
    assert (tmp_path / "demo" / "probabilita.csv").read_text().count("\n") == 3
