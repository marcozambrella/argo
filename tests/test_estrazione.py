from pathlib import Path

import numpy as np
import pytest

from sicurezza.estrazione import Estrattore, VideoDaEstrarre, estrai_elenco, leggi_elenco, scrivi_elenco

RADICE = Path(__file__).parent.parent
PESI_POSE = RADICE / "runs" / "yolo26n-pose.pt"
VIDEO_PERSONE = Path(__file__).parent / "dati" / "persone_2s.mp4"


def test_elenco_andata_e_ritorno(tmp_path):
    video = [VideoDaEstrarre("ucf/Robbery001_x264@10.0-20.0", "/x/Robbery001_x264.mp4", 10.0, 20.0)]
    scrivi_elenco(video, tmp_path / "elenco.jsonl")
    assert leggi_elenco(tmp_path / "elenco.jsonl") == video


@pytest.mark.skipif(not PESI_POSE.exists(), reason="pesi yolo26n-pose non scaricati in runs/")
def test_estrazione_reale_su_cpu_e_ripresa(tmp_path):
    estrattore = Estrattore(str(PESI_POSE), dispositivo="cpu")
    elenco = [VideoDaEstrarre("prova/persone", str(VIDEO_PERSONE))]
    estrai_elenco(elenco, tmp_path, estrattore)

    dati = np.load(tmp_path / "prova__persone.npz")
    assert int(dati["numero_frame"]) == 50                      # 2 s a 25 fps
    assert len(dati["persone_frame"]) == len(dati["persone_keypoint"]) > 0
    assert dati["persone_keypoint"].shape[1:] == (17, 3)
    assert len(np.unique(dati["persone_id_traccia"])) <= 8      # gli ID restano stabili, non uno nuovo per frame
    assert len(dati["armi_frame"]) == 0                          # nessun detector armi passato

    # seconda esecuzione: il file c'è già, non viene rifatto
    modificato_prima = (tmp_path / "prova__persone.npz").stat().st_mtime
    estrai_elenco(elenco, tmp_path, estrattore)
    assert (tmp_path / "prova__persone.npz").stat().st_mtime == modificato_prima


@pytest.mark.skipif(not PESI_POSE.exists() or not (RADICE / "runs" / "yolo26s.pt").exists(),
                    reason="pesi non scaricati in runs/")
def test_secondo_passaggio_armi_stessi_frame(tmp_path):
    """Con un detector qualsiasi (qui YOLO26s generico al posto di quello armi) i frame tornano allineati."""
    from ultralytics import YOLO

    from sicurezza.aggiungi_armi import armi_di_un_video
    video = VideoDaEstrarre("prova/persone", str(VIDEO_PERSONE))
    estrai_elenco([video], tmp_path, Estrattore(str(PESI_POSE), dispositivo="cpu"))
    armi = armi_di_un_video(YOLO(str(RADICE / "runs" / "yolo26s.pt")), video, "cpu")
    dati = np.load(tmp_path / "prova__persone.npz")
    assert len(armi["armi_frame"]) > 0 and armi["armi_frame"].max() < int(dati["numero_frame"])
