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


@pytest.mark.skipif(not PESI_POSE.exists(), reason="pesi yolo26n-pose non scaricati in runs/")
@pytest.mark.parametrize("tracker", ["tracktrack.yaml", "botsort.yaml"])
def test_lotti_e_tracker_a_mano_uguali_a_track_di_ultralytics(tracker):
    """D20: pose a lotti + tracker chiamato a mano deve dare gli stessi ID e box di model.track frame per frame."""
    from ultralytics import YOLO

    from sicurezza.video import leggi_frame
    frame = list(leggi_frame(str(VIDEO_PERSONE)))
    riferimento = YOLO(str(PESI_POSE))
    attesi = [riferimento.track(f, persist=i > 0, tracker=tracker, conf=0.1, device="cpu", verbose=False)[0].boxes
              for i, f in enumerate(frame)]

    estrattore = Estrattore(str(PESI_POSE), dispositivo="cpu", tracker=tracker, dimensione_lotto=8)
    dati = estrattore.estrai_video(VideoDaEstrarre("prova", str(VIDEO_PERSONE), telecamera_mobile=True))  # GMC come .track
    for indice, box_attesi in enumerate(attesi):
        righe = dati["persone_frame"] == indice
        id_attesi = box_attesi.id.int().numpy() if box_attesi.id is not None else np.empty(0, int)
        assert list(dati["persone_id_traccia"][righe]) == list(id_attesi), f"frame {indice}"
        np.testing.assert_allclose(dati["persone_box"][righe], box_attesi.xyxy.numpy().reshape(-1, 4), atol=0.5)


@pytest.mark.skipif(not PESI_POSE.exists(), reason="pesi yolo26n-pose non scaricati in runs/")
def test_senza_gmc_per_telecamere_fisse():
    estrattore = Estrattore(str(PESI_POSE), dispositivo="cpu")
    estrattore.nuovo_video(telecamera_mobile=False)
    assert estrattore.tracker.args.gmc_method == "none"
    estrattore.nuovo_video(telecamera_mobile=True)
    assert estrattore.tracker.args.gmc_method == "sparseOptFlow"
