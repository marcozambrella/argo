"""Tutti i test di Sicurezza NN in un file (richiesta di Marco: massimo 5 file Python + test)."""

from pathlib import Path
import json
import numpy as np
import pytest
import subprocess
import torch
import yaml

from sicurezza.dati import gruppi_visti_dal_detector, ClipAnnotata, INDICE_CLASSE, VideoDaEstrarre, classe_da_frase, dividi_per_gruppi, elenca_clip_cctv_fights, elenca_clip_chain_snatching, elenca_clip_rwf2000, elenca_clip_scvd, elenca_clip_ucf_uca, etichetta_finestra, inizi_finestre, leggi_annotazioni, leggi_elenco, leggi_frame, leggi_info_video, nome_file_estratto, prepara, scrivi_annotazioni, scrivi_elenco
from sicurezza.demo import confidenza_arma_per_frame, tara_warning_arma, analizza_video, arma_vicina_a_persona, classe_finale, frame_con_arma, warning_arma
from sicurezza.modelli import ARCO_ARMA_POLSO, ARCO_OSSO, ARCO_PERSONE_DIVERSE, CARATTERISTICHE_PER_NODO, MODELLI, NODI_PERSONE, NODI_PER_FRAME, addestra, carica_timeline, conta_parametri, costruisci_dataset, costruisci_finestra, crea_modello, matrice_tipi_arco, riepilogo
from sicurezza.strumenti import scegli_immagini_coco
from sicurezza.visione import conta_falsi_allarmi, converti_mock_attack, sequenza_del_frame, Estrattore, estrai_elenco, impacchetta_estrazione, unisci


# ==============================================================================================================
# aiuti_sintetici
# ==============================================================================================================

def scheletro_in_piedi(centro_x: float, piedi_y: float, altezza: float) -> np.ndarray:
    """17 keypoint (x, y, confidenza) di una persona in piedi, coordinate in pixel."""
    proporzioni = np.array([
        [0, 0.07], [-.02, .05], [.02, .05], [-.04, .06], [.04, .06], [-.1, .2], [.1, .2], [-.13, .38], [.13, .38],
        [-.14, .52], [.14, .52], [-.07, .53], [.07, .53], [-.07, .76], [.07, .76], [-.07, 1.0], [.07, 1.0]])
    keypoint = np.empty((17, 3), np.float32)
    keypoint[:, 0] = centro_x + proporzioni[:, 0] * altezza
    keypoint[:, 1] = piedi_y - altezza + proporzioni[:, 1] * altezza
    keypoint[:, 2] = 0.9
    return keypoint


def estrazione_finta(numero_frame=60, numero_persone=3, con_arma=True, larghezza=640, altezza=480, seme=0):
    generatore = np.random.default_rng(seme)
    righe_persone, righe_armi = [], []
    for frame in range(numero_frame):
        keypoint, box = [], []
        for persona in range(numero_persone):
            centro_x = 100 + 150 * persona + 2 * frame + generatore.normal(0, 1)
            scheletro = scheletro_in_piedi(centro_x, 400, 200)
            keypoint.append(scheletro)
            box.append([scheletro[:, 0].min(), scheletro[:, 1].min(), scheletro[:, 0].max(), scheletro[:, 1].max()])
        righe_persone.append((frame, {"id_traccia": np.arange(1, numero_persone + 1), "box": np.array(box),
                                      "confidenza": np.full(numero_persone, .9), "keypoint": np.array(keypoint)}))
        if con_arma and frame % 2 == 0:
            polso = keypoint[0][10, :2]
            righe_armi.append((frame, {"box": np.array([[*polso - 10, *polso + 10]]), "classe": np.array([0]),
                                       "confidenza": np.array([0.8])}))
        else:
            righe_armi.append((frame, {"box": np.empty((0, 4)), "classe": np.empty(0, int),
                                       "confidenza": np.empty(0)}))
    return impacchetta_estrazione(righe_persone, righe_armi, numero_frame, larghezza, altezza, 0.0)


def estrazione_finta_con_agitazione(agitazione: float, seme: int, numero_frame=60):
    """Come estrazione_finta, ma con i polsi che si muovono tanto quanto `agitazione` (finta "violenza")."""
    dati = estrazione_finta(numero_frame=numero_frame, numero_persone=2, con_arma=False, seme=seme)
    generatore = np.random.default_rng(seme)
    keypoint = dati["persone_keypoint"].copy()
    keypoint[:, 9:11, :2] += generatore.normal(0, agitazione, keypoint[:, 9:11, :2].shape)
    dati["persone_keypoint"] = keypoint
    return dati


# ==============================================================================================================
# test_finestre
# ==============================================================================================================

NORMALE, VIOLENZA = INDICE_CLASSE["normale"], INDICE_CLASSE["violenza"]


def test_prime_finestre_0_39_e_10_49():
    inizi = inizi_finestre(100)
    assert [(inizio, inizio + 39) for inizio in inizi[:2]] == [(0, 39), (10, 49)]
    # l'ultima finestra deve stare tutta nel video
    assert inizi[-1] == 60 and inizi[-1] + 40 <= 100


def test_video_troppo_corto_nessuna_finestra():
    assert len(inizi_finestre(39)) == 0
    assert list(inizi_finestre(40)) == [0]


def test_etichetta_soglia_20_frame():
    eventi = [(30, 200, VIOLENZA)]
    assert etichetta_finestra(0, eventi, classe_sfondo=NORMALE) is None        # 10 frame di evento: ambigua
    assert etichetta_finestra(10, eventi, classe_sfondo=NORMALE) == VIOLENZA   # 20 frame: esattamente la soglia
    assert etichetta_finestra(200, eventi, classe_sfondo=NORMALE) == NORMALE   # nessun evento
    assert etichetta_finestra(200, eventi, classe_sfondo=None) is None         # sorgente senza sfondo affidabile


def test_eventi_sovrapposti_non_contano_doppio():
    # due eventi lunghi che nella finestra 0-39 coprono gli stessi 15 frame (25-39): 15, non 25 → ambigua
    eventi = [(25, 80, VIOLENZA), (30, 80, VIOLENZA)]
    assert etichetta_finestra(0, eventi, classe_sfondo=NORMALE) is None


def test_evento_breve_conta_se_contenuto_per_intero():
    """D22: uno scippo di 12 frame (0,5 s) prende la finestra che lo contiene tutto, non quelle che lo tagliano."""
    SCIPPO = INDICE_CLASSE["scippo"]
    eventi = [(45, 57, SCIPPO)]
    assert etichetta_finestra(20, eventi, classe_sfondo=None) == SCIPPO     # 20-59: contiene 45-56
    assert etichetta_finestra(40, eventi, classe_sfondo=None) == SCIPPO     # 40-79: contiene 45-56
    assert etichetta_finestra(50, eventi, classe_sfondo=None) is None       # 50-89: lo taglia → ambigua
    assert etichetta_finestra(0, eventi, classe_sfondo=None) is None        # 0-39: non lo tocca, sfondo None


# ==============================================================================================================
# test_video
# ==============================================================================================================

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


# ==============================================================================================================
# test_etichette
# ==============================================================================================================

def crea_file(radice, *percorsi):
    for percorso in percorsi:
        (radice / percorso).parent.mkdir(parents=True, exist_ok=True)
        (radice / percorso).write_bytes(b"")


def test_rwf2000_gruppo_per_video_originale(tmp_path):
    crea_file(tmp_path, "RWF-2000/train/Fight/abc_0.avi", "RWF-2000/train/Fight/abc_1.avi",
              "RWF-2000/val/NonFight/x_y_3.avi")
    clip = elenca_clip_rwf2000(tmp_path)
    assert [c.gruppo for c in clip] == ["rwf2000/abc", "rwf2000/abc", "rwf2000/x_y"]
    assert [c.classe_intera for c in clip] == ["violenza", "violenza", "normale"]


def test_scvd_solo_versione_video_interi(tmp_path):
    crea_file(tmp_path, "SCVD/SCVD_converted/Train/Weaponized/w1.avi",
              "SCVD/SCVD_converted_sec_split/Train/Weaponized/w1_0.avi")
    clip = elenca_clip_scvd(tmp_path)
    assert len(clip) == 1 and clip[0].classe_intera == "violenza_con_arma"


def test_cctv_fights_eventi_e_fonti(tmp_path):
    crea_file(tmp_path, "parte/fight_0001.mp4", "parte/fight_0002.webm")
    database = {"fight_0001": {"subset": "training", "source": "CCTV", "annotations": [
        {"segment": [1.0, 3.0], "label": "Fight"}]},
        "fight_0002": {"subset": "testing", "source": "Mobile", "annotations": []}}
    (tmp_path / "groundtruth.json").write_text(json.dumps({"database": database}))
    tutte = elenca_clip_cctv_fights(tmp_path)
    assert len(tutte) == 2 and tutte[0].eventi_in_frame(500) == [(25, 75, INDICE_CLASSE["violenza"])]
    assert len(elenca_clip_cctv_fights(tmp_path, fonti_ammesse=("CCTV",))) == 1
    assert [c.da_estrarre().telecamera_mobile for c in tutte] == [False, True]   # CCTV fissa, cellulare mobile


def test_ucf_segmenti_uniti_con_margine(tmp_path):
    crea_file(tmp_path, "Anomaly-Videos/Robbery/Robbery001_x264.mp4")
    (tmp_path / "uca").mkdir()
    (tmp_path / "uca" / "UCFCrime_Train.json").write_text(json.dumps({"Robbery001_x264": {
        "duration": 100.0, "timestamps": [[10.0, 12.0], [13.0, 15.0], [50.0, 52.0], [70, 80]],
        "sentences": ["The man took out a gun and pointed it at the clerk", "The man punched the clerk",
                      "A man snatched the phone of the woman", "The clerk walked to the door"]}}))
    clip = elenca_clip_ucf_uca(tmp_path, cartella_uca=tmp_path / "uca")
    assert [(c.inizio_secondi, c.fine_secondi) for c in clip] == [(8.0, 17.0), (48.0, 54.0)]
    assert clip[0].eventi_secondi == [(2.0, 4.0, "violenza_con_arma"), (5.0, 7.0, "violenza")]
    assert clip[1].eventi_secondi == [(2.0, 4.0, "scippo")]
    assert all(c.classe_sfondo is None for c in clip)


def test_annotazioni_andata_e_ritorno(tmp_path):
    crea_file(tmp_path, "RWF-2000/train/Fight/abc_0.avi")
    clip = elenca_clip_rwf2000(tmp_path)
    scrivi_annotazioni(clip, tmp_path / "a.jsonl")
    assert leggi_annotazioni(tmp_path / "a.jsonl") == clip


def test_regole_uca_casi_tipici():
    giudica = classe_da_frase
    # violenza su persone
    assert giudica("Someone knocked down the man in white with one punch.") == "violenza"
    assert giudica("The man pushed the blonde woman to the ground.") == "violenza"
    assert giudica("Two men clashed and fought on the bus.") == "violenza"
    # oggetti e veicoli: non sono violenza
    assert giudica("The green taxi hit a tricycle while reversing and stopped.") is None
    assert giudica("The man kicked open the door and entered the room.") is None
    assert giudica("Two people pushed a stretcher cart and walked to the man in white.") is None
    assert giudica("Several women pushed the sofa to the door, and the two robbers ran away") is None
    # armi
    assert giudica("A man opened the door and pointed a gun at the clerk.") == "violenza_con_arma"
    assert giudica("The man in the blue hat threatened two store clerks with a knife.") == "violenza_con_arma"
    # scippi, ma non strappare un'arma di mano
    assert giudica("The man came in and snatched the child's mobile phone.") == "scippo"
    assert giudica("The man in the hat tried to snatch the gun from the security guard") != "scippo"


def test_chain_snatching_usa_solo_i_tempi_annotati(tmp_path):

    crea_file(tmp_path, "Chain_Snatching_Videos/Chain_Snatching01.mp4", "Chain_Snatching_Videos/Chain_Snatching02.mp4",
              "Chain_Snatching_Videos/Chain_Snatching03.mp4")
    (tmp_path / "tempi.json").write_text(json.dumps({
        "Chain_Snatching01.mp4": {"eventi": [[2.0, 3.6]], "nessuno_scippo": False},
        "Chain_Snatching02.mp4": {"eventi": [], "nessuno_scippo": True}}))
    clip = elenca_clip_chain_snatching(tmp_path, file_tempi=tmp_path / "tempi.json")
    assert len(clip) == 3                                   # tutti estratti, anche quelli non annotati
    assert clip[0].eventi_in_frame(200) == [(50, 90, INDICE_CLASSE["scippo"])]
    assert clip[1].eventi_secondi == [] and clip[2].eventi_secondi == []
    assert all(c.classe_sfondo is None for c in clip)      # fuori dallo scippo: finestre scartate


# ==============================================================================================================
# test_prepara_elenchi
# ==============================================================================================================

def test_finto_kaggle_input(tmp_path):
    ingresso = tmp_path / "input"
    for percorso in ["rwf2000/RWF-2000/train/Fight/a_0.avi", "rwf2000/RWF-2000/val/NonFight/b_0.avi",
                     "scvd/SCVD/SCVD_converted/Train/Violence/v1.avi",
                     "chain/Chain_Snatching_Videos/Chain_Snatching01.mp4",
                     "cctv/original_format-001-200/fight_0001.mp4",
                     "ucf/Real-world Anomaly Detection in Surveillance Videos (UCF)/Anomaly-Videos/Abuse/Abuse001_x264.mp4"]:
        (ingresso / percorso).parent.mkdir(parents=True, exist_ok=True)
        (ingresso / percorso).write_bytes(b"")
    (ingresso / "cctv" / "groundtruth.json").write_text(json.dumps({"database": {"fight_0001": {
        "subset": "training", "source": "CCTV", "annotations": [{"segment": [1, 2], "label": "Fight"}]}}}))

    clip = prepara(ingresso, tmp_path / "uscita")
    sorgenti = {c.sorgente for c in clip}
    assert sorgenti == {"rwf2000", "scvd", "chain_snatching", "cctv_fights", "ucf_uca"}
    # Abuse001 ha eventi di violenza in UCA (vedi annotazioni reali): almeno un segmento
    assert any(c.id_clip.startswith("ucf/Abuse001_x264@") for c in clip)
    assert leggi_annotazioni(tmp_path / "uscita" / "annotazioni.jsonl") == clip
    elenco = leggi_elenco(tmp_path / "uscita" / "elenco.jsonl")
    assert sorted(v.id_clip for v in elenco) == sorted(c.id_clip for c in clip)
    assert [v.id_clip.split("/")[0] for v in elenco[:3]] != ["rwf2000"] * 3   # sorgenti alternate


# ==============================================================================================================
# test_suddivisione
# ==============================================================================================================

def test_nessun_gruppo_in_due_split_e_quote_rispettate():
    generatore = np.random.default_rng(1)
    gruppi = np.repeat([f"v{i}" for i in range(200)], generatore.integers(5, 30, 200))
    etichette = np.array([hash(g) % 4 for g in gruppi])
    sorgenti = np.array(["a" if int(g[1:]) % 2 else "b" for g in gruppi])
    split = dividi_per_gruppi(gruppi, etichette, sorgenti)
    for gruppo in np.unique(gruppi):
        assert len(np.unique(split[gruppi == gruppo])) == 1
    quote = np.bincount(split, minlength=3) / len(split)
    np.testing.assert_allclose(quote, [0.70, 0.15, 0.15], atol=0.05)


def test_video_visti_dal_detector_sempre_in_train():
    """D26: i video UCF usati per addestrare il detector armi non devono finire in val o test."""
    gruppi = np.repeat([f"ucf/Robbery{i:03d}_x264" for i in range(60)], 10)
    etichette, sorgenti = np.full(600, 2), np.array(["ucf_uca"] * 600)
    forzati = frozenset(f"ucf/Robbery{i:03d}_x264" for i in range(0, 60, 3))
    split = dividi_per_gruppi(gruppi, etichette, sorgenti, gruppi_sempre_in_train=forzati)
    assert all(split[gruppi == gruppo][0] == 0 for gruppo in forzati)
    assert set(split.tolist()) == {0, 1, 2}                 # gli altri si distribuiscono ancora su tutti gli split
    assert len(gruppi_visti_dal_detector()) == 43           # elenco reale in annotations/armi/


def test_riproducibile():
    gruppi = np.repeat([f"v{i}" for i in range(50)], 10)
    etichette, sorgenti = np.zeros(500, int), np.array(["a"] * 500)
    assert (dividi_per_gruppi(gruppi, etichette, sorgenti) == dividi_per_gruppi(gruppi, etichette, sorgenti)).all()


# ==============================================================================================================
# test_estrazione
# ==============================================================================================================

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


@pytest.mark.skipif(not PESI_POSE.exists(), reason="pesi yolo26n-pose non scaricati in runs/")
def test_secondo_passaggio_armi_stessi_frame(tmp_path):
    """Con un detector qualsiasi (qui il modello pose al posto di quello armi) i frame tornano allineati."""
    from ultralytics import YOLO

    from sicurezza.visione import armi_di_un_video
    video = VideoDaEstrarre("prova/persone", str(VIDEO_PERSONE))
    estrai_elenco([video], tmp_path, Estrattore(str(PESI_POSE), dispositivo="cpu"))
    armi = armi_di_un_video(YOLO(str(PESI_POSE)), video, "cpu")
    dati = np.load(tmp_path / "prova__persone.npz")
    assert len(armi["armi_frame"]) > 0 and armi["armi_frame"].max() < int(dati["numero_frame"])
    uno_alla_volta = armi_di_un_video(YOLO(str(PESI_POSE)), video, "cpu", dimensione_lotto=1)
    np.testing.assert_array_equal(armi["armi_frame"], uno_alla_volta["armi_frame"])   # lotti = un frame alla volta
    np.testing.assert_allclose(armi["armi_box"], uno_alla_volta["armi_box"], atol=0.5)


@pytest.mark.skipif(not PESI_POSE.exists(), reason="pesi yolo26n-pose non scaricati in runs/")
@pytest.mark.parametrize("tracker", ["tracktrack.yaml", "botsort.yaml"])
def test_lotti_e_tracker_a_mano_uguali_a_track_di_ultralytics(tracker):
    """D20: pose a lotti + tracker chiamato a mano deve dare gli stessi ID e box di model.track frame per frame."""
    from ultralytics import YOLO

    from sicurezza.dati import leggi_frame
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


# ==============================================================================================================
# test_caratteristiche
# ==============================================================================================================

def salva(tmp_path, **opzioni):
    percorso = tmp_path / "finta.npz"
    np.savez_compressed(percorso, **estrazione_finta(**opzioni))
    return percorso


def test_forme_e_maschera(tmp_path):
    timeline = carica_timeline(salva(tmp_path, numero_persone=3))
    nodi, maschera, classe_arma = costruisci_finestra(timeline, 10)
    assert nodi.shape == (40, NODI_PER_FRAME, CARATTERISTICHE_PER_NODO)
    assert maschera[:, :3 * 17].all() and not maschera[:, 3 * 17:NODI_PERSONE].any()  # 4ª persona assente
    assert np.isfinite(nodi).all()


def test_normalizzazione_invariante_a_zoom(tmp_path):
    """Stessa scena ripresa a risoluzione doppia → stesse caratteristiche."""
    piccola = estrazione_finta(con_arma=False)
    grande = {chiave: valore for chiave, valore in piccola.items()}
    grande["persone_keypoint"] = piccola["persone_keypoint"] * np.array([2, 2, 1], np.float32)
    grande["persone_box"] = piccola["persone_box"] * 2
    np.savez(tmp_path / "p.npz", **piccola)
    np.savez(tmp_path / "g.npz", **grande)
    nodi_piccola, _, _ = costruisci_finestra(carica_timeline(tmp_path / "p.npz"), 0)
    nodi_grande, _, _ = costruisci_finestra(carica_timeline(tmp_path / "g.npz"), 0)
    np.testing.assert_allclose(nodi_piccola, nodi_grande, atol=1e-4)


def test_arma_nei_frame_pari(tmp_path):
    timeline = carica_timeline(salva(tmp_path, con_arma=True))
    nodi, maschera, classe_arma = costruisci_finestra(timeline, 0)
    assert maschera[0::2, NODI_PERSONE].all() and not maschera[1::2, NODI_PERSONE].any()
    assert (classe_arma[0::2, 0] == 1).all()  # classe 0 del detector → 1 (0 = nessuna arma)
    assert np.allclose(nodi[0::2, NODI_PERSONE, 2], 0.8, atol=1e-3)


def test_finestra_senza_persone(tmp_path):
    timeline = carica_timeline(salva(tmp_path, numero_persone=0, con_arma=False))
    nodi, maschera, _ = costruisci_finestra(timeline, 0)
    assert not maschera.any() and not nodi.any()


def test_ricucitura_unisce_frammenti_ma_non_persone_diverse():
    from sicurezza.modelli import ricuci_tracce
    frame = 60

    def traccia(dal, al, centro_x):
        box = np.full((frame, 4), np.nan, np.float32)
        box[dal:al] = [centro_x - 40, 100, centro_x + 40, 300]   # persona alta 200 px
        keypoint = np.full((frame, 17, 3), np.nan, np.float32)
        keypoint[dal:al] = 1.0
        return keypoint, box

    tracce = {1: traccia(0, 20, 100), 2: traccia(25, 60, 110),    # stessa persona: buco di 5 frame, 10 px più in là
              3: traccia(0, 60, 400),                              # un'altra persona, presente per tutto il tempo
              4: traccia(28, 60, 700)}                             # parte lontano: non è il seguito di nessuno
    keypoint, box = ricuci_tracce({k: v[0] for k, v in tracce.items()}, {k: v[1] for k, v in tracce.items()})
    assert sorted(box) == [1, 3, 4]
    assert (~np.isnan(box[1][:, 0])).sum() == 55 and not np.isnan(keypoint[1][59]).any()


# ==============================================================================================================
# test_modelli
# ==============================================================================================================

def lotto_finto(lotto=3, frame=40):
    nodi = torch.randn(lotto, frame, NODI_PER_FRAME, 5)
    maschera = torch.zeros(lotto, frame, NODI_PER_FRAME, dtype=torch.bool)
    maschera[0, :, :2 * 17] = True                  # 2 persone
    maschera[1, :, :17] = True                       # 1 persona + arma
    maschera[1, ::2, NODI_PERSONE] = True
    # il terzo esempio è una scena vuota
    nodi = nodi * maschera[..., None]
    classe_arma = torch.zeros(lotto, frame, 2, dtype=torch.long)
    classe_arma[1, ::2, 0] = 2
    return nodi, maschera, classe_arma


@pytest.mark.parametrize("nome", list(MODELLI))
def test_uscita_e_gradienti(nome):
    modello = crea_modello(nome)
    logit = modello(*lotto_finto())
    assert logit.shape == (3, 4) and torch.isfinite(logit).all()
    logit.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in modello.parameters() if p.grad is not None)


def test_nodi_mascherati_non_contano_nel_graph_transformer():
    """Cambiare i valori dei nodi assenti non deve cambiare l'uscita.
    (L'ST-GCN, come l'originale, non usa la maschera dentro le convoluzioni: escluso.)"""
    modello = crea_modello("graph_transformer").eval()
    nodi, maschera, classe_arma = lotto_finto()
    rumore = torch.randn_like(nodi) * (~maschera[..., None])
    with torch.no_grad():
        prima = modello(nodi, maschera, classe_arma)
        dopo = modello(nodi + rumore, maschera, classe_arma)
    torch.testing.assert_close(prima, dopo, atol=1e-4, rtol=1e-4)


def test_tipi_arco():
    tipi = matrice_tipi_arco()
    assert tipi[5, 7] == ARCO_OSSO                     # spalla_sx – gomito_sx
    assert tipi[0, 17] == ARCO_PERSONE_DIVERSE         # naso persona 0 – naso persona 1
    assert tipi[NODI_PERSONE, 9] == ARCO_ARMA_POLSO    # arma – polso_sx persona 0
    assert (tipi == tipi.T).all()


def test_dimensioni_ragionevoli():
    assert 0.5e6 < conta_parametri(crea_modello("graph_transformer")) < 2e6
    assert 2e6 < conta_parametri(crea_modello("stgcn")) < 4e6


# ==============================================================================================================
# test_allarme_arma
# ==============================================================================================================

def test_arma_nella_mano_conta_arma_lontana_no():
    persona = np.array([[100, 100, 200, 400]], float)
    assert arma_vicina_a_persona(np.array([190, 250, 210, 270]), persona)
    assert not arma_vicina_a_persona(np.array([500, 250, 520, 270]), persona)
    assert not arma_vicina_a_persona(np.array([190, 250, 210, 270]), np.empty((0, 4)))


def test_warning_su_finestra(tmp_path):
    np.savez(tmp_path / "f.npz", **estrazione_finta(con_arma=True))   # arma nei frame pari, confidenza 0,8
    timeline = carica_timeline(tmp_path / "f.npz")
    assert frame_con_arma(timeline, 0) == 20
    assert warning_arma(timeline, 0)
    assert not warning_arma(timeline, 0, soglia_confidenza=0.9)


def test_regola_violenza_piu_arma():
    probabilita_violenza = np.array([0.1, 0.7, 0.1, 0.1])
    assert classe_finale(probabilita_violenza, arma_confermata=True) == "violenza_con_arma"
    assert classe_finale(probabilita_violenza, arma_confermata=False) == "violenza"
    assert classe_finale(np.array([0.7, 0.1, 0.1, 0.1]), arma_confermata=True) == "normale"  # solo warning


# ==============================================================================================================
# test_armi
# ==============================================================================================================

def crea_export(radice, nomi, etichette_per_immagine, split="train"):
    (radice / split / "images").mkdir(parents=True)
    (radice / split / "labels").mkdir(parents=True)
    (radice / "data.yaml").write_text(yaml.safe_dump({"names": nomi, "nc": len(nomi)}))
    for nome_immagine, righe in etichette_per_immagine.items():
        (radice / split / "images" / f"{nome_immagine}.jpg").write_bytes(b"jpg")
        (radice / split / "labels" / f"{nome_immagine}.txt").write_text("\n".join(righe))


def test_unione_con_classi_rimappate(tmp_path):
    crea_export(tmp_path / "master", ["Pistol", "Knife", "Rifle", "Grenade"],
                {"a": ["0 .5 .5 .1 .1", "3 .2 .2 .1 .1"], "b": ["2 .5 .5 .2 .2"]})
    crea_export(tmp_path / "prospace", ["helmet", "pistol", "knife"],
                {"a": ["0 .5 .5 .1 .1", "2 .3 .3 .1 .1"]}, split="valid")
    conteggio = unisci([tmp_path / "master", tmp_path / "prospace"], tmp_path / "unito")

    assert (tmp_path / "unito/labels/train/s0_a.txt").read_text() == "0 .5 .5 .1 .1\n"   # granata tolta
    assert (tmp_path / "unito/labels/train/s0_b.txt").read_text() == "2 .5 .5 .2 .2\n"   # rifle → 2
    assert (tmp_path / "unito/labels/val/s1_a.txt").read_text() == "1 .3 .3 .1 .1\n"     # knife 2 → 1, casco tolto
    assert conteggio[("train", "tolta:Grenade")] == 1 and conteggio[("val", "tolta:helmet")] == 1
    configurazione = yaml.safe_load((tmp_path / "unito/data.yaml").read_text())
    assert configurazione["names"] == {0: "pistol", 1: "knife", 2: "rifle"} and "test" not in configurazione


# ==============================================================================================================
# test_inferenza
# ==============================================================================================================

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


# ==============================================================================================================
# test_catena_completa
# ==============================================================================================================

@pytest.fixture(scope="module")
def dataset_finto(tmp_path_factory):
    cartella = tmp_path_factory.mktemp("estratti")
    clip = []
    for numero in range(40):
        violenta = numero % 2 == 0
        id_clip = f"finto/clip{numero}"
        np.savez(cartella / nome_file_estratto(id_clip),
                 **estrazione_finta_con_agitazione(agitazione=40.0 if violenta else 0.0, seme=numero))
        clip.append(ClipAnnotata(id_clip=id_clip, sorgente="finto", gruppo=f"finto/{numero}", percorso_video="-",
                                 classe_intera="violenza" if violenta else "normale"))
    return costruisci_dataset(clip, [cartella])


def test_dataset(dataset_finto):
    assert len(dataset_finto["etichetta"]) == 40 * 3   # 60 frame → finestre a 0, 10, 20
    assert dataset_finto["nodi"].dtype == np.float16
    assert "violenza" in riepilogo(dataset_finto)


# il Graph-Transformer piccolo è veloce ma impara più lentamente (~150 passi); l'ST-GCN è lento su CPU ma rapido
CONFIGURAZIONI = {"graph_transformer": dict(epoche=20, dimensione_lotto=8, iperparametri_modello={"dimensione": 32,
                                                                                                   "blocchi": 1}),
                  "stgcn": dict(epoche=12, dimensione_lotto=16)}


@pytest.mark.lento
@pytest.mark.parametrize("nome_modello", list(CONFIGURAZIONI))
def test_addestramento_impara(dataset_finto, nome_modello, tmp_path):
    risultati = addestra(dataset_finto, nome_modello, tmp_path, tasso_apprendimento=2e-3, pazienza=20,
                         **CONFIGURAZIONI[nome_modello])
    assert (tmp_path / "migliore.pt").exists() and (tmp_path / "split.npz").exists()
    assert json.loads((tmp_path / "risultati.json").read_text())["modello"] == nome_modello
    # polsi agitati vs fermi: il compito è facilissimo, un modello che funziona lo impara
    assert risultati["val"]["macro_f1"] >= 0.8, risultati["storia"]


# ==============================================================================================================
# US Mock Attack (D30)
# ==============================================================================================================

def scrivi_frame_voc(cartella, sequenza, numero, oggetti):
    nome = f"{sequenza}_frame_{numero}"
    (cartella / f"{nome}.jpg").write_bytes(b"jpg")
    righe_oggetti = "".join(f"<object><name>{classe}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
                            f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>" for classe, x1, y1, x2, y2 in oggetti)
    (cartella / f"{nome}.xml").write_text(f"<annotation><filename>{nome}</filename><size><width>1920</width>"
                                          f"<height>1080</height><depth>3</depth></size>{righe_oggetti}</annotation>")


def test_mock_attack_conversione_e_split_per_sequenza(tmp_path):
    sorgente = tmp_path / "Images"
    sorgente.mkdir()
    for indice_sequenza in range(10):
        sequenza = f"Cam{indice_sequenza % 3 + 1}-From09-00-00To10-00-00_Segment_{indice_sequenza}_x264"
        for frame in range(1, 6):
            oggetti = [("Handgun", 960, 540, 1000, 600)] if frame == 1 else \
                      [("Short_rifle", 0, 0, 192, 108), ("Knife", 10, 10, 20, 20)] if frame == 2 else []
            scrivi_frame_voc(sorgente, sequenza, frame, oggetti)
    conteggio = converti_mock_attack(sorgente, tmp_path / "mock")
    assert sum(conteggio[(split, "immagini")] for split in ("train", "valid", "test")) == 50
    assert sequenza_del_frame("Cam1-From09-23-00To10-03-25_Segment_0_x264_frame_17.jpg") == \
        "Cam1-From09-23-00To10-03-25_Segment_0_x264"
    # nessuna sequenza divisa tra split
    split_di = {}
    for etichette in (tmp_path / "mock").glob("*/labels/*.txt"):
        split_di.setdefault(sequenza_del_frame(etichette.name), set()).add(etichette.parent.parent.name)
    assert all(len(splits) == 1 for splits in split_di.values()) and len(split_di) == 10
    # unione con le classi del progetto: Handgun → pistol (0), Short_rifle → rifle (2), Knife → knife (1)
    unisci([tmp_path / "mock"], tmp_path / "unito")
    righe = [riga.split()[0] for f in (tmp_path / "unito" / "labels").rglob("*.txt") for riga in f.read_text().splitlines()]
    assert sorted(set(righe)) == ["0", "1", "2"] and righe.count("0") == 10 and righe.count("1") == 10
    vuoti = [f for f in (tmp_path / "unito" / "labels").rglob("*.txt") if not f.read_text().strip()]
    assert len(vuoti) == 30                                     # frame senza armi restano come negativi
    primo = next((tmp_path / "mock").glob("*/labels/*_frame_1.txt")).read_text().split()
    assert primo[0] == "0" and abs(float(primo[1]) - 980 / 1920) < 1e-4  # centro x normalizzato


def test_taratura_warning_arma_trova_la_soglia_giusta(tmp_path):
    """D35: finestre con arma sicura in 8 frame e finestre con un falso positivo debole in 2 frame."""
    np.savez(tmp_path / "f.npz", **estrazione_finta(con_arma=True))       # arma 0,8 nei frame pari
    timeline = carica_timeline(tmp_path / "f.npz")
    confidenze = confidenza_arma_per_frame(timeline, 0, margine=0.5)
    assert confidenze.shape == (40,) and (confidenze[0::2] > 0.79).all() and not confidenze[1::2].any()

    generatore = np.random.default_rng(0)
    con_arma = np.zeros((50, 40), np.float32)
    con_arma[:, :8] = 0.8                                                  # arma vera: 8 frame a 0,8
    senza_arma = np.zeros((50, 40), np.float32)
    senza_arma[:, :2] = generatore.uniform(0.3, 0.45, (50, 2))            # rumore: 2 frame deboli
    risultato = tara_warning_arma({0.5: np.concatenate([con_arma, senza_arma])},
                                  np.array([True] * 50 + [False] * 50))
    migliore = risultato["migliore"]
    assert migliore["f1"] == 1.0 and migliore["precisione"] == 1.0
    assert len(risultato["tabella"]) == 12 * 15


def test_negativi_coco_senza_doppioni_e_sempre_gli_stessi():
    annotazioni = {"categories": [{"id": 49, "name": "knife"}, {"id": 77, "name": "cell phone"},
                                  {"id": 87, "name": "scissors"}, {"id": 28, "name": "umbrella"}],
                   "images": [{"id": i, "file_name": f"{i:012d}.jpg"} for i in range(1, 8)],
                   "annotations": [{"image_id": i, "category_id": c} for i, c in
                                   [(5, 49), (3, 49), (3, 77), (1, 49), (7, 77), (6, 28), (6, 28)]]}
    scelte = scegli_immagini_coco(annotazioni, per_categoria=2)
    assert scelte["knife"] == ["000000000001.jpg", "000000000003.jpg"]   # le prime per id
    assert scelte["cell phone"] == ["000000000007.jpg"]                  # la 3 è già tra i coltelli
    assert scelte["scissors"] == [] and scelte["umbrella"] == ["000000000006.jpg"]


def test_falsi_allarmi_contati_per_categoria(tmp_path):
    class RisultatoFinto:
        def __init__(self, armi):
            self.boxes = type("Box", (), {"cls": torch.tensor([classe for classe, _ in armi], dtype=torch.float32),
                                          "conf": torch.tensor([confidenza for _, confidenza in armi]),
                                          "xywhn": torch.tensor([[0.5, 0.5, 0.2, 0.1]] * len(armi))})()

    class DetectorFinto:  # "trova" un coltello solo nelle immagini col nome che inizia per "falso"
        def __init__(self):
            self.names = {0: "pistol", 1: "knife", 2: "rifle"}

        def predict(self, percorsi, conf, **_):
            return [RisultatoFinto([(1, 0.7)] if Path(p).name.startswith("falso") else []) for p in percorsi]

    for categoria, nomi in {"knife": ["a.jpg", "falso_b.jpg", "c.jpg"], "cell_phone": ["d.png", "e.jpg"]}.items():
        (tmp_path / categoria).mkdir()
        for nome in nomi:
            (tmp_path / categoria / nome).write_bytes(b"")
    (tmp_path / "instances_val2017.json").write_text("{}")             # non è un'immagine: ignorato
    risultato = conta_falsi_allarmi(DetectorFinto(), tmp_path, dimensione_lotto=2)
    assert risultato["per_categoria"]["knife"] == {"immagini": 3, "con_allarme": 1, "frazione": 0.3333,
                                                   "classi": {"knife": 1}}
    assert risultato["per_categoria"]["cell_phone"]["con_allarme"] == 0
    assert risultato["immagini_con_allarme"] == [{"immagine": "knife/falso_b.jpg", "armi": [("knife", 0.7, 0.02)]}]
