"""Tutti i test di Sicurezza NN in un file (richiesta di Marco: massimo 5 file Python + test)."""

from pathlib import Path
import json
import numpy as np
import pytest
import subprocess
import torch
import yaml

from sicurezza.dati import elenca_clip_normali_folla, prepara_normali, gruppi_visti_dal_detector, ClipAnnotata, INDICE_CLASSE, VideoDaEstrarre, classe_da_frase, dividi_per_gruppi, elenca_clip_cctv_fights, elenca_clip_chain_snatching, elenca_clip_phone_snatching, elenca_clip_rwf2000, elenca_clip_scvd, elenca_clip_ucf_uca, etichetta_finestra, inizi_finestre, leggi_annotazioni, leggi_elenco, leggi_frame, leggi_info_video, nome_file_estratto, prepara, scrivi_annotazioni, scrivi_elenco
from sicurezza.demo import probabilita_media, valuta_sistema, arma_vicina, persone_nel_frame, confidenza_arma_per_frame, tara_warning_arma, analizza_video, arma_vicina_a_persona, classe_finale, frame_con_arma, warning_arma
from sicurezza.modelli import analizza_predizioni, ricuci_tracce, armi_non_piu_grandi_della_persona, timeline_da_dati, ARCO_ARMA_POLSO, ARCO_OSSO, ARCO_PERSONE_DIVERSE, CARATTERISTICHE_PER_NODO, MODELLI, NODI_PERSONE, NODI_PER_FRAME, addestra, carica_timeline, conta_parametri, costruisci_dataset, costruisci_finestra, crea_modello, matrice_tipi_arco, riepilogo
from sicurezza.strumenti import risultati_per_il_sito, scarica_con_riprova, scegli_candidati_scippo_telefono, scegli_immagini_coco
from sicurezza.visione import mani_di_una_persona, precisione_media, probabilita_armi_in_mano, ritaglio_quadrato, ritagli_per_addestramento, prepara_sohas, conta_falsi_allarmi, converti_mock_attack, sequenza_del_frame, Estrattore, estrai_elenco, impacchetta_estrazione, unisci


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


def test_phone_snatching_stessa_logica_del_chain_snatching(tmp_path):
    crea_file(tmp_path, "phone_snatching/x7MHAcDekHw.mp4", "phone_snatching/6cECGbEud4g.mp4")
    (tmp_path / "tempi.json").write_text(json.dumps({"x7MHAcDekHw.mp4": {"eventi": [[1.0, 2.0]], "nessuno_scippo": False}}))
    clip = elenca_clip_phone_snatching(tmp_path, file_tempi=tmp_path / "tempi.json")
    assert [c.id_clip for c in clip] == ["phone_snatching/6cECGbEud4g", "phone_snatching/x7MHAcDekHw"]
    assert clip[0].eventi_secondi == [] and clip[1].eventi_secondi == [(1.0, 2.0, "scippo")]
    assert all(c.sorgente == "phone_snatching" and c.gruppo == c.id_clip for c in clip)


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


def ricuci_tracce_di_riferimento(keypoint_per_traccia, box_per_traccia, buco_massimo=15, distanza_massima=0.5,
                                 rapporto_altezze_massimo=1.5):
    """La prima versione (tutte contro tutte, lenta ma ovvia): la nuova deve dare esattamente lo stesso risultato."""
    def estremi(box):
        presenti = np.flatnonzero(~np.isnan(box[:, 0]))
        return presenti[0], presenti[-1]

    def centro_e_altezza(riga):
        return np.array([(riga[0] + riga[2]) / 2, (riga[1] + riga[3]) / 2]), riga[3] - riga[1]

    keypoint_per_traccia = dict(keypoint_per_traccia)
    box_per_traccia = {chiave: box.copy() for chiave, box in box_per_traccia.items()}
    for id_nuova in sorted(box_per_traccia, key=lambda chiave: estremi(box_per_traccia[chiave])[0]):
        if id_nuova not in box_per_traccia:
            continue
        inizio_nuova, _ = estremi(box_per_traccia[id_nuova])
        centro_nuova, altezza_nuova = centro_e_altezza(box_per_traccia[id_nuova][inizio_nuova])
        migliore, distanza_migliore = None, np.inf
        for id_vecchia, box_vecchia in box_per_traccia.items():
            _, fine_vecchia = estremi(box_vecchia)
            if id_vecchia == id_nuova or not (0 < inizio_nuova - fine_vecchia <= buco_massimo):
                continue
            centro_vecchia, altezza_vecchia = centro_e_altezza(box_vecchia[fine_vecchia])
            if max(altezza_vecchia, altezza_nuova) > rapporto_altezze_massimo * max(min(altezza_vecchia, altezza_nuova), 1):
                continue
            distanza = np.linalg.norm(centro_nuova - centro_vecchia) / max(altezza_vecchia, 1)
            if distanza <= distanza_massima and distanza < distanza_migliore:
                migliore, distanza_migliore = id_vecchia, distanza
        if migliore is not None:
            presenti = ~np.isnan(box_per_traccia[id_nuova][:, 0])
            box_per_traccia[migliore][presenti] = box_per_traccia[id_nuova][presenti]
            keypoint_unito = keypoint_per_traccia[migliore].copy()
            keypoint_unito[presenti] = keypoint_per_traccia[id_nuova][presenti]
            keypoint_per_traccia[migliore] = keypoint_unito
            del box_per_traccia[id_nuova], keypoint_per_traccia[id_nuova]
    return keypoint_per_traccia, box_per_traccia


@pytest.mark.parametrize("seme", range(5))
def test_ricuci_tracce_veloce_uguale_alla_versione_di_riferimento(seme):
    """Folla finta: tante tracce brevi e vicine, con buchi e pareggi di distanza (stesse posizioni ripetute)."""
    generatore = np.random.default_rng(seme)
    frame, keypoint_per_traccia, box_per_traccia = 120, {}, {}
    for id_traccia in generatore.permutation(80):
        dal = int(generatore.integers(0, 110))
        al = min(frame, dal + int(generatore.integers(1, 25)))
        x, y = generatore.choice([100.0, 120.0, 140.0]), generatore.choice([200.0, 210.0])
        altezza = generatore.choice([150.0, 200.0, 320.0])
        box = np.full((frame, 4), np.nan, np.float32)
        box[dal:al] = [x - 30, y, x + 30, y + altezza]
        keypoint = np.full((frame, 17, 3), np.nan, np.float32)
        keypoint[dal:al] = generatore.random((al - dal, 17, 3))
        keypoint_per_traccia[int(id_traccia)], box_per_traccia[int(id_traccia)] = keypoint, box
    attesi = ricuci_tracce_di_riferimento(keypoint_per_traccia, box_per_traccia)
    ottenuti = ricuci_tracce(keypoint_per_traccia, box_per_traccia)
    assert list(ottenuti[1]) == list(attesi[1]) and len(ottenuti[1]) < len(box_per_traccia)   # qualcuna si è unita
    for chiave in attesi[1]:
        np.testing.assert_array_equal(ottenuti[1][chiave], attesi[1][chiave])
        np.testing.assert_array_equal(ottenuti[0][chiave], attesi[0][chiave])


def test_timeline_di_una_folla_si_carica_in_fretta(tmp_path):
    """Migliaia di tracce brevi (scene di folla): prima ci volevano decine di minuti per una clip."""
    import time
    generatore = np.random.default_rng(0)
    frame, persone_per_frame = 600, 100
    righe_frame, righe_id, prossimo_id, attive = [], [], 0, []
    for numero in range(frame):
        attive = [(id_traccia, fine) for id_traccia, fine in attive if fine > numero]
        while len(attive) < persone_per_frame:
            attive.append((prossimo_id, numero + int(generatore.integers(10, 40))))
            prossimo_id += 1
        righe_frame += [numero] * len(attive)
        righe_id += [id_traccia for id_traccia, _ in attive]
    righe = len(righe_frame)
    box = generatore.uniform(0, 1000, (righe, 4)).astype(np.float32)
    box[:, 2:] = box[:, :2] + 80
    np.savez_compressed(tmp_path / "folla.npz", numero_frame=np.int32(frame), larghezza=np.int32(1920),
                        altezza=np.int32(1080), persone_frame=np.array(righe_frame, np.int32),
                        persone_id_traccia=np.array(righe_id, np.int32), persone_box=box,
                        persone_keypoint=generatore.random((righe, 17, 3), np.float32),
                        armi_frame=np.zeros(0, np.int32), armi_box=np.zeros((0, 4), np.float32),
                        armi_classe=np.zeros(0, np.int32), armi_confidenza=np.zeros(0, np.float16))
    inizio = time.perf_counter()
    timeline = carica_timeline(tmp_path / "folla.npz")
    assert prossimo_id > 2000 and len(timeline.box_per_traccia) <= prossimo_id
    assert time.perf_counter() - inizio < 60


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



def test_arma_in_mano_si_scarpa_no(tmp_path):
    """Criterio "polso": la pistola nella mano destra conta, la stessa "arma" sulla caviglia (scarpa nera) no."""
    np.savez(tmp_path / "f.npz", **estrazione_finta(con_arma=True))     # arma attaccata al polso nei frame pari
    timeline = carica_timeline(tmp_path / "f.npz")
    assert frame_con_arma(timeline, 0, vicinanza="polso:0.1") == 20
    box_persone, polsi = persone_nel_frame(timeline, 0)
    caviglia = timeline.keypoint_per_traccia[1][0][16, :2]
    scarpa = np.array([*caviglia - 10, *caviglia + 10])
    assert arma_vicina(scarpa, box_persone, polsi, "persona:0.5")       # vicino alla persona: sì
    assert not arma_vicina(scarpa, box_persone, polsi, "polso:0.2")     # vicino a un polso: no
    assert not arma_vicina(scarpa, np.empty((0, 4)), np.empty((0, 2, 3)), "polso:0.2")


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


def test_media_dei_classificatori():
    """DA DECIDERE 22 d: un checkpoint = la sua softmax; due checkpoint = la media delle due softmax."""
    torch.manual_seed(0)
    modelli = [crea_modello("stgcn").eval(), crea_modello("graph_transformer").eval()]
    nodi, maschera, classe_arma = (tensore[0].numpy() for tensore in lotto_finto())
    with torch.no_grad():
        singole = [probabilita_media([modello], nodi, maschera, classe_arma) for modello in modelli]
        media = probabilita_media(modelli, nodi, maschera, classe_arma)
        diretta = torch.softmax(modelli[0](*(torch.from_numpy(x)[None] for x in (nodi, maschera, classe_arma))), -1)[0]
    np.testing.assert_allclose(singole[0], diretta.numpy(), atol=1e-6)
    np.testing.assert_allclose(media, (singole[0] + singole[1]) / 2, atol=1e-6)
    assert media.shape == (4,) and abs(media.sum() - 1) < 1e-5


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


@pytest.mark.lento
def test_tetto_di_tempo_ferma_e_salva_il_migliore(dataset_finto, tmp_path):
    risultati = addestra(dataset_finto, "graph_transformer", tmp_path, epoche=5, dimensione_lotto=16,
                         ore_massime=1e-9)
    assert len(risultati["storia"]) == 1 and risultati["test"] is not None and (tmp_path / "migliore.pt").exists()


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
    confidenze = confidenza_arma_per_frame(timeline, 0, vicinanza="persona:0.5")
    assert confidenze.shape == (40,) and (confidenze[0::2] > 0.79).all() and not confidenze[1::2].any()

    generatore = np.random.default_rng(0)
    con_arma = np.zeros((50, 40), np.float32)
    con_arma[:, :8] = 0.8                                                  # arma vera: 8 frame a 0,8
    senza_arma = np.zeros((50, 40), np.float32)
    senza_arma[:, :2] = generatore.uniform(0.3, 0.45, (50, 2))            # rumore: 2 frame deboli
    risultato = tara_warning_arma({"persona:0.5": np.concatenate([con_arma, senza_arma])},
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


def test_estrai_passa_il_tracker_scelto_all_estrattore(monkeypatch):
    import sicurezza.visione as visione
    creati = []
    monkeypatch.setattr(visione, "Estrattore", lambda *posizionali, **nominali: creati.append(nominali) or "estrattore")
    monkeypatch.setattr(visione, "leggi_elenco", lambda percorso: [])
    monkeypatch.setattr(visione, "estrai_elenco", lambda *argomenti: None)
    visione.comando_estrai(["--elenco", "elenco.jsonl", "--uscita", "estratti", "--tracker", "botsort.yaml"])
    visione.comando_estrai(["--elenco", "elenco.jsonl", "--uscita", "estratti"])
    assert [nominali["tracker"] for nominali in creati] == ["botsort.yaml", visione.TRACKER_PREDEFINITO]


def test_addestra_mani_salva_dove_dice_anche_con_percorso_relativo(tmp_path, monkeypatch):
    import ultralytics
    from sicurezza.visione import addestra_mani
    chiamate = []

    class YoloFinto:
        def __init__(self, pesi):
            pass

        def train(self, **opzioni):
            chiamate.append(opzioni)

    monkeypatch.setattr(ultralytics, "YOLO", YoloFinto)
    monkeypatch.chdir(tmp_path)
    pesi = addestra_mani("ritagli", "uscita_relativa")
    assert Path(chiamate[0]["project"]).is_absolute()
    assert Path(chiamate[0]["project"]) / "yolo26s_mani" / "weights" / "best.pt" == pesi.resolve()


def test_valuta_mani_combina_mani_e_detector(tmp_path, monkeypatch):
    """Due immagini: nella prima un coltello che vedono solo le mani, nella seconda niente ma il detector sbaglia."""
    import cv2
    import sicurezza.visione as visione
    for split_cartella in ("images/test", "labels/test"):
        (tmp_path / split_cartella).mkdir(parents=True)
    for nome in ("a", "b"):
        cv2.imwrite(str(tmp_path / "images" / "test" / f"{nome}.jpg"), np.zeros((32, 32, 3), np.uint8))
    (tmp_path / "labels" / "test" / "a.txt").write_text("1 0.5 0.5 0.1 0.1\n")   # 1 = knife

    class Risultato:
        def __init__(self, box, classi=(), confidenze=()):
            self.boxes = type("Box", (), {"xyxy": torch.tensor(box, dtype=torch.float32).reshape(-1, 4),
                                          "cls": torch.tensor(classi, dtype=torch.float32),
                                          "conf": torch.tensor(confidenze, dtype=torch.float32)})()
            self.keypoints = type("Kp", (), {"xy": torch.zeros(len(self.boxes.xyxy), 17, 2),
                                             "conf": torch.ones(len(self.boxes.xyxy), 17)})()

    class Pose:
        def predict(self, immagine, **opzioni):
            return [Risultato([[1, 1, 30, 30]])]

    class Detector:
        names = {0: "pistol", 1: "knife", 2: "rifle"}
        chiamate = 0

        def predict(self, immagine, **opzioni):
            Detector.chiamate += 1
            return [Risultato([[1, 1, 5, 5]], [1], [0.1])] if Detector.chiamate == 1 else \
                [Risultato([[1, 1, 5, 5]], [1], [0.6])]

    mani_per_immagine = iter([np.array([[0.0, 0.9, 0.0]]), np.array([[0.0, 0.2, 0.0]])])
    monkeypatch.setattr(visione, "probabilita_armi_in_mano", lambda *argomenti: next(mani_per_immagine))
    valutazione = visione.valuta_mani(tmp_path, "test", Pose(), "mani", Detector())
    assert valutazione["con_classe"] == {"pistol": 0, "knife": 1, "rifle": 0}
    assert valutazione["mani"]["knife"]["precisione_media"] == 1.0
    assert valutazione["detector"]["knife"]["precisione_media"] == 0.5     # il falso allarme della b viene prima
    assert valutazione["combinati"]["knife"]["recall"] == 1.0
    assert valutazione["combinati"]["knife"]["falsi_allarmi"] == 1.0       # la b: 0,6 del detector ≥ 0,5


def test_titolo_della_presentazione_con_due_punti_e_percentuali(tmp_path):
    """drawtext di ffmpeg: ":" e "%" vanno protetti (con "82-98%" il montaggio si fermava)."""
    from sicurezza.strumenti import scena_titolo
    scena_titolo({"testo": "Folle: niente falsi allarmi", "sottotitolo": "82-98% giuste, 100 %{pts}", "secondi": 1},
                 320, 180, 25, tmp_path / "titolo.mp4")
    assert leggi_info_video(str(tmp_path / "titolo.mp4")).durata_secondi == 1.0


def test_scarica_con_riprova_riprova_e_poi_si_arrende(tmp_path, monkeypatch):
    tentativi = []

    def scarica_finto(url, destinazione):
        tentativi.append(url)
        if len(tentativi) < 3:
            raise TimeoutError("connessione bloccata")   # socket.timeout è un OSError
        Path(destinazione).write_bytes(b"ok")

    monkeypatch.setattr("urllib.request.urlretrieve", scarica_finto)
    scarica_con_riprova("http://esempio/immagine.jpg", tmp_path / "immagine.jpg")
    assert len(tentativi) == 3 and (tmp_path / "immagine.jpg").read_bytes() == b"ok"

    def rete_giu(url, destinazione):
        tentativi.append(url)
        raise OSError("rete giù")

    tentativi.clear()
    monkeypatch.setattr("urllib.request.urlretrieve", rete_giu)
    with pytest.raises(OSError):
        scarica_con_riprova("http://esempio/altra.jpg", tmp_path / "altra.jpg", tentativi=2)
    assert len(tentativi) == 2


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


def test_salva_split_senza_addestrare_uguale_a_quello_di_addestra(dataset_finto, tmp_path):
    """La taratura del WARNING ARMA (kernel solo CPU) usa lo split senza addestrare: deve essere lo stesso."""
    from sicurezza.modelli import comando_salva_split, indici_degli_split
    np.savez(tmp_path / "finestre.npz", **dataset_finto)
    comando_salva_split(["--finestre", str(tmp_path / "finestre.npz"), "--uscita", str(tmp_path / "split.npz")])
    split, attesi = np.load(tmp_path / "split.npz"), indici_degli_split(dataset_finto)
    for nome, indici in attesi.items():
        np.testing.assert_array_equal(split[nome], indici)
        assert set(split[f"clip_{nome}"]) == set(dataset_finto["id_clip"][indici])


def test_predici_da_checkpoint_sulle_finestre_di_uno_split(dataset_finto, tmp_path):
    from sicurezza.modelli import CLASSI, comando_predici, crea_modello, indici_degli_split
    modello = crea_modello("stgcn")
    torch.save({"nome_modello": "stgcn", "iperparametri_modello": {}, "classi": CLASSI, "stato": modello.state_dict(),
                "epoca": 1}, tmp_path / "migliore.pt")
    np.savez(tmp_path / "finestre.npz", **dataset_finto)
    np.savez(tmp_path / "split.npz", **indici_degli_split(dataset_finto))
    comando_predici(["--finestre", str(tmp_path / "finestre.npz"), "--checkpoint", str(tmp_path / "migliore.pt"),
                     "--split", str(tmp_path / "split.npz"), "--uscita", str(tmp_path / "probabilita.npy")])
    probabilita = np.load(tmp_path / "probabilita.npy")
    assert probabilita.shape == (len(indici_degli_split(dataset_finto)["test"]), len(CLASSI))
    np.testing.assert_allclose(probabilita.sum(1), 1, atol=1e-4)


@pytest.mark.lento  # un'epoca vera di ST-GCN su CPU: ~50 s
def test_risultati_per_il_sito_dal_vero_addestramento(dataset_finto, tmp_path):
    """Il formato letto da site/app.js (nome, macro_f1, latenza_cpu_ms) esce dai risultati.json veri di addestra."""
    risultati = addestra(dataset_finto, "stgcn", tmp_path / "stgcn", epoche=1, dimensione_lotto=16)
    taratura = {"migliore": {"soglia": 0.5, "frame_minimi": 5, "vicinanza": "polso:0.2", "f1": 0.8}, "tabella": []}
    sito = risultati_per_il_sito([json.loads(json.dumps(risultati))], taratura)
    modello = sito["modelli"][0]
    assert modello["nome"] == "ST-GCN" and 0 <= modello["macro_f1"] <= 1 and modello["latenza_cpu_ms"] > 0
    assert sito["classi"] == ["normale", "violenza", "violenza_con_arma", "scippo"]
    split = np.load(tmp_path / "stgcn" / "split.npz")
    assert modello["finestre_test"] == len(split["test"])
    assert not set(split["clip_test"]) & set(split["clip_train"]) and len(split["clip_test"]) > 0
    finestre_test = np.load(tmp_path / "stgcn" / "finestre_test.npz")
    assert len(finestre_test["sorgente"]) == len(split["test"]) == len(np.load(tmp_path / "stgcn" / "probabilita_test.npy"))
    # l'analisi rifà lo split sulle stesse finestre e si allinea alle probabilità salvate
    analisi = analizza_predizioni(dataset_finto, {"stgcn": np.load(tmp_path / "stgcn" / "probabilita_test.npy")})
    per_clip = analisi["modelli"]["stgcn"]["per_clip"]
    assert [c["id_clip"] for c in per_clip] == sorted(split["clip_test"].tolist())
    assert analisi["modelli"]["stgcn"]["per_sorgente"]["finto"]["finestre"] == len(split["test"])
    assert sito["warning_arma"]["frame_minimi"] == 5 and risultati_per_il_sito([])["modelli"] == []


def test_arma_piu_grande_della_persona_scartata():
    """D38: la pistola nella mano resta; un "fucile" grande come mezza scena vicino alle persone si scarta."""
    dati = dict(estrazione_finta(numero_frame=4, numero_persone=2))   # pistola 20×20 px nei frame pari
    dati["armi_frame"] = np.append(dati["armi_frame"], [1, 3]).astype(np.int32)
    dati["armi_box"] = np.concatenate([dati["armi_box"], [[0, 0, 600, 450], [5, 5, 15, 15]]]).astype(np.float32)
    dati["armi_classe"] = np.append(dati["armi_classe"], [2, 1]).astype(np.int32)
    dati["armi_confidenza"] = np.append(dati["armi_confidenza"], [0.9, 0.9]).astype(np.float16)
    tenute = armi_non_piu_grandi_della_persona(dati)
    assert tenute.tolist() == [True, True, False, True]        # 2 pistole, il riquadro enorme no, il coltellino sì
    timeline = timeline_da_dati(dati)
    assert len(timeline.armi_frame) == 3 and 2 not in timeline.armi_classe.tolist()
    assert len(timeline_da_dati(dati, filtra_armi_grandi=False).armi_frame) == 4


def scrivi_export_yolo(cartella, split, nome, righe):
    (cartella / split / "images").mkdir(parents=True, exist_ok=True)
    (cartella / split / "labels").mkdir(parents=True, exist_ok=True)
    (cartella / split / "images" / f"{nome}.jpg").write_bytes(b"jpg")
    (cartella / split / "labels" / f"{nome}.txt").write_text("".join(f"{riga}\n" for riga in righe))


def test_sohas_classi_innocue_tolte_e_primi_piani_fuori_dal_train(tmp_path):
    sohas = tmp_path / "Sohas_weapon-Detection-YOLOv5"
    (sohas).mkdir()
    (sohas / "dataset.yaml").write_text("nc: 6\nnames: ['pistol', 'smartphone', 'knife', 'monedero', 'billete', 'tarjeta']\n")
    for numero in range(20):                 # train: coltelli piccoli, 1 pistola in primo piano, telefoni impugnati
        righe = ["2 0.5 0.5 0.1 0.1"] if numero < 10 else ["1 0.5 0.5 0.2 0.2"] if numero < 19 else ["0 0.5 0.5 0.8 0.8"]
        (sohas / "obj_train_data" / "images" / "train").mkdir(parents=True, exist_ok=True)
        (sohas / "obj_train_data" / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (sohas / "obj_train_data" / "images" / "train" / f"img{numero}.jpg").write_bytes(b"jpg")
        (sohas / "obj_train_data" / "labels" / "train" / f"img{numero}.txt").write_text(righe[0] + "\n")
    (sohas / "obj_train_data" / "images" / "test").mkdir(parents=True)
    (sohas / "obj_train_data" / "labels" / "test").mkdir(parents=True)
    (sohas / "obj_train_data" / "images" / "test" / "prova.jpg").write_bytes(b"jpg")
    (sohas / "obj_train_data" / "labels" / "test" / "prova.txt").write_text("0 0.5 0.5 0.9 0.9\n")
    conteggio = prepara_sohas(sohas, tmp_path / "sohas_export")
    assert conteggio[("train", "immagini")] == 18 and conteggio[("valid", "immagini")] == 2
    assert conteggio[("test", "immagini")] == 1

    unito = unisci([tmp_path / "sohas_export"], tmp_path / "unito", area_massima_train=0.1)
    assert sum(unito[(split, voce)] for split in ("train", "val") for voce in ("immagini", "tolta:primo_piano")) == 20
    assert unito[("train", "tolta:primo_piano")] + unito[("val", "tolta:primo_piano")] == 1   # la pistola 0,8×0,8
    assert unito[("train", "knife")] + unito[("val", "knife")] == 10                         # 0,1×0,1 = 1%: resta
    assert unito[("train", "tolta:smartphone")] + unito[("val", "tolta:smartphone")] == 9    # immagini restano negative
    assert unito[("test", "pistol")] == 1                         # il test resta intero, anche coi primi piani
    righe_knife = [r for f in (tmp_path / "unito" / "labels").rglob("*.txt") for r in f.read_text().splitlines()]
    assert {r.split()[0] for r in righe_knife} == {"0", "1"}   # knife → indice 1 di CLASSI_ARMA, pistol → 0


def test_candidati_scippo_di_telefono():
    righe = ["[1] aaa | 30 | Canale | 100 | Phone Snatch at Shell Petrol Station",
             "[2] aaa | 30 | Canale | 100 | Phone Snatch at Shell Petrol Station",          # doppione
             "bbb | 1:22 | Social Media Library | Karachi Mobile Snatching Accused Arrested",
             "ccc | 44 | X | 1 | Chain Snatching from Old Lady",                                # catenina: già coperta
             "ddd | 26 | X | 1 | Mobile Snatching Prank",                                       # scherzo
             "eee | 26 | X | 1 | nothing phone 3 first look review",                            # niente furto
             "fff | 1:45 | X | 1 | Câmeras flagram furto de celular em clínica"]
    assert [i for i, _ in scegli_candidati_scippo_telefono(righe)] == ["aaa", "bbb", "fff"]


def test_rissa_piu_warning_diventa_rissa_con_arma():
    """Regola D6 chiesta da Marco: se il modello vede violenza e il WARNING ARMA è scattato → violenza con arma."""
    probabilita = np.array([[0.1, 0.8, 0.05, 0.05],    # violenza + warning → con arma (giusto)
                            [0.1, 0.8, 0.05, 0.05],    # violenza senza warning → resta violenza (giusto)
                            [0.9, 0.05, 0.03, 0.02]])  # normale + warning → resta normale (la regola non inventa risse)
    etichette = np.array([2, 1, 0])
    risultato = valuta_sistema(probabilita, etichette, np.array([True, False, True]))
    assert risultato["solo_modello"]["recall"]["violenza_con_arma"] == 0.0
    assert risultato["modello_piu_regola"]["recall"]["violenza_con_arma"] == 1.0
    assert risultato["modello_piu_regola"]["accuratezza"] == 1.0 and risultato["warning_su_finestre_senza_arma"] == 0.5


# ==============================================================================================================
# ingrandimento sulle mani (D44)
# ==============================================================================================================

def test_ritaglio_quadrato_fuori_dal_bordo_resta_nero():
    immagine = np.full((100, 200, 3), 7, np.uint8)
    ritaglio = ritaglio_quadrato(immagine, 0, 0, 40)              # centrato nell'angolo: 3/4 fuori
    assert ritaglio.shape == (40, 40, 3)
    assert (ritaglio[20:, 20:] == 7).all() and (ritaglio[:20] == 0).all() and (ritaglio[:, :20] == 0).all()


def test_mani_oltre_il_polso_e_lato_proporzionale_all_altezza():
    keypoint = scheletro_in_piedi(100, 400, 200)                  # persona alta 200 px
    box = np.array([80, 200, 120, 400])
    mani = mani_di_una_persona(box, keypoint)
    assert len(mani) == 2 and all(lato == 0.4 * 200 for _, _, lato in mani)
    polso, gomito = keypoint[10, :2], keypoint[8, :2]
    assert np.allclose(mani[1][:2], polso + 0.3 * (polso - gomito))
    keypoint[9:11, 2] = 0.1                                        # polsi non visti → nessun ritaglio
    assert mani_di_una_persona(box, keypoint) == []


def test_precisione_media():
    assert precisione_media(np.array([0.9, 0.8, 0.1]), np.array([True, True, False])) == 1.0
    assert precisione_media(np.array([0.9, 0.8, 0.1]), np.array([False, False, True])) == 1 / 3


class RisultatoPoseFinto:
    def __init__(self, persone):
        import torch as t
        self.boxes = type("B", (), {"xyxy": t.tensor([b for b, _ in persone], dtype=t.float32).reshape(-1, 4),
                                    "__len__": lambda _: len(persone)})()
        keypoint = np.array([k for _, k in persone], np.float32).reshape(-1, 17, 3)
        self.keypoints = type("K", (), {"xy": t.from_numpy(keypoint[..., :2]), "conf": t.from_numpy(keypoint[..., 2])})()


def test_ritagli_per_addestramento_positivi_e_mani_vuote(tmp_path):
    import cv2
    unita = tmp_path / "unito"
    for split in ("train", "test"):
        (unita / "images" / split).mkdir(parents=True)
        (unita / "labels" / split).mkdir(parents=True)
        cv2.imwrite(str(unita / "images" / split / "con_coltello.jpg"), np.zeros((480, 640, 3), np.uint8))
        (unita / "labels" / split / "con_coltello.txt").write_text("1 0.5 0.5 0.02 0.05\n")
        cv2.imwrite(str(unita / "images" / split / "vuota.jpg"), np.zeros((480, 640, 3), np.uint8))
        (unita / "labels" / split / "vuota.txt").write_text("")

    class PoseFinta:   # una persona in piedi in ogni immagine senza armi
        def predict(self, immagini, **_):
            return [RisultatoPoseFinto([([280, 200, 360, 400], scheletro_in_piedi(320, 400, 200))]) for _ in immagini]

    conteggio = ritagli_per_addestramento(unita, tmp_path / "ritagli", PoseFinta(), ritagli_per_arma=2)
    assert conteggio[("train", "knife")] == 2 and conteggio[("train", "niente")] == 2   # 2 copie, 2 mani
    assert conteggio[("test", "knife")] == 2
    ritaglio = cv2.imread(str(next((tmp_path / "ritagli" / "train" / "knife").glob("*.jpg"))))
    assert ritaglio.shape == (224, 224, 3)


def test_probabilita_armi_in_mano_massimo_tra_le_due_mani():
    import torch as t

    class ClassificatoreFinto:
        names = {0: "knife", 1: "niente", 2: "pistol", 3: "rifle"}

        def __init__(self):
            self.chiamate = 0

        def predict(self, ritagli, **_):   # prima mano: coltello 0,9; seconda mano: niente
            valori = [[0.9, 0.05, 0.03, 0.02], [0.01, 0.97, 0.01, 0.01]][:len(ritagli)]
            return [type("R", (), {"probs": type("P", (), {"data": t.tensor(v)})()})() for v in valori]

    keypoint = scheletro_in_piedi(320, 400, 200)[None]
    probabilita = probabilita_armi_in_mano(ClassificatoreFinto(), np.zeros((480, 640, 3), np.uint8),
                                           np.array([[280, 200, 360, 400]]), keypoint)
    assert probabilita.shape == (1, 3) and np.allclose(probabilita[0], [0.03, 0.9, 0.02])   # pistola, coltello, fucile
    assert probabilita_armi_in_mano(ClassificatoreFinto(), np.zeros((480, 640, 3), np.uint8),
                                    np.empty((0, 4)), np.empty((0, 17, 3))).shape == (0, 3)


class ClassificatoreManiFinto:
    """Coltello sicuro in ogni ritaglio con qualcosa di chiaro dentro, altrimenti niente."""
    names = {0: "knife", 1: "niente", 2: "pistol", 3: "rifle"}

    def predict(self, ritagli, **_):
        import torch as t
        valori = [[0.9, 0.1, 0, 0] if r.mean() > 10 else [0, 1.0, 0, 0] for r in ritagli]
        return [type("R", (), {"probs": type("P", (), {"data": t.tensor(v)})()})() for v in valori]


def test_aggiungi_armi_anche_le_mani(clip_a_30_fps):
    """Passaggio sui video con il classificatore delle mani: una probabilità per ogni riga-persona del .npz."""
    class DetectorVuoto:
        names = {0: "pistol", 1: "knife", 2: "rifle"}

        def predict(self, lotto, **_):
            import torch as t
            vuoto = type("B", (), {"xyxy": t.empty(0, 4), "cls": t.empty(0), "conf": t.empty(0), "__len__": lambda _: 0})
            return [type("R", (), {"boxes": vuoto()})() for _ in lotto]

    from sicurezza.visione import armi_di_un_video
    persone = {"persone_frame": np.array([0, 0, 5, 99], np.int32),
               "persone_box": np.tile(np.array([[40, 20, 120, 110]], np.float32), (4, 1)),
               "persone_keypoint": np.stack([scheletro_in_piedi(80, 110, 90)] * 4)}
    video = VideoDaEstrarre("prova/clip", str(clip_a_30_fps))
    uscita = armi_di_un_video(DetectorVuoto(), video, "cpu", modello_mani=ClassificatoreManiFinto(), persone=persone)
    assert uscita["persone_mani"].shape == (4, 3) and len(uscita["armi_frame"]) == 0
    assert np.allclose(uscita["persone_mani"][:, 1], 0.9, atol=1e-3)    # colonna coltello, una per riga-persona


def test_warning_dalle_mani(tmp_path):
    dati = dict(estrazione_finta(numero_frame=40, numero_persone=2, con_arma=False))
    dati["persone_mani"] = np.zeros((len(dati["persone_frame"]), 3), np.float16)
    dati["persone_mani"][dati["persone_frame"] < 8, 1] = 0.85           # coltello in mano nei primi 8 frame
    timeline = timeline_da_dati(dati)
    assert timeline.mani_per_frame.shape == (40, 3) and timeline.mani_per_frame[:8, 1].min() > 0.84
    assert frame_con_arma(timeline, 0, soglia_confidenza=0.5, vicinanza="mani") == 8
    assert frame_con_arma(timeline, 0, soglia_confidenza=0.5, vicinanza="polso:0.2") == 0        # il detector non vede
    assert frame_con_arma(timeline, 0, soglia_confidenza=0.5, vicinanza="polso:0.2+mani") == 8
    senza_mani = timeline_da_dati({k: v for k, v in dati.items() if k != "persone_mani"})
    assert senza_mani.mani_per_frame is None and frame_con_arma(senza_mani, 0, vicinanza="mani") == 0


def test_normali_folla_senza_hieve_se_il_server_non_risponde(tmp_path, monkeypatch):
    def server_giu(url, destinazione):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlretrieve", server_giu)
    (tmp_path / "input").mkdir()
    conteggio = prepara_normali(tmp_path / "input", tmp_path / "uscita", scarica_hieve=True)
    assert conteggio["hieve"] == 0


def test_normali_folla_preparati_e_letti(tmp_path):
    """D41/D46: sequenze di frame → mp4, solo i video normali scelti, una scena = un gruppo, tutto classe normale."""
    import cv2
    ingresso = tmp_path / "input"
    sequenza = ingresso / "mot-20" / "MOT20" / "train" / "MOT20-01"
    (sequenza / "img1").mkdir(parents=True)
    (sequenza / "seqinfo.ini").write_text("[Sequence]\nframeRate=25\n")
    for numero in range(1, 26):
        # larghezza dispari come MOT20-04 (1545×1080): libx264 con yuv420p vuole lati pari
        cv2.imwrite(str(sequenza / "img1" / f"{numero:06d}.jpg"), np.full((48, 65, 3), numero * 9, np.uint8))
    crea_file(ingresso / "shanghaitech-campus", "train/01_001.avi", "train/01_002.avi", "test/01_0014.avi")
    crea_file(ingresso / "avenue", "Avenue Dataset/training_videos/01.avi")
    for nome in ("N_0_0_0_1_0.mp4", "F_1_0_0_0_0.mp4"):
        cartella = ingresso / "ubi" / "UBI_FIGHTS" / ("normal" if nome.startswith("N") else "fight")
        cartella.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=25:duration=1",
                        "-pix_fmt", "yuv420p", str(cartella / nome)], check=True)
    conteggio = prepara_normali(ingresso, tmp_path / "uscita")
    assert conteggio == {"mot20": 1, "shanghaitech": 2, "ubi_fights": 1, "avenue": 1}   # 01_0014: è un video di test
    info = leggi_info_video(str(tmp_path / "uscita" / "normali_folla" / "mot20" / "MOT20-01.mp4"))
    assert info.durata_secondi == 1.0 and (info.larghezza, info.altezza) == (66, 48)   # 1 px nero a destra
    clip = elenca_clip_normali_folla(tmp_path / "uscita")
    assert {c.classe_intera for c in clip} == {"normale"} and len(clip) == 5
    gruppi = {c.id_clip: c.gruppo for c in clip}
    assert gruppi["normali_folla/shanghaitech/01_001"] == gruppi["normali_folla/shanghaitech/01_002"] == "folla_shanghaitech/01"
    assert {c.sorgente for c in clip} == {"folla_mot20", "folla_shanghaitech", "folla_ubi_fights", "folla_avenue"}

