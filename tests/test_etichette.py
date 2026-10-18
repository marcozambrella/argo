import json

from sicurezza.config import INDICE_CLASSE
from sicurezza.etichette import leggi_annotazioni, scrivi_annotazioni
from sicurezza.etichette import cctv_fights, rwf2000, scvd, ucf_uca


def crea_file(radice, *percorsi):
    for percorso in percorsi:
        (radice / percorso).parent.mkdir(parents=True, exist_ok=True)
        (radice / percorso).write_bytes(b"")


def test_rwf2000_gruppo_per_video_originale(tmp_path):
    crea_file(tmp_path, "RWF-2000/train/Fight/abc_0.avi", "RWF-2000/train/Fight/abc_1.avi",
              "RWF-2000/val/NonFight/x_y_3.avi")
    clip = rwf2000.elenca_clip(tmp_path)
    assert [c.gruppo for c in clip] == ["rwf2000/abc", "rwf2000/abc", "rwf2000/x_y"]
    assert [c.classe_intera for c in clip] == ["violenza", "violenza", "normale"]


def test_scvd_solo_versione_video_interi(tmp_path):
    crea_file(tmp_path, "SCVD/SCVD_converted/Train/Weaponized/w1.avi",
              "SCVD/SCVD_converted_sec_split/Train/Weaponized/w1_0.avi")
    clip = scvd.elenca_clip(tmp_path)
    assert len(clip) == 1 and clip[0].classe_intera == "violenza_con_arma"


def test_cctv_fights_eventi_e_fonti(tmp_path):
    crea_file(tmp_path, "parte/fight_0001.mp4", "parte/fight_0002.webm")
    database = {"fight_0001": {"subset": "training", "source": "CCTV", "annotations": [
        {"segment": [1.0, 3.0], "label": "Fight"}]},
        "fight_0002": {"subset": "testing", "source": "Mobile", "annotations": []}}
    (tmp_path / "groundtruth.json").write_text(json.dumps({"database": database}))
    tutte = cctv_fights.elenca_clip(tmp_path)
    assert len(tutte) == 2 and tutte[0].eventi_in_frame(500) == [(25, 75, INDICE_CLASSE["violenza"])]
    assert len(cctv_fights.elenca_clip(tmp_path, fonti_ammesse=("CCTV",))) == 1


def test_ucf_segmenti_uniti_con_margine(tmp_path):
    crea_file(tmp_path, "Anomaly-Videos/Robbery/Robbery001_x264.mp4")
    (tmp_path / "uca").mkdir()
    (tmp_path / "uca" / "UCFCrime_Train.json").write_text(json.dumps({"Robbery001_x264": {
        "duration": 100.0, "timestamps": [[10.0, 12.0], [13.0, 15.0], [50.0, 52.0], [70, 80]],
        "sentences": ["The man took out a gun and pointed it at the clerk", "The man punched the clerk",
                      "A man snatched the phone of the woman", "The clerk walked to the door"]}}))
    clip = ucf_uca.elenca_clip(tmp_path, cartella_uca=tmp_path / "uca")
    assert [(c.inizio_secondi, c.fine_secondi) for c in clip] == [(8.0, 17.0), (48.0, 54.0)]
    assert clip[0].eventi_secondi == [(2.0, 4.0, "violenza_con_arma"), (5.0, 7.0, "violenza")]
    assert clip[1].eventi_secondi == [(2.0, 4.0, "scippo")]
    assert all(c.classe_sfondo is None for c in clip)


def test_annotazioni_andata_e_ritorno(tmp_path):
    crea_file(tmp_path, "RWF-2000/train/Fight/abc_0.avi")
    clip = rwf2000.elenca_clip(tmp_path)
    scrivi_annotazioni(clip, tmp_path / "a.jsonl")
    assert leggi_annotazioni(tmp_path / "a.jsonl") == clip


def test_regole_uca_casi_tipici():
    giudica = ucf_uca.classe_da_frase
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
