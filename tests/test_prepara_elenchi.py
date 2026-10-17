import json

from sicurezza.estrazione import leggi_elenco
from sicurezza.etichette import leggi_annotazioni
from sicurezza.prepara_elenchi import prepara


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
