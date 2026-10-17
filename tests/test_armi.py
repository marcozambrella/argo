import yaml

from sicurezza.armi.unisci_dataset import unisci


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
