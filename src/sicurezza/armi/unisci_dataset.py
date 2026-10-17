"""Unisce più export Roboflow in formato YOLO in un unico dataset con classi pistol / knife / rifle (P2).

Le classi non previste (grenade, helmet, ...) vengono tolte dalle etichette, ma le immagini restano: diventano
esempi "senza arma" o con meno armi, utili contro i falsi positivi.

Uso:
    python -m sicurezza.armi.unisci_dataset --sorgenti export_master/ export_prospace/ --uscita armi_unite/
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml

CLASSI_ARMA = ("pistol", "knife", "rifle")
SINONIMI = {"pistol": "pistol", "handgun": "pistol", "gun": "pistol", "knife": "knife", "rifle": "rifle"}
CARTELLE_SPLIT = {"train": "train", "valid": "val", "val": "val", "test": "test"}
ESTENSIONI_IMMAGINE = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def nomi_classi_export(cartella_export: Path) -> list[str]:
    configurazione = yaml.safe_load((cartella_export / "data.yaml").read_text())
    nomi = configurazione["names"]
    return [nomi[indice] for indice in sorted(nomi)] if isinstance(nomi, dict) else list(nomi)


def unisci(cartelle_export: list[Path], cartella_uscita: Path) -> Counter:
    conteggio = Counter()
    for numero_sorgente, cartella_export in enumerate(cartelle_export):
        nomi = nomi_classi_export(cartella_export)
        nuovo_indice = {indice: CLASSI_ARMA.index(SINONIMI[nome.lower()])
                        for indice, nome in enumerate(nomi) if nome.lower() in SINONIMI}
        for cartella_split_export, split in CARTELLE_SPLIT.items():
            cartella_immagini = cartella_export / cartella_split_export / "images"
            if not cartella_immagini.exists():
                continue
            (cartella_uscita / "images" / split).mkdir(parents=True, exist_ok=True)
            (cartella_uscita / "labels" / split).mkdir(parents=True, exist_ok=True)
            for immagine in sorted(cartella_immagini.iterdir()):
                if immagine.suffix.lower() not in ESTENSIONI_IMMAGINE:
                    continue
                nome_nuovo = f"s{numero_sorgente}_{immagine.name}"   # niente collisioni tra export diversi
                shutil.copy2(immagine, cartella_uscita / "images" / split / nome_nuovo)
                file_etichette = cartella_export / cartella_split_export / "labels" / (immagine.stem + ".txt")
                righe_nuove = []
                if file_etichette.exists():
                    for riga in file_etichette.read_text().splitlines():
                        valori = riga.split()
                        if valori and int(valori[0]) in nuovo_indice:
                            righe_nuove.append(" ".join([str(nuovo_indice[int(valori[0])]), *valori[1:]]))
                            conteggio[(split, CLASSI_ARMA[nuovo_indice[int(valori[0])]])] += 1
                        elif valori:
                            conteggio[(split, f"tolta:{nomi[int(valori[0])]}")] += 1
                (cartella_uscita / "labels" / split / (Path(nome_nuovo).stem + ".txt")).write_text(
                    "\n".join(righe_nuove) + ("\n" if righe_nuove else ""))
                conteggio[(split, "immagini")] += 1
    split_presenti = [split for split in ("train", "val", "test") if (cartella_uscita / "images" / split).exists()]
    (cartella_uscita / "data.yaml").write_text(yaml.safe_dump({
        "path": str(cartella_uscita.resolve()), **{split: f"images/{split}" for split in split_presenti},
        "names": dict(enumerate(CLASSI_ARMA))}, sort_keys=False))
    return conteggio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sorgenti", nargs="+", required=True)
    parser.add_argument("--uscita", required=True)
    argomenti = parser.parse_args()
    for (split, voce), numero in sorted(unisci([Path(s) for s in argomenti.sorgenti], Path(argomenti.uscita)).items()):
        print(f"{split:6s} {voce:20s} {numero}")


if __name__ == "__main__":
    main()
