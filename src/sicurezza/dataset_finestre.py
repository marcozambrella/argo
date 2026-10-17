"""Dalle estrazioni (.npz) e dalle annotazioni (ClipAnnotata) al dataset di finestre etichettate.

Uso:
    python -m sicurezza.dataset_finestre --annotazioni annotazioni.jsonl --estratti estratti/ --uscita finestre.npz
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .caratteristiche import carica_timeline, costruisci_finestra
from .config import CLASSI
from .estrazione import nome_file_estratto
from .etichette import ClipAnnotata, leggi_annotazioni
from .finestre import etichetta_finestra, inizi_finestre


def finestre_di_una_clip(clip: ClipAnnotata, percorso_npz: Path):
    timeline = carica_timeline(percorso_npz)
    eventi = clip.eventi_in_frame(timeline.numero_frame)
    for inizio in inizi_finestre(timeline.numero_frame):
        classe = etichetta_finestra(int(inizio), eventi, clip.indice_classe_sfondo())
        if classe is not None:
            yield int(inizio), classe, costruisci_finestra(timeline, int(inizio))


def costruisci_dataset(clip: list[ClipAnnotata], cartelle_estratti: list[Path]) -> dict[str, np.ndarray]:
    percorso_per_nome = {file.name: file for cartella in cartelle_estratti for file in Path(cartella).glob("*.npz")}
    nodi, maschere, classi_arma, etichette, gruppi, sorgenti, id_clip, inizi = [], [], [], [], [], [], [], []
    clip_mancanti = 0
    for singola_clip in clip:
        percorso_npz = percorso_per_nome.get(nome_file_estratto(singola_clip.id_clip))
        if percorso_npz is None:
            clip_mancanti += 1
            continue
        for inizio, classe, (nodi_finestra, maschera, classe_arma) in finestre_di_una_clip(singola_clip, percorso_npz):
            nodi.append(nodi_finestra.astype(np.float16))
            maschere.append(maschera)
            classi_arma.append(classe_arma)
            etichette.append(classe)
            gruppi.append(singola_clip.gruppo)
            sorgenti.append(singola_clip.sorgente)
            id_clip.append(singola_clip.id_clip)
            inizi.append(inizio)
    if clip_mancanti:
        print(f"ATTENZIONE: {clip_mancanti} clip senza .npz (non ancora estratte o errori di estrazione)")
    return {"nodi": np.stack(nodi), "maschera": np.stack(maschere), "classe_arma": np.stack(classi_arma),
            "etichetta": np.array(etichette, np.int64), "gruppo": np.array(gruppi), "sorgente": np.array(sorgenti),
            "id_clip": np.array(id_clip), "inizio_finestra": np.array(inizi, np.int32)}


def riepilogo(dataset: dict[str, np.ndarray]) -> str:
    righe = [f"{len(dataset['etichetta'])} finestre"]
    conteggio = Counter(zip(dataset["sorgente"].tolist(), dataset["etichetta"].tolist()))
    for (sorgente, classe), numero in sorted(conteggio.items()):
        righe.append(f"  {sorgente:16s} {CLASSI[classe]:18s} {numero}")
    return "\n".join(righe)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotazioni", required=True)
    parser.add_argument("--estratti", nargs="+", required=True)
    parser.add_argument("--uscita", required=True)
    argomenti = parser.parse_args()
    dataset = costruisci_dataset(leggi_annotazioni(argomenti.annotazioni), [Path(c) for c in argomenti.estratti])
    np.savez(argomenti.uscita, **dataset)
    print(riepilogo(dataset))


if __name__ == "__main__":
    main()
