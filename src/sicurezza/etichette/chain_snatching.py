"""Chain-snatching (Kaggle aniketsahu09/chain-snatching-cctv-dataset): 170 video di scippi di collane, P1.

ATTENZIONE: il dataset NON ha i tempi dell'evento. Etichettare tutta la clip come "scippo" è PROVVISORIO e
rumoroso (lo scippo dura pochi secondi): come etichettarlo è DA DECIDERE con Marco (vedi DIARIO.md).
"""

from pathlib import Path

from . import ClipAnnotata


def elenca_clip(radice_dataset: str | Path) -> list[ClipAnnotata]:
    return [ClipAnnotata(
        id_clip=f"chain_snatching/{percorso.stem}", sorgente="chain_snatching",
        gruppo=f"chain_snatching/{percorso.stem}", percorso_video=str(percorso), classe_intera="scippo",
        note="PROVVISORIO: tutta la clip = scippo, mancano i tempi dell'evento")
        for percorso in sorted(Path(radice_dataset).glob("**/Chain_Snatching_Videos/*.mp4"))]
