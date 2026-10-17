"""SCVD (Kaggle toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd).

Si usa la versione SCVD_converted (video interi); SCVD_converted_sec_split contiene gli stessi video tagliati a
pezzi e userla insieme duplicherebbe i dati.
"""

from pathlib import Path

from . import ClipAnnotata

CLASSE_PER_CARTELLA = {"Normal": "normale", "Violence": "violenza", "Weaponized": "violenza_con_arma"}


def elenca_clip(radice_dataset: str | Path) -> list[ClipAnnotata]:
    clip = []
    for percorso in sorted(Path(radice_dataset).glob("**/SCVD_converted/*/*/*.avi")):
        cartella_classe, cartella_split = percorso.parent.name, percorso.parent.parent.name
        if cartella_classe not in CLASSE_PER_CARTELLA:
            continue
        clip.append(ClipAnnotata(
            id_clip=f"scvd/{cartella_split}/{cartella_classe}/{percorso.stem}", sorgente="scvd",
            gruppo=f"scvd/{percorso.stem}", percorso_video=str(percorso),
            classe_intera=CLASSE_PER_CARTELLA[cartella_classe], split_originale=cartella_split))
    return clip
