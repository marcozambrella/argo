"""RWF-2000 (copia Kaggle vulamnguyen/rwf2000): clip di 5 s già tagliate. Fight → violenza, NonFight → normale."""

from pathlib import Path

from . import ClipAnnotata

CLASSE_PER_CARTELLA = {"Fight": "violenza", "NonFight": "normale"}


def elenca_clip(radice_dataset: str | Path) -> list[ClipAnnotata]:
    clip = []
    for percorso in sorted(Path(radice_dataset).glob("**/RWF-2000/*/*/*.avi")):
        cartella_classe, cartella_split = percorso.parent.name, percorso.parent.parent.name
        if cartella_classe not in CLASSE_PER_CARTELLA:
            continue
        # i nomi sono <id YouTube>_<n>: clip dello stesso video originale devono finire nello stesso split
        video_originale = percorso.stem.rsplit("_", 1)[0]
        clip.append(ClipAnnotata(
            id_clip=f"rwf2000/{cartella_split}/{cartella_classe}/{percorso.stem}", sorgente="rwf2000",
            gruppo=f"rwf2000/{video_originale}", percorso_video=str(percorso),
            classe_intera=CLASSE_PER_CARTELLA[cartella_classe], split_originale=cartella_split))
    return clip
