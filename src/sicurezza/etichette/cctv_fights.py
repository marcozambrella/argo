"""CCTV-Fights (copia Kaggle mohamedtariki/vsd-cctv-fights, annotazioni originali NTU in groundtruth.json).

Formato stile ActivityNet: database → id video → annotations [{segment: [inizio, fine] secondi, label: "Fight"}].
Intervalli di rissa → violenza; resto del video → normale. Ogni video ha anche "source": CCTV, Mobile, Car, Other.
Il groundtruth NON è nel repo (accordo di licenza NTU): si legge dal dataset Kaggle.
"""

import json
from pathlib import Path

from . import ClipAnnotata


def elenca_clip(radice_dataset: str | Path, percorso_groundtruth: str | Path | None = None,
                fonti_ammesse: tuple[str, ...] | None = None) -> list[ClipAnnotata]:
    """`fonti_ammesse` = None tiene tutto; ("CCTV",) tiene solo le telecamere di sorveglianza (DA DECIDERE)."""
    radice_dataset = Path(radice_dataset)
    percorso_groundtruth = percorso_groundtruth or next(radice_dataset.glob("**/groundtruth.json"))
    database = json.loads(Path(percorso_groundtruth).read_text())["database"]
    percorso_per_id = {percorso.stem: percorso for percorso in radice_dataset.glob("**/fight_*.*")
                       if percorso.suffix in {".mp4", ".webm", ".avi", ".mkv"}}
    clip = []
    for id_video, dati in sorted(database.items()):
        if fonti_ammesse is not None and dati.get("source") not in fonti_ammesse:
            continue
        if id_video not in percorso_per_id:
            continue
        eventi = [(float(a["segment"][0]), float(a["segment"][1]), "violenza") for a in dati["annotations"]]
        clip.append(ClipAnnotata(
            id_clip=f"cctv_fights/{id_video}", sorgente="cctv_fights", gruppo=f"cctv_fights/{id_video}",
            percorso_video=str(percorso_per_id[id_video]), eventi_secondi=eventi, classe_sfondo="normale",
            split_originale=dati.get("subset"), note=f"fonte={dati.get('source')}"))
    return clip
