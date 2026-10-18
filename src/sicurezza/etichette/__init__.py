"""Etichette per sorgente: ogni modulo trasforma un dataset in una lista di ClipAnnotata.

Una ClipAnnotata dice cosa estrarre (video, eventuale segmento) e come etichettarne le finestre.
L'estrazione non dipende dalle etichette: si possono cambiare le regole senza rifare l'estrazione.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import INDICE_CLASSE
from ..estrazione import VideoDaEstrarre
from ..finestre import Evento, secondi_in_frame


@dataclass
class ClipAnnotata:
    id_clip: str                    # univoco; diventa anche il nome del .npz
    sorgente: str                   # rwf2000, cctv_fights, scvd, ucf_uca, chain_snatching
    gruppo: str                     # video originale: tutte le clip dello stesso gruppo stanno nello stesso split (P4)
    percorso_video: str
    inizio_secondi: float | None = None
    fine_secondi: float | None = None
    eventi_secondi: list[tuple[float, float, str]] = field(default_factory=list)  # relativi all'inizio della clip
    classe_intera: str | None = None    # tutta la clip ha questa classe (dataset già tagliati)
    classe_sfondo: str | None = None    # classe delle finestre senza eventi (None = scartarle)
    split_originale: str | None = None
    telecamera_mobile: bool = False     # D20: accende la compensazione del movimento nel tracker
    note: str = ""

    def da_estrarre(self) -> VideoDaEstrarre:
        return VideoDaEstrarre(self.id_clip, self.percorso_video, self.inizio_secondi, self.fine_secondi,
                               self.telecamera_mobile)

    def eventi_in_frame(self, numero_frame: int) -> list[Evento]:
        if self.classe_intera is not None:
            return [(0, numero_frame, INDICE_CLASSE[self.classe_intera])]
        return [(*secondi_in_frame(inizio, fine), INDICE_CLASSE[classe]) for inizio, fine, classe in self.eventi_secondi]

    def indice_classe_sfondo(self) -> int | None:
        return None if self.classe_sfondo is None else INDICE_CLASSE[self.classe_sfondo]


def scrivi_annotazioni(clip: list[ClipAnnotata], percorso: str | Path) -> None:
    with open(percorso, "w") as file_annotazioni:
        for singola_clip in clip:
            file_annotazioni.write(json.dumps(asdict(singola_clip)) + "\n")


def leggi_annotazioni(percorso: str | Path) -> list[ClipAnnotata]:
    with open(percorso) as file_annotazioni:
        clip = [ClipAnnotata(**json.loads(riga)) for riga in file_annotazioni if riga.strip()]
    for singola_clip in clip:
        singola_clip.eventi_secondi = [tuple(evento) for evento in singola_clip.eventi_secondi]
    return clip
