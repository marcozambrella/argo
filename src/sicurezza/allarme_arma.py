"""WARNING ARMA (D6) e regola finale "violenza + arma confermata = violenza con arma".

Soglie PROVVISORIE dal piano (≥ 0,5 di confidenza in ≥ 5 frame su 40, vicino a una persona): da tarare sul
validation set. "Vicino" = centro dell'arma dentro il box della persona allargato di MARGINE_VICINANZA (PROVVISORIO).
"""

import numpy as np

from .caratteristiche import TimelineVideo
from .config import CLASSI, FRAME_PER_FINESTRA

SOGLIA_CONFIDENZA_ARMA = 0.5
FRAME_MINIMI_CON_ARMA = 5
MARGINE_VICINANZA = 0.5


def arma_vicina_a_persona(box_arma: np.ndarray, box_persone: np.ndarray, margine: float = MARGINE_VICINANZA) -> bool:
    if len(box_persone) == 0:
        return False
    centro_x, centro_y = (box_arma[0] + box_arma[2]) / 2, (box_arma[1] + box_arma[3]) / 2
    larghezza = box_persone[:, 2] - box_persone[:, 0]
    altezza = box_persone[:, 3] - box_persone[:, 1]
    dentro = ((centro_x >= box_persone[:, 0] - margine * larghezza) & (centro_x <= box_persone[:, 2] + margine * larghezza)
              & (centro_y >= box_persone[:, 1] - margine * altezza) & (centro_y <= box_persone[:, 3] + margine * altezza))
    return bool(dentro.any())


def frame_con_arma(timeline: TimelineVideo, inizio_finestra: int, frame_per_finestra: int = FRAME_PER_FINESTRA,
                   soglia_confidenza: float = SOGLIA_CONFIDENZA_ARMA, margine: float = MARGINE_VICINANZA) -> int:
    """Quanti frame della finestra hanno almeno un'arma sicura vicino a una persona tracciata."""
    frame_validi = set()
    fine_finestra = inizio_finestra + frame_per_finestra
    candidati = np.flatnonzero((timeline.armi_frame >= inizio_finestra) & (timeline.armi_frame < fine_finestra)
                               & (timeline.armi_confidenza >= soglia_confidenza))
    for indice in candidati:
        frame = int(timeline.armi_frame[indice])
        if frame in frame_validi:
            continue
        box_persone = np.array([box[frame] for box in timeline.box_per_traccia.values() if not np.isnan(box[frame, 0])])
        if arma_vicina_a_persona(timeline.armi_box[indice], box_persone.reshape(-1, 4), margine):
            frame_validi.add(frame)
    return len(frame_validi)


def warning_arma(timeline: TimelineVideo, inizio_finestra: int, frame_minimi: int = FRAME_MINIMI_CON_ARMA,
                 **opzioni) -> bool:
    return frame_con_arma(timeline, inizio_finestra, **opzioni) >= frame_minimi


def classe_finale(probabilita: np.ndarray, arma_confermata: bool) -> str:
    """Classe più probabile, con la regola di D6: violenza + arma confermata → violenza con arma."""
    classe = CLASSI[int(np.argmax(probabilita))]
    if classe == "violenza" and arma_confermata:
        return "violenza_con_arma"
    return classe

