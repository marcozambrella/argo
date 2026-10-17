"""Da un .npz estratto alle finestre-grafo: [40 frame × 70 nodi × 5 caratteristiche] (D7, D8 nel diario).

Nodi di ogni frame: 4 persone × 17 keypoint, poi 2 nodi arma.
Caratteristiche di ogni nodo: x, y relative al centro della scena e divise per l'altezza mediana delle persone
(così distanza e zoom della telecamera contano poco), confidenza, velocità x, velocità y.
"""

from dataclasses import dataclass

import numpy as np

from .config import FRAME_PER_FINESTRA
from .scheletro import NUMERO_KEYPOINT

NUMERO_PERSONE = 4                      # K di D8
NUMERO_ARMI = 2
NODI_PERSONE = NUMERO_PERSONE * NUMERO_KEYPOINT
NODI_PER_FRAME = NODI_PERSONE + NUMERO_ARMI
CARATTERISTICHE_PER_NODO = 5
SOGLIA_VISIBILITA_KEYPOINT = 0.1        # sotto questa confidenza YOLO mette il keypoint a (0, 0): lo si azzera


@dataclass
class TimelineVideo:
    """Dati di un .npz riorganizzati per accesso rapido a finestre."""
    numero_frame: int
    larghezza: int
    altezza: int
    keypoint_per_traccia: dict[int, np.ndarray]   # id traccia → (numero_frame, 17, 3), NaN dove la persona manca
    box_per_traccia: dict[int, np.ndarray]        # id traccia → (numero_frame, 4), NaN dove la persona manca
    armi_frame: np.ndarray
    armi_box: np.ndarray
    armi_classe: np.ndarray
    armi_confidenza: np.ndarray


def carica_timeline(percorso_npz) -> TimelineVideo:
    dati = np.load(percorso_npz)
    numero_frame = int(dati["numero_frame"])
    keypoint_per_traccia, box_per_traccia = {}, {}
    for id_traccia in np.unique(dati["persone_id_traccia"]):
        righe = dati["persone_id_traccia"] == id_traccia
        frame_traccia = dati["persone_frame"][righe]
        keypoint = np.full((numero_frame, NUMERO_KEYPOINT, 3), np.nan, np.float32)
        box = np.full((numero_frame, 4), np.nan, np.float32)
        keypoint[frame_traccia] = dati["persone_keypoint"][righe]
        box[frame_traccia] = dati["persone_box"][righe]
        keypoint_per_traccia[int(id_traccia)] = keypoint
        box_per_traccia[int(id_traccia)] = box
    return TimelineVideo(numero_frame, int(dati["larghezza"]), int(dati["altezza"]), keypoint_per_traccia,
                         box_per_traccia, dati["armi_frame"], dati["armi_box"], dati["armi_classe"],
                         dati["armi_confidenza"].astype(np.float32))


def punteggio_presenza_e_movimento(keypoint: np.ndarray, box: np.ndarray) -> float:
    """Criterio PROVVISORIO per scegliere le 4 persone (DA DECIDERE nel diario):
    frame in cui la persona c'è × (1 + movimento medio dei keypoint in altezze-corpo per frame)."""
    presente = ~np.isnan(box[:, 0])
    if presente.sum() < 2:
        return float(presente.sum())
    altezza_corpo = np.nanmedian(box[:, 3] - box[:, 1]) + 1e-6
    spostamenti = np.linalg.norm(np.diff(keypoint[:, :, :2], axis=0), axis=-1) / altezza_corpo
    movimento_medio = float(np.nanmean(spostamenti)) if np.isfinite(spostamenti).any() else 0.0
    return float(presente.sum()) * (1.0 + movimento_medio)


CRITERI_SCELTA_PERSONE = {"presenza_e_movimento": punteggio_presenza_e_movimento}


def costruisci_finestra(timeline: TimelineVideo, inizio_finestra: int,
                        frame_per_finestra: int = FRAME_PER_FINESTRA,
                        criterio_scelta_persone: str = "presenza_e_movimento"):
    """Ritorna (nodi [T, 70, 5] float32, maschera [T, 70] bool, classe_arma [T, 2] int8 con 0 = nessuna arma)."""
    fine_finestra = inizio_finestra + frame_per_finestra
    punteggio = CRITERI_SCELTA_PERSONE[criterio_scelta_persone]

    tracce_in_finestra = []
    for id_traccia, box in timeline.box_per_traccia.items():
        box_finestra = box[inizio_finestra:fine_finestra]
        if not np.isnan(box_finestra[:, 0]).all():
            keypoint_finestra = timeline.keypoint_per_traccia[id_traccia][inizio_finestra:fine_finestra]
            tracce_in_finestra.append((punteggio(keypoint_finestra, box_finestra), keypoint_finestra, box_finestra))
    tracce_scelte = sorted(tracce_in_finestra, key=lambda traccia: -traccia[0])[:NUMERO_PERSONE]

    # scala e centro della scena: altezza mediana delle persone scelte, centro medio dei loro box
    if tracce_scelte:
        tutti_i_box = np.concatenate([box for _, _, box in tracce_scelte])
        altezza_riferimento = float(np.nanmedian(tutti_i_box[:, 3] - tutti_i_box[:, 1]))
        centro_scena = np.array([np.nanmean((tutti_i_box[:, 0] + tutti_i_box[:, 2]) / 2),
                                 np.nanmean((tutti_i_box[:, 1] + tutti_i_box[:, 3]) / 2)], np.float32)
    else:
        altezza_riferimento, centro_scena = 0.0, np.array([timeline.larghezza / 2, timeline.altezza / 2], np.float32)
    if not np.isfinite(altezza_riferimento) or altezza_riferimento < 1:
        altezza_riferimento = 0.3 * timeline.altezza

    def normalizza(coordinate):
        return (coordinate - centro_scena) / altezza_riferimento

    nodi = np.zeros((frame_per_finestra, NODI_PER_FRAME, CARATTERISTICHE_PER_NODO), np.float32)
    maschera = np.zeros((frame_per_finestra, NODI_PER_FRAME), bool)
    for posto_persona, (_, keypoint_finestra, _) in enumerate(tracce_scelte):
        nodi_persona = slice(posto_persona * NUMERO_KEYPOINT, (posto_persona + 1) * NUMERO_KEYPOINT)
        presente = ~np.isnan(keypoint_finestra[:, 0, 0])
        visibile = presente[:, None] & (np.nan_to_num(keypoint_finestra[:, :, 2]) >= SOGLIA_VISIBILITA_KEYPOINT)
        posizione = np.where(visibile[..., None], normalizza(np.nan_to_num(keypoint_finestra[:, :, :2])), 0.0)
        nodi[:, nodi_persona, 0:2] = posizione
        nodi[:, nodi_persona, 2] = np.where(visibile, np.nan_to_num(keypoint_finestra[:, :, 2]), 0.0)
        maschera[:, nodi_persona] = presente[:, None]
        # velocità solo dove il keypoint è visibile sia in questo frame sia nel precedente
        visibile_anche_prima = visibile[1:] & visibile[:-1]
        nodi[1:, nodi_persona, 3:5] = np.where(visibile_anche_prima[..., None], np.diff(posizione, axis=0), 0.0)

    classe_arma = np.zeros((frame_per_finestra, NUMERO_ARMI), np.int8)
    nella_finestra = (timeline.armi_frame >= inizio_finestra) & (timeline.armi_frame < fine_finestra)
    for indice_arma in np.flatnonzero(nella_finestra):
        frame_relativo = timeline.armi_frame[indice_arma] - inizio_finestra
        posti_liberi = np.flatnonzero(classe_arma[frame_relativo] == 0)
        confidenza = timeline.armi_confidenza[indice_arma]
        if len(posti_liberi):
            posto = posti_liberi[0]
        else:  # già 2 armi in questo frame: tiene le 2 più sicure
            nodi_armi = nodi[frame_relativo, NODI_PERSONE:, 2]
            posto = int(np.argmin(nodi_armi))
            if confidenza <= nodi_armi[posto]:
                continue
        box = timeline.armi_box[indice_arma]
        nodo = NODI_PERSONE + posto
        nodi[frame_relativo, nodo, 0:2] = normalizza(np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]))
        nodi[frame_relativo, nodo, 2] = confidenza
        maschera[frame_relativo, nodo] = True
        classe_arma[frame_relativo, posto] = timeline.armi_classe[indice_arma] + 1
    return nodi, maschera, classe_arma
