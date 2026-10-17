"""Finestre temporali di 40 frame con passo 10 ed etichettatura dalle annotazioni (D2 e P3 nel diario)."""

from collections.abc import Sequence

import numpy as np

from .config import FPS_ANALISI, FRAME_MINIMI_EVENTO, FRAME_PER_FINESTRA, PASSO_FINESTRE

# Evento annotato: (primo frame, frame dopo l'ultimo, indice classe) -> intervallo semiaperto [primo, fine)
Evento = tuple[int, int, int]


def inizi_finestre(numero_frame: int, frame_per_finestra: int = FRAME_PER_FINESTRA,
                   passo: int = PASSO_FINESTRE) -> np.ndarray:
    """Frame iniziali delle finestre: 0, 10, 20, ... finché la finestra sta tutta nel video."""
    if numero_frame < frame_per_finestra:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, numero_frame - frame_per_finestra + 1, passo)


def secondi_in_frame(inizio_secondi: float, fine_secondi: float, fps: int = FPS_ANALISI) -> tuple[int, int]:
    return round(inizio_secondi * fps), round(fine_secondi * fps)


def frame_coperti_per_classe(inizio_finestra: int, eventi: Sequence[Evento],
                             frame_per_finestra: int = FRAME_PER_FINESTRA) -> dict[int, int]:
    """Per ogni classe, quanti frame della finestra cadono in almeno un suo evento (sovrapposizioni contate una volta)."""
    fine_finestra = inizio_finestra + frame_per_finestra
    maschere_per_classe: dict[int, np.ndarray] = {}
    for primo_frame, frame_dopo_ultimo, classe in eventi:
        da, a = max(primo_frame, inizio_finestra), min(frame_dopo_ultimo, fine_finestra)
        if da < a:
            maschera = maschere_per_classe.setdefault(classe, np.zeros(frame_per_finestra, dtype=bool))
            maschera[da - inizio_finestra:a - inizio_finestra] = True
    return {classe: int(maschera.sum()) for classe, maschera in maschere_per_classe.items()}


def etichetta_finestra(inizio_finestra: int, eventi: Sequence[Evento], classe_sfondo: int | None,
                       frame_per_finestra: int = FRAME_PER_FINESTRA,
                       frame_minimi_evento: int = FRAME_MINIMI_EVENTO) -> int | None:
    """Classe della finestra, oppure None se va scartata.

    - una classe copre ≥ frame_minimi_evento frame → quella classe (se più di una, quella che copre di più);
    - qualche evento la tocca ma nessuno arriva alla soglia → None (ambigua, scartata);
    - nessun evento → `classe_sfondo` (es. "normale" per CCTV-Fights; None dove il resto del video non è affidabile).
    """
    copertura = frame_coperti_per_classe(inizio_finestra, eventi, frame_per_finestra)
    if not copertura:
        return classe_sfondo
    classe_prevalente, frame_coperti = max(copertura.items(), key=lambda coppia: coppia[1])
    return classe_prevalente if frame_coperti >= frame_minimi_evento else None
