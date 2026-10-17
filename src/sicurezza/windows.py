"""Finestre temporali di 40 frame con passo 10 ed etichettatura dalle annotazioni (decisioni D2 e regola ≥20 frame)."""

from collections.abc import Sequence

import numpy as np

from .config import FPS, MIN_COVERAGE, STRIDE, WINDOW

# Evento annotato: (primo frame, frame dopo l'ultimo, classe) -> intervallo semiaperto [a, b)
Event = tuple[int, int, int]


def window_starts(n_frames: int, window: int = WINDOW, stride: int = STRIDE) -> np.ndarray:
    """Indici di partenza: 0, 10, 20, ... finché la finestra sta tutta nel video."""
    if n_frames < window:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n_frames - window + 1, stride)


def seconds_to_frames(t0: float, t1: float, fps: int = FPS) -> tuple[int, int]:
    return round(t0 * fps), round(t1 * fps)


def class_coverage(start: int, events: Sequence[Event], window: int = WINDOW) -> dict[int, int]:
    """Per ogni classe, quanti frame della finestra [start, start+window) cadono in almeno un suo evento."""
    end = start + window
    covered: dict[int, np.ndarray] = {}
    for a, b, cls in events:
        lo, hi = max(a, start), min(b, end)
        if lo < hi:
            mask = covered.setdefault(cls, np.zeros(window, dtype=bool))
            mask[lo - start:hi - start] = True
    return {cls: int(mask.sum()) for cls, mask in covered.items()}


def label_window(start: int, events: Sequence[Event], background: int | None,
                 window: int = WINDOW, min_coverage: int = MIN_COVERAGE) -> int | None:
    """Classe della finestra, oppure None se va scartata.

    - una classe copre ≥ min_coverage frame → quella classe (se più di una, quella che copre di più);
    - qualche evento la tocca ma nessuno arriva alla soglia → None (ambigua, scartata);
    - nessun evento → `background` (es. "normale" per CCTV-Fights, None dove il resto del video non è affidabile).
    """
    cov = class_coverage(start, events, window)
    if not cov:
        return background
    cls, frames = max(cov.items(), key=lambda kv: kv[1])
    return cls if frames >= min_coverage else None
