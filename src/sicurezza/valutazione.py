"""Metriche (macro-F1, recall per classe, matrice di confusione) e latenza su CPU."""

import time

import numpy as np
import torch

from .caratteristiche import CARATTERISTICHE_PER_NODO, NODI_PER_FRAME, NUMERO_ARMI
from .config import CLASSI, FRAME_PER_FINESTRA


def matrice_confusione(vere: np.ndarray, predette: np.ndarray, numero_classi: int = len(CLASSI)) -> np.ndarray:
    matrice = np.zeros((numero_classi, numero_classi), np.int64)
    np.add.at(matrice, (vere, predette), 1)
    return matrice  # righe = classe vera, colonne = classe predetta


def metriche(vere: np.ndarray, predette: np.ndarray) -> dict:
    matrice = matrice_confusione(vere, predette)
    veri_positivi = np.diag(matrice).astype(float)
    recall = veri_positivi / np.maximum(matrice.sum(1), 1)
    precisione = veri_positivi / np.maximum(matrice.sum(0), 1)
    f1 = 2 * precisione * recall / np.maximum(precisione + recall, 1e-12)
    presenti = matrice.sum(1) > 0  # macro-F1 solo sulle classi che compaiono davvero
    return {
        "accuratezza": float(veri_positivi.sum() / max(matrice.sum(), 1)),
        "macro_f1": float(f1[presenti].mean()) if presenti.any() else 0.0,
        "recall": dict(zip(CLASSI, recall.round(4).tolist())),
        "precisione": dict(zip(CLASSI, precisione.round(4).tolist())),
        "f1": dict(zip(CLASSI, f1.round(4).tolist())),
        "matrice_confusione": matrice.tolist(),
    }


@torch.no_grad()
def latenza_cpu_millisecondi(modello: torch.nn.Module, ripetizioni: int = 20) -> float:
    """Millisecondi per una finestra (lotto da 1) su CPU, come nella demo sul portatile."""
    modello = modello.to("cpu").eval()
    nodi = torch.randn(1, FRAME_PER_FINESTRA, NODI_PER_FRAME, CARATTERISTICHE_PER_NODO)
    maschera = torch.ones(1, FRAME_PER_FINESTRA, NODI_PER_FRAME, dtype=torch.bool)
    classe_arma = torch.zeros(1, FRAME_PER_FINESTRA, NUMERO_ARMI, dtype=torch.long)
    modello(nodi, maschera, classe_arma)
    inizio = time.perf_counter()
    for _ in range(ripetizioni):
        modello(nodi, maschera, classe_arma)
    return (time.perf_counter() - inizio) / ripetizioni * 1000
