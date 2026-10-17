"""Split train/val/test PER VIDEO ORIGINALE (P4): le finestre si sovrappongono, quindi finestre dello stesso
video in split diversi falserebbero i risultati. Bilanciato per (sorgente, classe prevalente del gruppo)."""

from collections import Counter, defaultdict

import numpy as np

QUOTE_PREDEFINITE = (0.70, 0.15, 0.15)
NOMI_SPLIT = ("train", "val", "test")


def dividi_per_gruppi(gruppi: np.ndarray, etichette: np.ndarray, sorgenti: np.ndarray,
                      quote=QUOTE_PREDEFINITE, seme: int = 0) -> np.ndarray:
    """Ritorna per ogni finestra l'indice dello split (0 train, 1 val, 2 test)."""
    finestre_per_gruppo = Counter(gruppi.tolist())
    classe_prevalente = {}
    for gruppo in finestre_per_gruppo:
        classi_gruppo = etichette[gruppi == gruppo]
        classe_prevalente[gruppo] = int(np.bincount(classi_gruppo).argmax())
    sorgente_gruppo = dict(zip(gruppi.tolist(), sorgenti.tolist()))

    strati = defaultdict(list)
    for gruppo in sorted(finestre_per_gruppo):
        strati[(sorgente_gruppo[gruppo], classe_prevalente[gruppo])].append(gruppo)

    generatore = np.random.default_rng(seme)
    split_del_gruppo = {}
    soglie = np.cumsum(quote)
    for chiave in sorted(strati):
        gruppi_strato = strati[chiave]
        generatore.shuffle(gruppi_strato)
        totale = sum(finestre_per_gruppo[gruppo] for gruppo in gruppi_strato)
        cumulate = 0
        for gruppo in gruppi_strato:
            # il gruppo va nello split in cui cade il suo punto centrale (in finestre cumulate)
            centro = (cumulate + finestre_per_gruppo[gruppo] / 2) / totale
            split_del_gruppo[gruppo] = int(np.searchsorted(soglie, centro, side="right"))
            cumulate += finestre_per_gruppo[gruppo]
    return np.array([min(split_del_gruppo[gruppo], len(quote) - 1) for gruppo in gruppi.tolist()], np.int64)
