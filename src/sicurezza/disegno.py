"""Disegno dell'overlay sui frame della demo.

>>> SEGNAPOSTO DESIGN <<<  Colori, font, posizioni e testi sono provvisori: la grafica la decide Marco.
Tutto lo stile è in STILE_OVERLAY e TESTI_CLASSI, così si cambia senza toccare la logica.
"""

import cv2
import numpy as np

from .config import CLASSI
from .scheletro import OSSA

STILE_OVERLAY = {  # SEGNAPOSTO DESIGN — colori in BGR
    "colore_per_classe": {"normale": (90, 200, 90), "violenza": (40, 140, 255), "violenza_con_arma": (40, 40, 230),
                          "scippo": (230, 160, 40)},
    "colore_scheletro": (255, 255, 255),
    "colore_arma": (0, 0, 255),
    "colore_testo": (255, 255, 255),
    "sfondo_pannello": (20, 20, 20),
    "opacita_pannello": 0.6,
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "scala_font": 0.55,
    "spessore": 2,
    "soglia_visibilita_keypoint": 0.3,
}
TESTI_CLASSI = {"normale": "Normale", "violenza": "Violenza", "violenza_con_arma": "Violenza con arma",
                "scippo": "Scippo"}  # SEGNAPOSTO DESIGN
TESTO_WARNING_ARMA = "WARNING ARMA"     # SEGNAPOSTO DESIGN
NOMI_ARMI = ("pistola", "coltello", "fucile")


def disegna_persone(frame: np.ndarray, persone: dict) -> None:
    stile = STILE_OVERLAY
    for id_traccia, box, keypoint in zip(persone["id_traccia"], persone["box"], persone["keypoint"]):
        visibile = keypoint[:, 2] >= stile["soglia_visibilita_keypoint"]
        for inizio, fine in OSSA:
            if visibile[inizio] and visibile[fine]:
                cv2.line(frame, tuple(map(int, keypoint[inizio, :2])), tuple(map(int, keypoint[fine, :2])),
                         stile["colore_scheletro"], stile["spessore"], cv2.LINE_AA)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), stile["colore_scheletro"], 1)
        cv2.putText(frame, f"ID {int(id_traccia)}", (x1, max(15, y1 - 5)), stile["font"], stile["scala_font"],
                    stile["colore_testo"], 1, cv2.LINE_AA)


def disegna_armi(frame: np.ndarray, armi: dict, soglia_confidenza: float) -> None:
    stile = STILE_OVERLAY
    for box, classe, confidenza in zip(armi["box"], armi["classe"], armi["confidenza"]):
        if confidenza < soglia_confidenza:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), stile["colore_arma"], stile["spessore"])
        cv2.putText(frame, f"{NOMI_ARMI[int(classe)]} {confidenza:.2f}", (x1, max(15, y1 - 5)), stile["font"],
                    stile["scala_font"], stile["colore_arma"], 1, cv2.LINE_AA)


def disegna_pannello(frame: np.ndarray, probabilita: np.ndarray | None, classe: str | None, warning: bool) -> None:
    """Pannello in alto a sinistra: classe decisa, barre delle probabilità, WARNING ARMA."""
    stile = STILE_OVERLAY
    larghezza_pannello, altezza_riga, inizio_barre = 330, 24, 160
    altezza_pannello = altezza_riga * (len(CLASSI) + 2) + 10
    sovrapposizione = frame.copy()
    cv2.rectangle(sovrapposizione, (0, 0), (larghezza_pannello, altezza_pannello), stile["sfondo_pannello"], -1)
    cv2.addWeighted(sovrapposizione, stile["opacita_pannello"], frame, 1 - stile["opacita_pannello"], 0, frame)
    if classe is None:
        cv2.putText(frame, "in attesa di 40 frame...", (10, altezza_riga), stile["font"], stile["scala_font"],
                    stile["colore_testo"], 1, cv2.LINE_AA)
        return
    cv2.putText(frame, TESTI_CLASSI[classe], (10, altezza_riga), stile["font"], stile["scala_font"] * 1.3,
                stile["colore_per_classe"][classe], stile["spessore"], cv2.LINE_AA)
    for riga, (nome_classe, valore) in enumerate(zip(CLASSI, probabilita), start=1):
        y = altezza_riga * (riga + 1)
        cv2.rectangle(frame, (inizio_barre, y - 14), (inizio_barre + int(160 * valore), y - 2),
                      stile["colore_per_classe"][nome_classe], -1)
        cv2.putText(frame, TESTI_CLASSI[nome_classe], (10, y - 3), stile["font"], 0.45,
                    stile["colore_testo"], 1, cv2.LINE_AA)
    if warning:
        y = altezza_riga * (len(CLASSI) + 2)
        cv2.putText(frame, TESTO_WARNING_ARMA, (10, y), stile["font"], stile["scala_font"] * 1.2,
                    stile["colore_arma"], stile["spessore"], cv2.LINE_AA)
