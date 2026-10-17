"""Estrazioni finte ma realistiche per testare caratteristiche, modelli e training senza video."""

import numpy as np

from sicurezza.estrazione import impacchetta_estrazione


def scheletro_in_piedi(centro_x: float, piedi_y: float, altezza: float) -> np.ndarray:
    """17 keypoint (x, y, confidenza) di una persona in piedi, coordinate in pixel."""
    proporzioni = np.array([
        [0, 0.07], [-.02, .05], [.02, .05], [-.04, .06], [.04, .06], [-.1, .2], [.1, .2], [-.13, .38], [.13, .38],
        [-.14, .52], [.14, .52], [-.07, .53], [.07, .53], [-.07, .76], [.07, .76], [-.07, 1.0], [.07, 1.0]])
    keypoint = np.empty((17, 3), np.float32)
    keypoint[:, 0] = centro_x + proporzioni[:, 0] * altezza
    keypoint[:, 1] = piedi_y - altezza + proporzioni[:, 1] * altezza
    keypoint[:, 2] = 0.9
    return keypoint


def estrazione_finta(numero_frame=60, numero_persone=3, con_arma=True, larghezza=640, altezza=480, seme=0):
    generatore = np.random.default_rng(seme)
    righe_persone, righe_armi = [], []
    for frame in range(numero_frame):
        keypoint, box = [], []
        for persona in range(numero_persone):
            centro_x = 100 + 150 * persona + 2 * frame + generatore.normal(0, 1)
            scheletro = scheletro_in_piedi(centro_x, 400, 200)
            keypoint.append(scheletro)
            box.append([scheletro[:, 0].min(), scheletro[:, 1].min(), scheletro[:, 0].max(), scheletro[:, 1].max()])
        righe_persone.append((frame, {"id_traccia": np.arange(1, numero_persone + 1), "box": np.array(box),
                                      "confidenza": np.full(numero_persone, .9), "keypoint": np.array(keypoint)}))
        if con_arma and frame % 2 == 0:
            polso = keypoint[0][10, :2]
            righe_armi.append((frame, {"box": np.array([[*polso - 10, *polso + 10]]), "classe": np.array([0]),
                                       "confidenza": np.array([0.8])}))
        else:
            righe_armi.append((frame, {"box": np.empty((0, 4)), "classe": np.empty(0, int),
                                       "confidenza": np.empty(0)}))
    return impacchetta_estrazione(righe_persone, righe_armi, numero_frame, larghezza, altezza, 0.0)
