"""D68, CPU: SCVD non dice che arma c'è nelle clip "Weaponized". Per ogni clip tre frame (a 1/4, 1/2 e 3/4) in
provini da 12 clip, con il nome della clip scritto sopra: si guardano e si scrive il tipo d'arma (fuoco/coltello)
in annotations/armi/tipo_arma_scvd.json."""
from pathlib import Path
import cv2
import numpy as np

ingresso = Path("/kaggle/input")
uscita = Path("/kaggle/working")
CLIP_PER_PROVINO = 12
LATO = 320


def tre_frame(percorso: Path) -> list:
    video = cv2.VideoCapture(str(percorso))
    totale = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame = []
    for frazione in (0.25, 0.5, 0.75):
        video.set(cv2.CAP_PROP_POS_FRAMES, int(totale * frazione))
        letto, immagine = video.read()
        if not letto:
            immagine = np.zeros((LATO, LATO, 3), np.uint8)
        altezza, larghezza = immagine.shape[:2]
        scala = LATO / max(altezza, larghezza)
        immagine = cv2.resize(immagine, (int(larghezza * scala), int(altezza * scala)))
        quadro = np.zeros((LATO, LATO, 3), np.uint8)
        quadro[:immagine.shape[0], :immagine.shape[1]] = immagine
        frame.append(quadro)
    video.release()
    return frame


percorsi = sorted(ingresso.glob("**/SCVD_converted/*/Weaponized/*.avi"))
print(len(percorsi), "clip Weaponized", flush=True)
for numero_provino in range(0, len(percorsi), CLIP_PER_PROVINO):
    righe = []
    for percorso in percorsi[numero_provino:numero_provino + CLIP_PER_PROVINO]:
        riga = np.hstack(tre_frame(percorso))
        nome = f"{percorso.parent.parent.name}/{percorso.stem}"
        cv2.putText(riga, nome, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        righe.append(riga)
    cv2.imwrite(str(uscita / f"provino_{numero_provino // CLIP_PER_PROVINO:02d}.jpg"), np.vstack(righe),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
print("fatto", flush=True)
