"""Prepara la foto del lampione per il titolo del sito: versione notturna dello scontorno.

Foto: «Rome (Italy), street light -- 2013 -- 3484.jpg», Dietmar Rabich, Wikimedia Commons, CC BY-SA 4.0
https://commons.wikimedia.org/wiki/File:Rome_(Italy),_street_light_--_2013_--_3484.jpg
Modifiche (quindi anche il risultato è CC BY-SA 4.0): sfondo tolto con rembg (modello isnet-general-use),
ritaglio, toni notturni e riflesso caldo sotto la campana.

Uso (dalla cartella site/):
    uvx --python 3.12 --from "rembg[cpu,cli]" rembg i -m isnet-general-use foto.jpg scontornato.png
    python3 dati/strumenti/prepara_lampione.py scontornato.png immagini/lampione-pastorale.webp
"""
import sys

import numpy as np
from PIL import Image

ALTEZZA_FINALE = 1400
PROLUNGA_PALO = 700        # la foto taglia il palo: lo allungo ripetendo le ultime righe (palo liscio) fino al bordo del titolo
RIGHE_RIPETUTE = 24
# bocca della campana nella foto ritagliata, in frazioni di larghezza e altezza (misurata sull'alfa)
BOCCA_X, BOCCA_Y = 0.454, 0.443
LUCE_CALDA = np.array([246, 220, 162], dtype=np.float32)   # --luce del sito
LUNA = np.array([150, 160, 178], dtype=np.float32)          # luce ambiente fredda, molto debole

if __name__ == "__main__":
    immagine = Image.open(sys.argv[1]).convert("RGBA")
    immagine = immagine.crop(immagine.getbbox())
    larghezza = round(immagine.width * ALTEZZA_FINALE / immagine.height)
    immagine = immagine.resize((larghezza, ALTEZZA_FINALE), Image.LANCZOS)

    pixel = np.asarray(immagine).astype(np.float32)
    colore, alfa = pixel[..., :3] / 255, pixel[..., 3:]
    luminanza = (colore @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]

    # notte: poca saturazione, molto più scuro, una punta di freddo
    notte = (0.55 * luminanza + 0.45 * colore) * 0.34 * (LUNA / 255) * 1.6

    # riflesso caldo: forte vicino alla bocca della campana e solo sotto il suo bordo superiore
    altezza, larghezza = luminanza.shape[:2]
    y, x = np.mgrid[0:altezza, 0:larghezza].astype(np.float32)
    centro_x, centro_y = BOCCA_X * larghezza, BOCCA_Y * altezza
    distanza = np.hypot((x - centro_x) / 1.3, y - centro_y)
    riflesso = np.exp(-(distanza / (0.06 * altezza)) ** 2) * np.clip((y - centro_y + 0.05 * altezza) / (0.05 * altezza), 0, 1)
    notte += riflesso[..., None] * (0.35 + 0.9 * luminanza) * (LUCE_CALDA / 255)

    risultato = np.concatenate([np.clip(notte, 0, 1) * 255, alfa], axis=-1).astype(np.uint8)
    striscia = risultato[-RIGHE_RIPETUTE:]
    prolunga = np.concatenate([striscia] * (PROLUNGA_PALO // RIGHE_RIPETUTE + 1))[:PROLUNGA_PALO]
    risultato = np.concatenate([risultato, prolunga])
    Image.fromarray(risultato, "RGBA").save(sys.argv[2], quality=86, method=6)
    print(sys.argv[2], larghezza, "x", risultato.shape[0], "· bocca della campana al", round(BOCCA_Y * altezza / risultato.shape[0] * 100, 2), "% in verticale")
