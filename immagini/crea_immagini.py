"""Immagini a tema Argo (D62): banner del README e anteprima social del repo, dal dipinto "Giunone e Argo" di
Rubens (1610 circa, pubblico dominio, Wikimedia Commons: File:Peter Paul Rubens - Juno and Argus - WGA20280.jpg).
Solo la coda del pavone con gli occhi di Argo, in bicromia con la palette del sito (Pietra e cenere), più la scritta.

Uso:  uv run python immagini/crea_immagini.py --font <cartella con AlegreyaSansSC-ExtraBold.ttf e IBMPlexMono-Regular.ttf>
(font OFL da github.com/google/fonts). Il dipinto va in immagini/sorgenti/rubens_giunone_e_argo.jpg.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CARTELLA = Path(__file__).parent
CENERE, LUCE, TESTO, TENUE = (27, 26, 24), (232, 196, 128), (231, 225, 212), (163, 155, 142)
RITAGLIO_PAVONE = (0.10, 0.15, 0.48, 0.46)    # la coda aperta, in frazioni del dipinto (sinistra, alto, destra, basso)


def bicromia(immagine: Image.Image) -> Image.Image:
    grigio = np.asarray(immagine.convert("L"), np.float32) / 255
    grigio = np.clip((grigio - 0.08) / 0.75, 0, 1) ** 1.15
    colori = np.array(CENERE, np.float32) + grigio[..., None] * (np.array(LUCE, np.float32) - np.array(CENERE, np.float32))
    return Image.fromarray(colori.astype(np.uint8))


def logo(lato: int) -> Image.Image:
    with tempfile.TemporaryDirectory() as cartella:
        uscita = Path(cartella) / "logo.png"
        subprocess.run(["rsvg-convert", "-w", str(lato), "-h", str(lato), "-o", str(uscita),
                        str(CARTELLA.parent / "src/sicurezza/pannello/logo.svg")], check=True)
        return Image.open(uscita).convert("RGBA")


def componi(larghezza: int, altezza: int, cartella_font: Path, uscita: Path) -> None:
    dipinto = Image.open(CARTELLA / "sorgenti" / "rubens_giunone_e_argo.jpg")
    w, h = dipinto.size
    sinistra, alto, destra, basso = RITAGLIO_PAVONE
    pavone = dipinto.crop((int(sinistra * w), int(alto * h), int(destra * w), int(basso * h)))
    # la coda occupa la parte destra; a sinistra sfuma nel colore di fondo, dove va la scritta
    lato_destro = int(larghezza * 0.62)
    scala = max(lato_destro / pavone.width, altezza / pavone.height)
    pavone = pavone.resize((int(pavone.width * scala), int(pavone.height * scala)), Image.LANCZOS)
    pavone = bicromia(pavone.crop((0, (pavone.height - altezza) // 2, lato_destro, (pavone.height - altezza) // 2 + altezza)))
    tela = Image.new("RGB", (larghezza, altezza), CENERE)
    tela.paste(pavone, (larghezza - lato_destro, 0))
    sfumatura = np.clip((np.arange(larghezza) - (larghezza - lato_destro)) / (lato_destro * 0.45), 0, 1) ** 1.4
    maschera = Image.fromarray((np.tile(sfumatura, (altezza, 1)) * 255).astype(np.uint8))
    tela = Image.composite(tela, Image.new("RGB", tela.size, CENERE), maschera)
    disegno = ImageDraw.Draw(tela)
    margine = int(altezza * 0.13)
    titolo = ImageFont.truetype(str(cartella_font / "AlegreyaSansSC-ExtraBold.ttf"), int(altezza * 0.30))
    sottotitolo = ImageFont.truetype(str(cartella_font / "IBMPlexMono-Regular.ttf"), int(altezza * 0.045))
    occhio = logo(int(altezza * 0.22))
    tela.paste(occhio, (margine, int(altezza * 0.20)), occhio)
    disegno.text((margine - int(altezza * 0.015), int(altezza * 0.40)), "ARGO", font=titolo, fill=TESTO)
    righe = ["violenza · violenza con arma · scippi", "dalle telecamere CCTV, con WARNING ARMA"]
    for numero, riga in enumerate(righe):
        disegno.text((margine, int(altezza * (0.76 + 0.07 * numero))), riga, font=sottotitolo, fill=TENUE)
    disegno.text((larghezza - margine // 2, altezza - int(altezza * 0.04)),
                 "P. P. Rubens, Giunone e Argo (1610 c.), pubblico dominio", font=ImageFont.truetype(
                     str(cartella_font / "IBMPlexMono-Regular.ttf"), int(altezza * 0.022)), fill=TENUE, anchor="rd")
    tela.save(uscita, quality=90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", required=True, type=Path)
    argomenti = parser.parse_args()
    componi(1600, 560, argomenti.font, CARTELLA / "argo_banner.jpg")
    componi(1280, 640, argomenti.font, CARTELLA / "argo_anteprima_social.jpg")   # anteprima del repo (Settings)
    print("immagini/argo_banner.jpg, immagini/argo_anteprima_social.jpg")
