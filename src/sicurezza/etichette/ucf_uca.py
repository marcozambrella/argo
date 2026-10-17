"""UCF-Crime (Kaggle bypktt/ucf-crimes) con i tempi degli eventi da UCA (annotations/uca/, Apache 2.0). D10.

UCA descrive TUTTI gli eventi di ogni video, anche quelli normali, con una frase e i secondi di inizio e fine.
La classe di ogni frase si ricava dal testo con le regole a parole chiave qui sotto.

!!! REGOLE IN BOZZA, DA DECIDERE CON MARCO (vedi DIARIO.md): sono rumorose. Esempi di errori già visti:
    "pushed an old man in a wheelchair" → violenza; negli incidenti stradali "hit" → violenza.
    Per questo le categorie RoadAccidents, Explosion, Arson, Vandalism, Shoplifting sono escluse.
Si estraggono solo i segmenti candidati (± MARGINE_SECONDI); le finestre fuori dagli eventi si scartano.
"""

import json
import random
import re
from pathlib import Path

from . import ClipAnnotata

CARTELLA_UCA = Path(__file__).resolve().parents[3] / "annotations" / "uca"

CATEGORIE_UCF_USATE = ("Abuse", "Arrest", "Assault", "Burglary", "Fighting", "Robbery", "Shooting", "Stealing")
MARGINE_SECONDI = 2.0                    # PROVVISORIO: contesto prima e dopo ogni evento
NUMERO_VIDEO_NORMALI = 150               # PROVVISORIO
SECONDI_PER_VIDEO_NORMALE = 120.0        # PROVVISORIO

# --- BOZZA delle regole a parole chiave -------------------------------------------------------------------
PAROLE_ARMA = r"\b(gun|guns|pistol|handgun|rifle|shotgun|firearm|knife|knives|machete|dagger|stab\w*|shot|shoot\w*|fired|weapon\w*)\b"
PAROLE_MINACCIA = r"\b(point\w*|threat\w*|aim\w*|wav\w*|brandish\w*)\b"
PAROLE_VIOLENZA = (r"\b(hit|hits|hitting|punch\w*|kick\w*|beat\w*|slap\w*|fight\w*|fought|push\w*|shov\w*|wrestl\w*|"
                   r"attack\w*|strangl\w*|chok\w*|assault\w*)\b")
PAROLE_SCIPPO = (r"\b(snatch\w*|grabb?\w* (the|his|her|a) \w*\s?(phone|bag|purse|necklace|chain|wallet|handbag|backpack)|"
                 r"robbed (him|her)|took (his|her) (phone|bag|purse|wallet|necklace))\b")


def classe_da_frase(frase: str) -> str | None:
    """BOZZA: classe di una frase UCA, oppure None se la frase non descrive un evento utile."""
    testo = frase.lower()
    arma = re.search(PAROLE_ARMA, testo)
    violenza = re.search(PAROLE_VIOLENZA, testo)
    if arma and (violenza or re.search(PAROLE_MINACCIA, testo)):
        return "violenza_con_arma"
    if re.search(PAROLE_SCIPPO, testo):
        return "scippo"
    if violenza:
        return "violenza"
    return None
# ------------------------------------------------------------------------------------------------------------


def carica_uca(cartella_uca: Path = CARTELLA_UCA) -> dict[str, dict]:
    annotazioni = {}
    for file_uca in sorted(Path(cartella_uca).glob("UCFCrime_*.json")):
        annotazioni.update(json.loads(file_uca.read_text()))
    return annotazioni


def categoria_ucf(id_video: str) -> str:
    return "Normal" if id_video.startswith("Normal") else re.match(r"[A-Za-z]+", id_video).group(0)


def unisci_intervalli(intervalli: list[tuple[float, float]]) -> list[tuple[float, float]]:
    uniti = []
    for inizio, fine in sorted(intervalli):
        if uniti and inizio <= uniti[-1][1]:
            uniti[-1] = (uniti[-1][0], max(uniti[-1][1], fine))
        else:
            uniti.append((inizio, fine))
    return uniti


def elenca_clip(radice_dataset: str | Path, cartella_uca: Path = CARTELLA_UCA, seme: int = 0) -> list[ClipAnnotata]:
    percorso_per_id = {percorso.stem: percorso for percorso in Path(radice_dataset).glob("**/*.mp4")}
    annotazioni = carica_uca(cartella_uca)
    clip = []
    for id_video, dati in sorted(annotazioni.items()):
        categoria = categoria_ucf(id_video)
        if categoria not in CATEGORIE_UCF_USATE or id_video not in percorso_per_id:
            continue
        durata = float(dati["duration"])
        eventi = [(float(inizio), float(fine), classe) for (inizio, fine), frase
                  in zip(dati["timestamps"], dati["sentences"]) if (classe := classe_da_frase(frase))]
        segmenti = unisci_intervalli([(max(0.0, inizio - MARGINE_SECONDI), min(durata, fine + MARGINE_SECONDI))
                                      for inizio, fine, _ in eventi])
        for inizio_segmento, fine_segmento in segmenti:
            eventi_nel_segmento = [(max(inizio, inizio_segmento) - inizio_segmento,
                                    min(fine, fine_segmento) - inizio_segmento, classe)
                                   for inizio, fine, classe in eventi if inizio < fine_segmento and fine > inizio_segmento]
            clip.append(ClipAnnotata(
                id_clip=f"ucf/{id_video}@{inizio_segmento:.1f}-{fine_segmento:.1f}", sorgente="ucf_uca",
                gruppo=f"ucf/{id_video}", percorso_video=str(percorso_per_id[id_video]),
                inizio_secondi=inizio_segmento, fine_secondi=fine_segmento, eventi_secondi=eventi_nel_segmento,
                classe_sfondo=None, note=f"categoria={categoria}; regole UCA in BOZZA"))

    # campione PROVVISORIO di video normali: un pezzo da ogni video, posizione casuale ma riproducibile
    generatore = random.Random(seme)
    normali = sorted(id_video for id_video in annotazioni
                     if categoria_ucf(id_video) == "Normal" and id_video in percorso_per_id)
    for id_video in generatore.sample(normali, min(NUMERO_VIDEO_NORMALI, len(normali))):
        durata = float(annotazioni[id_video]["duration"])
        inizio = generatore.uniform(0.0, max(0.0, durata - SECONDI_PER_VIDEO_NORMALE))
        fine = min(durata, inizio + SECONDI_PER_VIDEO_NORMALE)
        clip.append(ClipAnnotata(
            id_clip=f"ucf/{id_video}@{inizio:.1f}-{fine:.1f}", sorgente="ucf_uca", gruppo=f"ucf/{id_video}",
            percorso_video=str(percorso_per_id[id_video]), inizio_secondi=inizio, fine_secondi=fine,
            classe_intera="normale", note="categoria=Normal"))
    return clip
