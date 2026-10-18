"""Sicurezza NN — dati. File unico (richiesta di Marco: massimo 5 file Python)."""

from collections import Counter
from collections import defaultdict
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from dataclasses import field
from pathlib import Path
import argparse
import json
import random
import re
import subprocess
import sys

import numpy as np


# ==============================================================================================================
# config
# Costanti condivise da estrazione, training e demo (decisioni D2 e D5 nel diario).
# ==============================================================================================================

FPS_ANALISI = 25                 # tutti i video vengono letti a 25 fps
FRAME_PER_FINESTRA = 40          # 1,6 secondi
PASSO_FINESTRE = 10              # finestre 0-39, 10-49, 20-59, ...
FRAME_MINIMI_EVENTO = 20         # frame di evento necessari perché la finestra prenda la classe dell'evento

CLASSI = ("normale", "violenza", "violenza_con_arma", "scippo")
INDICE_CLASSE = {nome: indice for indice, nome in enumerate(CLASSI)}


# ==============================================================================================================
# scheletro
# Scheletro COCO a 17 keypoint (formato dei modelli YOLO pose).
# ==============================================================================================================

NOMI_KEYPOINT = (
    "naso", "occhio_sx", "occhio_dx", "orecchio_sx", "orecchio_dx", "spalla_sx", "spalla_dx",
    "gomito_sx", "gomito_dx", "polso_sx", "polso_dx", "anca_sx", "anca_dx",
    "ginocchio_sx", "ginocchio_dx", "caviglia_sx", "caviglia_dx",
)
NUMERO_KEYPOINT = len(NOMI_KEYPOINT)
POLSI = (NOMI_KEYPOINT.index("polso_sx"), NOMI_KEYPOINT.index("polso_dx"))

OSSA = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)

# per il ribaltamento orizzontale (data augmentation): sinistra e destra si scambiano
SPECCHIO_KEYPOINT = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)
CENTRO_SCHELETRO = 0  # naso come radice per la partizione "spaziale" di ST-GCN


# ==============================================================================================================
# video
# Lettura video a fps costante tramite ffmpeg (conversione al volo, nessun file riscritto).
# ==============================================================================================================

@dataclass(frozen=True)
class InfoVideo:
    larghezza: int
    altezza: int
    durata_secondi: float


def leggi_info_video(percorso_video: str) -> InfoVideo:
    uscita_ffprobe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration:stream_side_data=rotation:format=duration",
         "-of", "json", str(percorso_video)],
        capture_output=True, text=True, check=True,
    ).stdout
    dati = json.loads(uscita_ffprobe)
    flusso_video = dati["streams"][0]
    larghezza, altezza = int(flusso_video["width"]), int(flusso_video["height"])
    # ffmpeg ruota automaticamente i video con metadato di rotazione: le dimensioni in uscita vanno scambiate
    rotazione = next((int(dato["rotation"]) for dato in flusso_video.get("side_data_list", []) if "rotation" in dato), 0)
    if rotazione % 180:
        larghezza, altezza = altezza, larghezza
    durata = float(flusso_video.get("duration") or dati.get("format", {}).get("duration") or 0.0)
    return InfoVideo(larghezza, altezza, durata)


def leggi_frame(percorso_video: str, fps: int = FPS_ANALISI, inizio_secondi: float | None = None,
                fine_secondi: float | None = None) -> Iterator[np.ndarray]:
    """Frame BGR (altezza, larghezza, 3) uint8 a `fps` costanti, opzionalmente solo tra inizio e fine (secondi)."""
    info = leggi_info_video(percorso_video)
    comando = ["ffmpeg", "-v", "error", "-nostdin"]
    if inizio_secondi:
        comando += ["-ss", f"{inizio_secondi:.3f}"]
    comando += ["-i", str(percorso_video)]
    if fine_secondi is not None:
        comando += ["-t", f"{fine_secondi - (inizio_secondi or 0):.3f}"]
    comando += ["-vf", f"fps={fps}", "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    byte_per_frame = info.larghezza * info.altezza * 3
    processo_ffmpeg = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       bufsize=byte_per_frame)
    try:
        while True:
            dati_frame = processo_ffmpeg.stdout.read(byte_per_frame)
            if len(dati_frame) < byte_per_frame:
                break
            yield np.frombuffer(dati_frame, np.uint8).reshape(info.altezza, info.larghezza, 3)
    finally:
        processo_ffmpeg.stdout.close()
        processo_ffmpeg.kill()
        processo_ffmpeg.wait()


# ==============================================================================================================
# finestre
# Finestre temporali di 40 frame con passo 10 ed etichettatura dalle annotazioni (D2 e P3 nel diario).
# ==============================================================================================================

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

    - un evento "conta" se copre ≥ frame_minimi_evento frame della finestra (P3), oppure, se l'evento è più corto
      di così, se la finestra lo contiene per intero (D22: gli scippi durano spesso meno di 0,8 s);
    - tra gli eventi che contano vince la classe che copre più frame;
    - qualche evento la tocca ma nessuno conta → None (ambigua, scartata);
    - nessun evento → `classe_sfondo` (es. "normale" per CCTV-Fights; None dove il resto del video non è affidabile).
    """
    fine_finestra = inizio_finestra + frame_per_finestra
    classi_che_contano = set()
    tocca_qualcosa = False
    for primo_frame, frame_dopo_ultimo, classe in eventi:
        coperti = min(frame_dopo_ultimo, fine_finestra) - max(primo_frame, inizio_finestra)
        if coperti <= 0:
            continue
        tocca_qualcosa = True
        durata_evento = frame_dopo_ultimo - primo_frame
        if coperti >= min(frame_minimi_evento, durata_evento):
            classi_che_contano.add(classe)
    if not tocca_qualcosa:
        return classe_sfondo
    if not classi_che_contano:
        return None
    copertura = frame_coperti_per_classe(inizio_finestra, eventi, frame_per_finestra)
    return max(classi_che_contano, key=lambda classe: copertura[classe])


# ==============================================================================================================
# estrazione
# ==============================================================================================================

@dataclass
class VideoDaEstrarre:
    """Un video (o un suo segmento) da estrarre. `id_clip` diventa il nome del file .npz."""
    id_clip: str
    percorso_video: str
    inizio_secondi: float | None = None
    fine_secondi: float | None = None
    telecamera_mobile: bool = False   # D20: la compensazione del movimento (GMC) del tracker serve solo qui


def leggi_elenco(percorso_elenco: str | Path) -> list[VideoDaEstrarre]:
    with open(percorso_elenco) as file_elenco:
        return [VideoDaEstrarre(**json.loads(riga)) for riga in file_elenco if riga.strip()]


def scrivi_elenco(video_da_estrarre: list[VideoDaEstrarre], percorso_elenco: str | Path) -> None:
    with open(percorso_elenco, "w") as file_elenco:
        for video in video_da_estrarre:
            file_elenco.write(json.dumps(asdict(video)) + "\n")


def nome_file_estratto(id_clip: str) -> str:
    return id_clip.replace("/", "__") + ".npz"


# ==============================================================================================================
# etichette/__init__
# Etichette per sorgente: ogni modulo trasforma un dataset in una lista di ClipAnnotata.
#
# Una ClipAnnotata dice cosa estrarre (video, eventuale segmento) e come etichettarne le finestre.
# L'estrazione non dipende dalle etichette: si possono cambiare le regole senza rifare l'estrazione.
# ==============================================================================================================

@dataclass
class ClipAnnotata:
    id_clip: str                    # univoco; diventa anche il nome del .npz
    sorgente: str                   # rwf2000, cctv_fights, scvd, ucf_uca, chain_snatching
    gruppo: str                     # video originale: tutte le clip dello stesso gruppo stanno nello stesso split (P4)
    percorso_video: str
    inizio_secondi: float | None = None
    fine_secondi: float | None = None
    eventi_secondi: list[tuple[float, float, str]] = field(default_factory=list)  # relativi all'inizio della clip
    classe_intera: str | None = None    # tutta la clip ha questa classe (dataset già tagliati)
    classe_sfondo: str | None = None    # classe delle finestre senza eventi (None = scartarle)
    split_originale: str | None = None
    telecamera_mobile: bool = False     # D20: accende la compensazione del movimento nel tracker
    note: str = ""

    def da_estrarre(self) -> VideoDaEstrarre:
        return VideoDaEstrarre(self.id_clip, self.percorso_video, self.inizio_secondi, self.fine_secondi,
                               self.telecamera_mobile)

    def eventi_in_frame(self, numero_frame: int) -> list[Evento]:
        if self.classe_intera is not None:
            return [(0, numero_frame, INDICE_CLASSE[self.classe_intera])]
        return [(*secondi_in_frame(inizio, fine), INDICE_CLASSE[classe]) for inizio, fine, classe in self.eventi_secondi]

    def indice_classe_sfondo(self) -> int | None:
        return None if self.classe_sfondo is None else INDICE_CLASSE[self.classe_sfondo]


def scrivi_annotazioni(clip: list[ClipAnnotata], percorso: str | Path) -> None:
    with open(percorso, "w") as file_annotazioni:
        for singola_clip in clip:
            file_annotazioni.write(json.dumps(asdict(singola_clip)) + "\n")


def leggi_annotazioni(percorso: str | Path) -> list[ClipAnnotata]:
    with open(percorso) as file_annotazioni:
        clip = [ClipAnnotata(**json.loads(riga)) for riga in file_annotazioni if riga.strip()]
    for singola_clip in clip:
        singola_clip.eventi_secondi = [tuple(evento) for evento in singola_clip.eventi_secondi]
    return clip


# ==============================================================================================================
# etichette/rwf2000
# RWF-2000 (copia Kaggle vulamnguyen/rwf2000): clip di 5 s già tagliate. Fight → violenza, NonFight → normale.
# ==============================================================================================================

CLASSE_PER_CARTELLA_RWF2000 = {"Fight": "violenza", "NonFight": "normale"}


def elenca_clip_rwf2000(radice_dataset: str | Path) -> list[ClipAnnotata]:
    clip = []
    for percorso in sorted(Path(radice_dataset).glob("**/RWF-2000/*/*/*.avi")):
        cartella_classe, cartella_split = percorso.parent.name, percorso.parent.parent.name
        if cartella_classe not in CLASSE_PER_CARTELLA_RWF2000:
            continue
        # i nomi sono <id YouTube>_<n>: clip dello stesso video originale devono finire nello stesso split
        video_originale = percorso.stem.rsplit("_", 1)[0]
        clip.append(ClipAnnotata(
            id_clip=f"rwf2000/{cartella_split}/{cartella_classe}/{percorso.stem}", sorgente="rwf2000",
            gruppo=f"rwf2000/{video_originale}", percorso_video=str(percorso),
            classe_intera=CLASSE_PER_CARTELLA_RWF2000[cartella_classe], split_originale=cartella_split))
    return clip


# ==============================================================================================================
# etichette/cctv_fights
# CCTV-Fights (copia Kaggle mohamedtariki/vsd-cctv-fights, annotazioni originali NTU in groundtruth.json).
#
# Formato stile ActivityNet: database → id video → annotations [{segment: [inizio, fine] secondi, label: "Fight"}].
# Intervalli di rissa → violenza; resto del video → normale. Ogni video ha anche "source": CCTV, Mobile, Car, Other.
# Il groundtruth NON è nel repo (accordo di licenza NTU): si legge dal dataset Kaggle.
# ==============================================================================================================

def elenca_clip_cctv_fights(radice_dataset: str | Path, percorso_groundtruth: str | Path | None = None,
                fonti_ammesse: tuple[str, ...] | None = None) -> list[ClipAnnotata]:
    """`fonti_ammesse` = None tiene tutto; ("CCTV",) tiene solo le telecamere di sorveglianza (DA DECIDERE)."""
    radice_dataset = Path(radice_dataset)
    percorso_groundtruth = percorso_groundtruth or next(radice_dataset.glob("**/groundtruth.json"))
    database = json.loads(Path(percorso_groundtruth).read_text())["database"]
    percorso_per_id = {percorso.stem: percorso for percorso in radice_dataset.glob("**/fight_*.*")
                       if percorso.suffix in {".mp4", ".webm", ".avi", ".mkv"}}
    clip = []
    for id_video, dati in sorted(database.items()):
        if fonti_ammesse is not None and dati.get("source") not in fonti_ammesse:
            continue
        if id_video not in percorso_per_id:
            continue
        eventi = [(float(a["segment"][0]), float(a["segment"][1]), "violenza") for a in dati["annotations"]]
        clip.append(ClipAnnotata(
            id_clip=f"cctv_fights/{id_video}", sorgente="cctv_fights", gruppo=f"cctv_fights/{id_video}",
            percorso_video=str(percorso_per_id[id_video]), eventi_secondi=eventi, classe_sfondo="normale",
            split_originale=dati.get("subset"), telecamera_mobile=dati.get("source") != "CCTV",
            note=f"fonte={dati.get('source')}"))
    return clip


# ==============================================================================================================
# etichette/scvd
# SCVD (Kaggle toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd).
#
# Si usa la versione SCVD_converted (video interi); SCVD_converted_sec_split contiene gli stessi video tagliati a
# pezzi e userla insieme duplicherebbe i dati.
# ==============================================================================================================

CLASSE_PER_CARTELLA_SCVD = {"Normal": "normale", "Violence": "violenza", "Weaponized": "violenza_con_arma"}


def elenca_clip_scvd(radice_dataset: str | Path) -> list[ClipAnnotata]:
    clip = []
    for percorso in sorted(Path(radice_dataset).glob("**/SCVD_converted/*/*/*.avi")):
        cartella_classe, cartella_split = percorso.parent.name, percorso.parent.parent.name
        if cartella_classe not in CLASSE_PER_CARTELLA_SCVD:
            continue
        clip.append(ClipAnnotata(
            id_clip=f"scvd/{cartella_split}/{cartella_classe}/{percorso.stem}", sorgente="scvd",
            gruppo=f"scvd/{percorso.stem}", percorso_video=str(percorso),
            classe_intera=CLASSE_PER_CARTELLA_SCVD[cartella_classe], split_originale=cartella_split))
    return clip


# ==============================================================================================================
# etichette/ucf_uca
# UCF-Crime (Kaggle bypktt/ucf-crimes) con i tempi degli eventi da UCA (annotations/uca/, Apache 2.0). D10.
#
# UCA descrive TUTTI gli eventi di ogni video, anche quelli normali, con una frase e i secondi di inizio e fine.
# La classe di ogni frase si ricava dal testo con le regole a parole chiave qui sotto.
#
# Regole a parole chiave migliorate (D14): un verbo violento conta solo se riferito a una persona; gli urti di veicoli
# e le porte aperte a calci non contano. Le categorie RoadAccidents, Explosion, Arson, Vandalism, Shoplifting sono escluse.
# Si estraggono solo i segmenti candidati (± MARGINE_SECONDI); le finestre fuori dagli eventi si scartano.
# ==============================================================================================================

CARTELLA_UCA = Path(__file__).resolve().parents[2] / "annotations" / "uca"

CATEGORIE_UCF_USATE = ("Abuse", "Arrest", "Assault", "Burglary", "Fighting", "Robbery", "Shooting", "Stealing")
MARGINE_SECONDI = 2.0                    # PROVVISORIO: contesto prima e dopo ogni evento
NUMERO_VIDEO_NORMALI = 150               # PROVVISORIO
SECONDI_PER_VIDEO_NORMALE = 120.0        # PROVVISORIO

# --- Regole a parole chiave (D14: regole migliorate, 18/10) ---------------------------------------------------
PAROLE_ARMA = r"\b(gun|guns|pistol|handgun|rifle|shotgun|firearm|knife|knives|machete|dagger|stab\w*|shot|shoot\w*|fired|weapon\w*)\b"
PAROLE_MINACCIA = r"\b(point\w*|threat\w*|aim\w*|wav\w*|brandish\w*)\b"
# verbi violenti: contano solo se entro poche parole c'è una persona o una parte del corpo (niente porte, carrelli...)
VERBI_VIOLENZA = (r"(hit|hits|hitting|punch\w*|kick\w*|beat|beats|beating|beaten|slap\w*|push\w*|shov\w*|"
                  r"attack\w*|strangl\w*|chok\w*|assault\w*|knock\w*|grabb?\w*|drag\w*|threw|throw\w*|pull\w*|"
                  r"stomp\w*|smash\w*|wrestl\w*|tackl\w*)")
PERSONE = (r"(him|her|them|himself|herself|each other|one another|another|someone|somebody|man|men|woman|women|boy|"
           r"boys|girl|girls|child|children|kid|kids|person|people|guy|guys|clerk|cashier|guard|officer|policeman|"
           r"policemen|police|prisoner|victim|customer|customers|driver|passenger|robber|thief|suspect|owner|staff|"
           r"waiter|teacher|student|elderly|old \w+|baby|head|face|neck|back|body|chest|stomach|belly|leg|legs|arm|"
           r"arms|hair|ground)")
VIOLENZA_SU_PERSONA = rf"\b{VERBI_VIOLENZA}\b(\s+\S+){{0,4}}?\s+{PERSONE}\b"
NOMI_VIOLENZA = r"\b(fight|fights|fighting|fought|brawl\w*|scuffl\w*|altercation|punches|beat each other|tussl\w*)\b"
# frasi che sembrano violente ma non lo sono
ECCEZIONI_VIOLENZA = (r"\b(car|cars|taxi|truck|vehicle|bus|van|motorcycle|motorbike|tricycle|bicycle|bike|train)\b"
                      r"[^.]{0,30}\b(hit|hits|knock\w*|crash\w*)\b|\bkick\w* open\b|\bpush\w* open\b|"
                      r"\bpush\w* (a|the|his|her) (cart|stroller|wheelchair|door|trolley)")
PAROLE_SCIPPO = (r"\b(snatch\w*|grabb?\w* (the|his|her|a) \w*\s?(phone|bag|purse|necklace|chain|wallet|handbag|backpack)|"
                 r"robbed (him|her)|took (his|her) (phone|bag|purse|wallet|necklace))\b")


def descrive_violenza(testo: str) -> bool:
    if re.search(ECCEZIONI_VIOLENZA, testo) and not re.search(NOMI_VIOLENZA, testo):
        return False
    return bool(re.search(VIOLENZA_SU_PERSONA, testo) or re.search(NOMI_VIOLENZA, testo))


def classe_da_frase(frase: str) -> str | None:
    """Classe di una frase UCA, oppure None se la frase non descrive un evento utile."""
    testo = frase.lower()
    arma = re.search(PAROLE_ARMA, testo)
    violenza = descrive_violenza(testo)
    if arma and (violenza or re.search(PAROLE_MINACCIA, testo)):
        return "violenza_con_arma"
    if re.search(PAROLE_SCIPPO, testo) and not arma:  # "strappare la pistola/il coltello" è una colluttazione armata
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


def elenca_clip_ucf_uca(radice_dataset: str | Path, cartella_uca: Path = CARTELLA_UCA, seme: int = 0) -> list[ClipAnnotata]:
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
                classe_sfondo=None, note=f"categoria={categoria}"))

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


# ==============================================================================================================
# etichette/chain_snatching
# Chain-snatching (Kaggle aniketsahu09/chain-snatching-cctv-dataset): 170 video di scippi di collane (P1).
#
# Il dataset non ha i tempi dell'evento: li annota Marco a mano (D15) con `python -m sicurezza.strumenti annota-scippi` in
# annotations/chain_snatching/tempi_scippo.json. Diventano finestre "scippo" solo i tratti annotati; il resto del
# video si scarta (prima e dopo lo scippo la scena non è affidabilmente "normale"). Tutti i video restano comunque
# nell'elenco di estrazione, così l'estrazione non deve aspettare la fine dell'annotazione.
# ==============================================================================================================

FILE_TEMPI_SCIPPO = Path(__file__).resolve().parents[2] / "annotations" / "chain_snatching" / "tempi_scippo.json"


def elenca_clip_chain_snatching(radice_dataset: str | Path, file_tempi: Path = FILE_TEMPI_SCIPPO) -> list[ClipAnnotata]:
    tempi = json.loads(Path(file_tempi).read_text()) if Path(file_tempi).exists() else {}
    clip = []
    for percorso in sorted(Path(radice_dataset).glob("**/Chain_Snatching_Videos/*.mp4")):
        annotazione = tempi.get(percorso.name)
        if annotazione is None:
            eventi, nota = [], "non ancora annotato: nessuna finestra"
        elif annotazione["nessuno_scippo"]:
            eventi, nota = [], "nessuno scippo visibile: nessuna finestra"
        else:
            eventi, nota = [(float(inizio), float(fine), "scippo") for inizio, fine in annotazione["eventi"]], ""
        clip.append(ClipAnnotata(
            id_clip=f"chain_snatching/{percorso.stem}", sorgente="chain_snatching",
            gruppo=f"chain_snatching/{percorso.stem}", percorso_video=str(percorso), eventi_secondi=eventi,
            classe_sfondo=None, note=nota))
    return clip


# ==============================================================================================================
# prepara_elenchi
# Costruisce annotazioni.jsonl (ClipAnnotata) ed elenco.jsonl (VideoDaEstrarre) da tutte le sorgenti trovate.
#
# Su Kaggle i dataset sono montati sotto /kaggle/input: ogni modulo di etichette cerca i propri file da lì.
# Si estrae un SOVRAINSIEME (es. CCTV-Fights con anche i video da cellulare): così le scelte su cosa usare per
# l'addestramento restano aperte e non serve rifare l'estrazione.
#
# Uso:
#     python -m sicurezza.dati prepara-elenchi --input /kaggle/input --uscita /kaggle/working
# ==============================================================================================================

SORGENTI = {"rwf2000": elenca_clip_rwf2000, "cctv_fights": elenca_clip_cctv_fights, "scvd": elenca_clip_scvd,
            "ucf_uca": elenca_clip_ucf_uca, "chain_snatching": elenca_clip_chain_snatching}


def prepara(cartella_input: Path, cartella_uscita: Path, sorgenti=tuple(SORGENTI)):
    clip = []
    for nome in sorgenti:
        try:
            clip_sorgente = SORGENTI[nome](cartella_input)
        except StopIteration:  # es. groundtruth.json non trovato perché il dataset non è montato
            clip_sorgente = []
        print(f"{nome:16s} {len(clip_sorgente)} clip")
        clip += clip_sorgente
    cartella_uscita.mkdir(parents=True, exist_ok=True)
    scrivi_annotazioni(clip, cartella_uscita / "annotazioni.jsonl")
    # ordine alternato per sorgente: se la sessione si ferma a metà, ogni sorgente ha già una parte estratta
    per_sorgente = {}
    for singola_clip in clip:
        per_sorgente.setdefault(singola_clip.sorgente, []).append(singola_clip)
    alternate = [gruppo[indice] for indice in range(max(map(len, per_sorgente.values()), default=0))
                 for gruppo in per_sorgente.values() if indice < len(gruppo)]
    scrivi_elenco([singola_clip.da_estrarre() for singola_clip in alternate], cartella_uscita / "elenco.jsonl")
    secondi = Counter()
    for singola_clip in clip:
        if singola_clip.fine_secondi is not None:
            secondi[singola_clip.sorgente] += singola_clip.fine_secondi - (singola_clip.inizio_secondi or 0)
    for sorgente, totale in sorted(secondi.items()):
        print(f"  {sorgente}: {totale / 3600:.2f} ore note in anticipo (segmenti)")
    return clip


def comando_prepara_elenchi(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--sorgenti", nargs="*", default=list(SORGENTI))
    argomenti = parser.parse_args(argomenti)
    prepara(Path(argomenti.input), Path(argomenti.uscita), argomenti.sorgenti)


# ==============================================================================================================
# suddivisione
# Split train/val/test PER VIDEO ORIGINALE (P4): le finestre si sovrappongono, quindi finestre dello stesso
# video in split diversi falserebbero i risultati. Bilanciato per (sorgente, classe prevalente del gruppo).
# ==============================================================================================================

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

# ==============================================================================================================
# riga di comando: python -m sicurezza.dati <comando> [opzioni]
# ==============================================================================================================
COMANDI = {"prepara-elenchi": comando_prepara_elenchi}


def main(argomenti: list[str] | None = None) -> None:
    argomenti = sys.argv[1:] if argomenti is None else argomenti
    if not argomenti or argomenti[0] not in COMANDI:
        print("uso: python -m sicurezza.dati {prepara-elenchi} [opzioni]  (--help dopo il comando)")
        raise SystemExit(2)
    COMANDI[argomenti[0]](argomenti[1:])


if __name__ == "__main__":
    main()
