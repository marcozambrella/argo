"""Sicurezza NN — dati. File unico (richiesta di Marco: massimo 5 file Python)."""

from collections import Counter
from dataclasses import asdict, dataclass
from dataclasses import field
from pathlib import Path
import argparse
import json
import random
import re
import socket
import subprocess
import sys
import urllib.request
import zipfile

import numpy as np


# ==============================================================================================================
# config
# Costanti condivise da estrazione, training e demo (decisioni D2 e D5 nel diario).
# ==============================================================================================================

FPS_ANALISI = 25                 # tutti i video vengono letti a 25 fps
FRAME_PER_FINESTRA = 40          # 1,6 secondi
PASSO_FINESTRE = 10              # finestre 0-39, 10-49, 20-59, ...
FRAME_MINIMI_EVENTO = 20         # frame di evento necessari perché la finestra prenda la classe dell'evento

# cartella annotations/ del repo (questo file è in src/sicurezza/, due cartelle più giù)
CARTELLA_ANNOTAZIONI = Path(__file__).resolve().parents[2] / "annotations"

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
GOMITI = (NOMI_KEYPOINT.index("gomito_sx"), NOMI_KEYPOINT.index("gomito_dx"))   # stesso ordine dei polsi

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
    larghezza = int(flusso_video["width"])
    altezza = int(flusso_video["height"])

    # ffmpeg ruota automaticamente i video con metadato di rotazione: le dimensioni in uscita vanno scambiate
    rotazione = 0
    for dato in flusso_video.get("side_data_list", []):
        if "rotation" in dato:
            rotazione = int(dato["rotation"])
            break
    if rotazione % 180 != 0:
        larghezza, altezza = altezza, larghezza

    # la durata a volte manca nel flusso video: allora si prende quella del file
    durata = flusso_video.get("duration")
    if not durata:
        durata = dati.get("format", {}).get("duration")
    if not durata:
        durata = 0.0
    return InfoVideo(larghezza, altezza, float(durata))


def leggi_frame(percorso_video: str, fps: int = FPS_ANALISI, inizio_secondi=None, fine_secondi=None):
    """Frame BGR (altezza, larghezza, 3) uint8 a `fps` costanti, opzionalmente solo tra inizio e fine (secondi).

    È un generatore (`yield`): dà un frame alla volta, perché un video intero non starebbe in memoria."""
    info = leggi_info_video(percorso_video)
    comando = ["ffmpeg", "-v", "error", "-nostdin"]
    if inizio_secondi:
        comando += ["-ss", f"{inizio_secondi:.3f}"]
    comando += ["-i", str(percorso_video)]
    if fine_secondi is not None:
        durata = fine_secondi - (inizio_secondi or 0)
        comando += ["-t", f"{durata:.3f}"]
    comando += ["-vf", f"fps={fps}", "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    byte_per_frame = info.larghezza * info.altezza * 3
    processo_ffmpeg = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       bufsize=byte_per_frame)
    try:
        while True:
            dati_frame = processo_ffmpeg.stdout.read(byte_per_frame)
            if len(dati_frame) < byte_per_frame:
                break
            frame = np.frombuffer(dati_frame, np.uint8).reshape(info.altezza, info.larghezza, 3)
            yield frame
    finally:
        processo_ffmpeg.stdout.close()
        processo_ffmpeg.kill()
        processo_ffmpeg.wait()


# ==============================================================================================================
# finestre
# Finestre temporali di 40 frame con passo 10 ed etichettatura dalle annotazioni (D2 e P3 nel diario).
# ==============================================================================================================

# Un evento annotato è una tupla (primo frame, frame dopo l'ultimo, indice classe):
# l'intervallo è semiaperto [primo, fine), cioè il frame "fine" non fa parte dell'evento.


def inizi_finestre(numero_frame: int, frame_per_finestra: int = FRAME_PER_FINESTRA,
                   passo: int = PASSO_FINESTRE) -> np.ndarray:
    """Frame iniziali delle finestre: 0, 10, 20, ... finché la finestra sta tutta nel video."""
    if numero_frame < frame_per_finestra:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, numero_frame - frame_per_finestra + 1, passo)


def secondi_in_frame(inizio_secondi: float, fine_secondi: float, fps: int = FPS_ANALISI):
    primo_frame = round(inizio_secondi * fps)
    frame_dopo_ultimo = round(fine_secondi * fps)
    return primo_frame, frame_dopo_ultimo


def frame_coperti_per_classe(inizio_finestra: int, eventi: list,
                             frame_per_finestra: int = FRAME_PER_FINESTRA) -> dict:
    """Per ogni classe, quanti frame della finestra cadono in almeno un suo evento.
    Se due eventi della stessa classe si sovrappongono, i frame in comune si contano una volta sola."""
    fine_finestra = inizio_finestra + frame_per_finestra
    # per ogni classe una maschera di 40 frame: True dove c'è un evento di quella classe
    maschere_per_classe = {}
    for primo_frame, frame_dopo_ultimo, classe in eventi:
        da = max(primo_frame, inizio_finestra)
        a = min(frame_dopo_ultimo, fine_finestra)
        if da >= a:
            continue   # l'evento non tocca la finestra
        if classe not in maschere_per_classe:
            maschere_per_classe[classe] = np.zeros(frame_per_finestra, dtype=bool)
        maschere_per_classe[classe][da - inizio_finestra:a - inizio_finestra] = True

    coperti = {}
    for classe, maschera in maschere_per_classe.items():
        coperti[classe] = int(maschera.sum())
    return coperti


def etichetta_finestra(inizio_finestra: int, eventi: list, classe_sfondo,
                       frame_per_finestra: int = FRAME_PER_FINESTRA,
                       frame_minimi_evento: int = FRAME_MINIMI_EVENTO):
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
    classe_migliore = None
    for classe in classi_che_contano:
        if classe_migliore is None or copertura[classe] > copertura[classe_migliore]:
            classe_migliore = classe
    return classe_migliore


# ==============================================================================================================
# estrazione
# ==============================================================================================================

@dataclass
class VideoDaEstrarre:
    """Un video (o un suo segmento) da estrarre. `id_clip` diventa il nome del file .npz."""
    id_clip: str
    percorso_video: str
    inizio_secondi: float = None      # None = dall'inizio del video
    fine_secondi: float = None        # None = fino alla fine
    telecamera_mobile: bool = False   # D20: la compensazione del movimento (GMC) del tracker serve solo qui


def leggi_elenco(percorso_elenco) -> list:
    video_da_estrarre = []
    with open(percorso_elenco) as file_elenco:
        for riga in file_elenco:
            if not riga.strip():
                continue
            campi = json.loads(riga)
            # **campi: ogni chiave del dizionario diventa il campo con lo stesso nome
            video_da_estrarre.append(VideoDaEstrarre(**campi))
    return video_da_estrarre


def scrivi_elenco(video_da_estrarre: list, percorso_elenco) -> None:
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
    inizio_secondi: float = None
    fine_secondi: float = None
    # eventi (inizio, fine, classe) in secondi, relativi all'inizio della clip;
    # field(default_factory=list): ogni clip ha la sua lista vuota (non una lista condivisa da tutte)
    eventi_secondi: list = field(default_factory=list)
    classe_intera: str = None           # tutta la clip ha questa classe (dataset già tagliati)
    classe_sfondo: str = None           # classe delle finestre senza eventi (None = scartarle)
    split_originale: str = None
    telecamera_mobile: bool = False     # D20: accende la compensazione del movimento nel tracker
    note: str = ""

    def da_estrarre(self) -> VideoDaEstrarre:
        return VideoDaEstrarre(self.id_clip, self.percorso_video, self.inizio_secondi, self.fine_secondi,
                               self.telecamera_mobile)

    def eventi_in_frame(self, numero_frame: int) -> list:
        if self.classe_intera is not None:
            return [(0, numero_frame, INDICE_CLASSE[self.classe_intera])]
        eventi = []
        for inizio, fine, classe in self.eventi_secondi:
            primo_frame, frame_dopo_ultimo = secondi_in_frame(inizio, fine)
            eventi.append((primo_frame, frame_dopo_ultimo, INDICE_CLASSE[classe]))
        return eventi

    def indice_classe_sfondo(self):
        if self.classe_sfondo is None:
            return None
        return INDICE_CLASSE[self.classe_sfondo]


def scrivi_annotazioni(clip: list, percorso) -> None:
    with open(percorso, "w") as file_annotazioni:
        for singola_clip in clip:
            file_annotazioni.write(json.dumps(asdict(singola_clip)) + "\n")


def leggi_annotazioni(percorso) -> list:
    clip = []
    with open(percorso) as file_annotazioni:
        for riga in file_annotazioni:
            if not riga.strip():
                continue
            singola_clip = ClipAnnotata(**json.loads(riga))
            # in JSON le tuple diventano liste: si rimettono come tuple (inizio, fine, classe)
            eventi = []
            for evento in singola_clip.eventi_secondi:
                eventi.append(tuple(evento))
            singola_clip.eventi_secondi = eventi
            clip.append(singola_clip)
    return clip


# ==============================================================================================================
# etichette/rwf2000
# RWF-2000 (copia Kaggle vulamnguyen/rwf2000): clip di 5 s già tagliate. Fight → violenza, NonFight → normale.
# ==============================================================================================================

CLASSE_PER_CARTELLA_RWF2000 = {"Fight": "violenza", "NonFight": "normale"}


def elenca_clip_rwf2000(radice_dataset) -> list:
    clip = []
    for percorso in sorted(Path(radice_dataset).glob("**/RWF-2000/*/*/*.avi")):
        cartella_classe = percorso.parent.name
        cartella_split = percorso.parent.parent.name
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

def elenca_clip_cctv_fights(radice_dataset, percorso_groundtruth=None, fonti_ammesse=None) -> list:
    """`fonti_ammesse` = None tiene tutto; ("CCTV",) tiene solo le telecamere di sorveglianza (DA DECIDERE).

    Se il groundtruth.json non si trova (dataset non montato) solleva StopIteration: `prepara` la gestisce."""
    radice_dataset = Path(radice_dataset)
    if percorso_groundtruth is None:
        # next(...) prende il primo file trovato; se non ce n'è nessuno solleva StopIteration
        percorso_groundtruth = next(radice_dataset.glob("**/groundtruth.json"))
    database = json.loads(Path(percorso_groundtruth).read_text())["database"]

    percorso_per_id = {}
    for percorso in radice_dataset.glob("**/fight_*.*"):
        if percorso.suffix in {".mp4", ".webm", ".avi", ".mkv"}:
            percorso_per_id[percorso.stem] = percorso

    clip = []
    for id_video, dati in sorted(database.items()):
        if fonti_ammesse is not None and dati.get("source") not in fonti_ammesse:
            continue
        if id_video not in percorso_per_id:
            continue
        eventi = []
        for annotazione in dati["annotations"]:
            inizio, fine = annotazione["segment"]
            eventi.append((float(inizio), float(fine), "violenza"))
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
# D68: Marco ha segnato il tipo d'arma delle clip "Weaponized" (`strumenti annota-tipo-arma`). Dove l'arma non si
# capisce (lista vuota) la clip non è idonea all'addestramento e si toglie (Marco, 01/11).
FILE_TIPO_ARMA_SCVD = CARTELLA_ANNOTAZIONI / "armi" / "tipo_arma_scvd.json"


def elenca_clip_scvd(radice_dataset, file_tipo_arma: Path = FILE_TIPO_ARMA_SCVD) -> list:
    tipi_arma = {}
    if Path(file_tipo_arma).exists():
        tipi_arma = json.loads(Path(file_tipo_arma).read_text())
    clip = []
    for percorso in sorted(Path(radice_dataset).glob("**/SCVD_converted/*/*/*.avi")):
        cartella_classe = percorso.parent.name
        cartella_split = percorso.parent.parent.name
        if cartella_classe not in CLASSE_PER_CARTELLA_SCVD:
            continue
        nome_annotato = f"{cartella_split}__{percorso.stem}.mp4"   # i nomi delle clip convertite in mp4
        if cartella_classe == "Weaponized" and tipi_arma.get(nome_annotato) == []:
            continue   # arma che non si capisce: non idonea
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

CARTELLA_UCA = CARTELLA_ANNOTAZIONI / "uca"

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


def classe_da_frase(frase: str):
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


# D68: le armi si misurano a parte per armi da fuoco e coltelli. In UCF il tipo si ricava dalla frase;
# "weapon" da solo non dice il tipo (None).
PAROLE_ARMA_DA_FUOCO = r"\b(gun|guns|pistol|handgun|rifle|shotgun|firearm|shot|shoot\w*|fired)\b"
PAROLE_ARMA_BIANCA = r"\b(knife|knives|machete|dagger|stab\w*)\b"


def tipi_arma_da_frase(frase: str) -> list:
    """I tipi d'arma ("fuoco", "coltello") nominati in una frase UCA con violenza armata."""
    if classe_da_frase(frase) != "violenza_con_arma":
        return []
    testo = frase.lower()
    tipi = []
    if re.search(PAROLE_ARMA_DA_FUOCO, testo):
        tipi.append("fuoco")
    if re.search(PAROLE_ARMA_BIANCA, testo):
        tipi.append("coltello")
    return tipi


def tipi_arma_ucf(cartella_uca: Path = CARTELLA_UCA) -> dict:
    """id del video UCF → tipi d'arma delle sue frasi armate (solo i video che ne hanno)."""
    tipi_per_video = {}
    for id_video, dati in carica_uca(cartella_uca).items():
        tipi = []
        for frase in dati["sentences"]:
            for tipo in tipi_arma_da_frase(frase):
                if tipo not in tipi:
                    tipi.append(tipo)
        if len(tipi) > 0:
            tipi_per_video[id_video] = sorted(tipi)
    return tipi_per_video


def tipi_arma_scvd(percorso) -> dict:
    """id_clip SCVD → tipi d'arma, dal file di `strumenti annota-tipo-arma` (chiavi "Test__t_w001_converted.mp4")."""
    tipi_per_clip = {}
    for nome_file, tipi in json.loads(Path(percorso).read_text()).items():
        split, nome = Path(nome_file).stem.split("__")
        tipi_per_clip[f"scvd/{split}/Weaponized/{nome}"] = tipi
    return tipi_per_clip
# ------------------------------------------------------------------------------------------------------------


def carica_uca(cartella_uca: Path = CARTELLA_UCA) -> dict:
    annotazioni = {}
    for file_uca in sorted(Path(cartella_uca).glob("UCFCrime_*.json")):
        annotazioni.update(json.loads(file_uca.read_text()))
    return annotazioni


def categoria_ucf(id_video: str) -> str:
    if id_video.startswith("Normal"):
        return "Normal"
    # la categoria sono le lettere iniziali del nome: "Robbery048_x264" -> "Robbery"
    return re.match(r"[A-Za-z]+", id_video).group(0)


def unisci_intervalli(intervalli: list) -> list:
    """Unisce gli intervalli (inizio, fine) che si sovrappongono."""
    uniti = []
    for inizio, fine in sorted(intervalli):
        if len(uniti) > 0 and inizio <= uniti[-1][1]:
            inizio_precedente, fine_precedente = uniti[-1]
            uniti[-1] = (inizio_precedente, max(fine_precedente, fine))
        else:
            uniti.append((inizio, fine))
    return uniti


def elenca_clip_ucf_uca(radice_dataset, cartella_uca: Path = CARTELLA_UCA, seme: int = 0) -> list:
    percorso_per_id = {}
    for percorso in Path(radice_dataset).glob("**/*.mp4"):
        percorso_per_id[percorso.stem] = percorso
    annotazioni = carica_uca(cartella_uca)
    clip = []
    for id_video, dati in sorted(annotazioni.items()):
        categoria = categoria_ucf(id_video)
        if categoria not in CATEGORIE_UCF_USATE or id_video not in percorso_per_id:
            continue
        durata = float(dati["duration"])

        # ogni frase di UCA ha i suoi tempi: si tengono solo le frasi che descrivono un evento utile
        eventi = []
        for tempi, frase in zip(dati["timestamps"], dati["sentences"]):
            classe = classe_da_frase(frase)
            if classe is None:
                continue
            inizio, fine = tempi
            eventi.append((float(inizio), float(fine), classe))

        # attorno a ogni evento un po' di contesto; gli eventi vicini finiscono nello stesso segmento
        intervalli = []
        for inizio, fine, classe in eventi:
            intervalli.append((max(0.0, inizio - MARGINE_SECONDI), min(durata, fine + MARGINE_SECONDI)))
        segmenti = unisci_intervalli(intervalli)

        for inizio_segmento, fine_segmento in segmenti:
            # gli eventi del segmento, con i tempi contati dall'inizio del segmento
            eventi_nel_segmento = []
            for inizio, fine, classe in eventi:
                if inizio < fine_segmento and fine > inizio_segmento:
                    inizio_relativo = max(inizio, inizio_segmento) - inizio_segmento
                    fine_relativa = min(fine, fine_segmento) - inizio_segmento
                    eventi_nel_segmento.append((inizio_relativo, fine_relativa, classe))
            clip.append(ClipAnnotata(
                id_clip=f"ucf/{id_video}@{inizio_segmento:.1f}-{fine_segmento:.1f}", sorgente="ucf_uca",
                gruppo=f"ucf/{id_video}", percorso_video=str(percorso_per_id[id_video]),
                inizio_secondi=inizio_segmento, fine_secondi=fine_segmento, eventi_secondi=eventi_nel_segmento,
                classe_sfondo=None, note=f"categoria={categoria}"))

    # campione PROVVISORIO di video normali: un pezzo da ogni video, posizione casuale ma riproducibile
    generatore = random.Random(seme)
    normali = []
    for id_video in annotazioni:
        if categoria_ucf(id_video) == "Normal" and id_video in percorso_per_id:
            normali.append(id_video)
    normali.sort()
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

FILE_TEMPI_SCIPPO = CARTELLA_ANNOTAZIONI / "chain_snatching" / "tempi_scippo.json"


def elenca_clip_chain_snatching(radice_dataset, file_tempi: Path = FILE_TEMPI_SCIPPO) -> list:
    percorsi_video = sorted(Path(radice_dataset).glob("**/Chain_Snatching_Videos/*.mp4"))
    return clip_da_tempi_annotati(percorsi_video, file_tempi, "chain_snatching")


# D40: scippi di telefono da video YouTube con licenza CC BY (`strumenti scarica-video-cc`), annotati da Marco con
# `strumenti annota-scippi --video data/phone_snatching --uscita annotations/phone_snatching/tempi_scippo.json`.
FILE_TEMPI_SCIPPO_TELEFONO = CARTELLA_ANNOTAZIONI / "phone_snatching" / "tempi_scippo.json"


def elenca_clip_phone_snatching(radice_dataset, file_tempi: Path = FILE_TEMPI_SCIPPO_TELEFONO) -> list:
    percorsi_video = sorted(Path(radice_dataset).glob("**/phone_snatching/*.mp4"))
    return clip_da_tempi_annotati(percorsi_video, file_tempi, "phone_snatching")


def clip_da_tempi_annotati(percorsi_video: list, file_tempi: Path, sorgente: str) -> list:
    """Video annotati con lo strumento degli scippi: si usano solo i tratti segnati (D15, D22)."""
    tempi = {}
    if Path(file_tempi).exists():
        tempi = json.loads(Path(file_tempi).read_text())
    clip = []
    for percorso in percorsi_video:
        annotazione = tempi.get(percorso.name)
        if annotazione is not None and annotazione.get("non_idoneo"):
            continue   # Marco: non è una ripresa utilizzabile, non si estrae nemmeno
        eventi = []
        if annotazione is None:
            nota = "non ancora annotato: nessuna finestra"
        elif annotazione["nessuno_scippo"]:
            nota = "nessuno scippo visibile: nessuna finestra"
        else:
            nota = ""
            for inizio, fine in annotazione["eventi"]:
                eventi.append((float(inizio), float(fine), "scippo"))
        clip.append(ClipAnnotata(
            id_clip=f"{sorgente}/{percorso.stem}", sorgente=sorgente,
            gruppo=f"{sorgente}/{percorso.stem}", percorso_video=str(percorso), eventi_secondi=eventi,
            classe_sfondo=None, note=nota))
    return clip


# ==============================================================================================================
# dati nuovi (D68, 01/11): più scippi e più armi, scelti da Marco dopo la ricerca
# - Snatch 1.0 (Zenodo 10812645, CC BY-SA 4.0): telecamere del traffico di Hyderabad. 35 clip brevi di scippo,
#   annotate da Marco (annotations/snatch10/), e 29 registrazioni lunghe "Normal" delle stesse strade: da queste
#   si prendono pezzi normali, così "strada indiana" non vuol dire "scippo". Lo scarica il kernel da Zenodo.
# - scippi da Kaggle (gauravsingh72509/snatching-datasets, manutejachindukuru/snatching): si usano solo i video
#   annotati da Marco (annotations/scippi_kaggle/), che non contiene i doppioni del chain-snatching.
# - WVD (Kaggle thelarka/weapon-violence-dataset-wvd, CC0): video sintetici GTA-V. Solo train, mai val/test.
# ==============================================================================================================

FILE_TEMPI_SNATCH10 = CARTELLA_ANNOTAZIONI / "snatch10" / "tempi_scippo.json"
FILE_TEMPI_SCIPPI_KAGGLE = CARTELLA_ANNOTAZIONI / "scippi_kaggle" / "tempi_scippo.json"
PEZZI_PER_REGISTRAZIONE_NORMALE = 4
SECONDI_PER_PEZZO_NORMALE = 60.0


def elenca_clip_snatch10(radice_dataset, file_tempi: Path = FILE_TEMPI_SNATCH10) -> list:
    percorsi_video = sorted(Path(radice_dataset).glob("**/Snatch Theft/*.mp4"))
    return clip_da_tempi_annotati(percorsi_video, file_tempi, "snatch10")


def elenca_clip_snatch10_normali(radice_dataset, seme: int = 0) -> list:
    """Pezzi da un minuto, in posizioni casuali ma riproducibili, delle registrazioni lunghe senza scippi."""
    generatore = random.Random(seme)
    clip = []
    for percorso in sorted(Path(radice_dataset).glob("**/Chain Snatching Videos/Normal/*.avi")):
        durata = leggi_info_video(str(percorso)).durata_secondi
        for numero in range(PEZZI_PER_REGISTRAZIONE_NORMALE):
            inizio = generatore.uniform(0.0, max(0.0, durata - SECONDI_PER_PEZZO_NORMALE))
            fine = min(durata, inizio + SECONDI_PER_PEZZO_NORMALE)
            clip.append(ClipAnnotata(
                id_clip=f"snatch10_normali/{percorso.stem}@{inizio:.1f}-{fine:.1f}", sorgente="snatch10_normali",
                gruppo=f"snatch10_normali/{percorso.stem}", percorso_video=str(percorso), inizio_secondi=inizio,
                fine_secondi=fine, classe_intera="normale"))
    return clip


def elenca_clip_scippi_kaggle(radice_dataset, file_tempi: Path = FILE_TEMPI_SCIPPI_KAGGLE) -> list:
    annotati = set()
    if Path(file_tempi).exists():
        annotati = set(json.loads(Path(file_tempi).read_text()))
    percorsi_video = []
    for schema in ("**/SN_*.mp4", "**/snatching_*.mp4"):
        for percorso in sorted(Path(radice_dataset).glob(schema)):
            if percorso.name in annotati:   # gli altri sono doppioni del chain-snatching (non annotati apposta)
                percorsi_video.append(percorso)
    return clip_da_tempi_annotati(percorsi_video, file_tempi, "scippi_kaggle")


CLASSE_PER_CARTELLA_WVD = {"Hot": "violenza_con_arma", "Cold": "violenza_con_arma", "NotViolent": "normale"}


def elenca_clip_wvd(radice_dataset) -> list:
    """Nome dei file: "<hot|cold|...>-<arma o scena>-<ora del giorno>.avi"; le 6 ore della stessa arma = un gruppo."""
    clip = []
    for percorso in sorted(Path(radice_dataset).glob("**/WVD-VideoFormat/*/*.avi")):
        cartella_classe = percorso.parent.name
        if cartella_classe not in CLASSE_PER_CARTELLA_WVD:
            continue
        arma_o_scena = percorso.stem.split("-")[1]
        clip.append(ClipAnnotata(
            id_clip=f"wvd/{cartella_classe}/{percorso.stem}", sorgente="wvd",
            gruppo=f"wvd/{cartella_classe}/{arma_o_scena}", percorso_video=str(percorso),
            classe_intera=CLASSE_PER_CARTELLA_WVD[cartella_classe]))
    return clip


# tipo d'arma delle clip WVD armate, dal nome dell'arma; mazze, martelli, bottiglie... sono armi ma senza tipo
ARMI_DA_FUOCO_WVD = ("MG", "MGCombat", "PistolAP", "PistolCombat", "RifleBullpup", "RifleCarbine", "ShotgunAssault",
                     "ShotgunPump", "SMGMicro", "SMGMp5")
COLTELLI_WVD = ("Knife", "Machette")


SORGENTI_ESPERIMENTI = ("rwf2000", "cctv_fights", "scvd", "ucf_uca", "chain_snatching", "normali_folla", "snatch10",
                        "scippi_kaggle", "wvd")


def ritaglia_pezzi_normali_snatch10(radice_snatch10: Path, cartella_uscita: Path) -> int:
    """Taglia i pezzi di `elenca_clip_snatch10_normali` dalle registrazioni lunghe, uno per file
    ("<registrazione>@<inizio>-<fine>.mp4"): Zenodo dà ~2 MB/s a Kaggle, così i kernel dei pixel caricano
    ~1 GB invece di 12,5 (dataset privato argo-snatch10-ridotto)."""
    cartella_uscita.mkdir(parents=True, exist_ok=True)
    clip = elenca_clip_snatch10_normali(radice_snatch10)
    for singola in clip:
        nome = singola.id_clip.split("/", 1)[1]
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{singola.inizio_secondi:.3f}", "-i",
                        singola.percorso_video, "-t", f"{singola.fine_secondi - singola.inizio_secondi:.3f}",
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-an",
                        str(cartella_uscita / f"{nome}.mp4")], check=True)
    return len(clip)


def prepara_video_esperimenti(cartella_input: Path, cartella_dati: Path) -> dict:
    """D68, per i kernel che leggono i pixel (rete video, ritagli delle mani): prepara le scene di folla come
    l'estrazione normali, mette tutto sotto cartella_dati e aggiunge i pezzi normali di Snatch 1.0 già tagliati
    (dataset argo-snatch10-ridotto, cartella pezzi_normali/). Ritorna id_clip → ClipAnnotata."""
    # niente collegamenti a /kaggle/input: con Python ≥ 3.13 il glob "**" non entra nelle cartelle collegate (01/11:
    # così l'estrazione dei dati nuovi aveva perso scippi Kaggle e WVD). Elenchi separati e poi uniti.
    # Niente HiEve: il suo server (02/11) va a ~9 KB/s, 1,5 GB = giorni, e il timeout non scatta perché i dati
    # arrivano. Le clip HiEve restano senza video: i modelli dei pixel non le vedono e il confronto le salta.
    prepara_normali(cartella_input, cartella_dati, False)
    sorgenti_input = []
    for nome in SORGENTI_ESPERIMENTI:
        if nome != "normali_folla":
            sorgenti_input.append(nome)
    clip_per_id = {}
    for clip in prepara(cartella_input, cartella_dati / "elenchi_input", sorgenti_input):
        clip_per_id[clip.id_clip] = clip
    for clip in prepara(cartella_dati, cartella_dati / "elenchi_folla", ["normali_folla"]):
        clip_per_id[clip.id_clip] = clip
    # i pezzi hanno lo stesso id_clip dell'estrazione, ma il file comincia già all'inizio del pezzo
    for percorso in sorted(Path(cartella_input).glob("**/pezzi_normali/*.mp4")):
        id_clip = f"snatch10_normali/{percorso.stem}"
        registrazione = percorso.stem.split("@")[0]
        clip_per_id[id_clip] = ClipAnnotata(id_clip=id_clip, sorgente="snatch10_normali",
                                            gruppo=f"snatch10_normali/{registrazione}", percorso_video=str(percorso),
                                            classe_intera="normale")
    return clip_per_id


def tipi_arma_wvd(id_clip: str) -> list:
    arma = id_clip.split("/")[-1].split("-")[1]
    if arma in ARMI_DA_FUOCO_WVD:
        return ["fuoco"]
    if arma in COLTELLI_WVD:
        return ["coltello"]
    return []


# ==============================================================================================================
# anomalie (D69)
# NWPU Campus, CHAD e ATMA-V dicono solo quali fotogrammi sono anomali. Marco guarda ogni tratto anomalo (spezzone
# con 3 s di margine, `strumenti annota-anomalie`) e dà il tipo; qui si fanno le finestre per la rete video: il centro
# di ogni finestra da 1,6 s decide l'etichetta. Dentro uno spezzone e su un fotogramma anomalo → il tipo di Marco
# ("innocua" → normale); dentro uno spezzone ma non anomalo (il margine) → scartata; anomalo ma non segnato →
# scartata; il resto è normale delle stesse telecamere (una finestra ogni 2 s). Un video con uno spezzone
# "non_idoneo" esce tutto. Nomi degli spezzoni: <dataset>__<video>@<inizio s>-<fine s>.mp4.
# ==============================================================================================================

SECONDI_FINESTRA = FRAME_PER_FINESTRA / FPS_ANALISI
PASSO_ANOMALIE_SECONDI = 0.4
FINESTRE_ANOMALE_PER_NORMALE = 5   # normali: una finestra ogni 5 passi (2 s)


def tratti_anomali(anomalo: np.ndarray, fps: float) -> list:
    """(primo, ultimo) fotogramma di ogni tratto anomalo; i tratti separati da meno di 2 s si uniscono."""
    tratti = []
    indici = np.where(anomalo > 0)[0]
    if len(indici) == 0:
        return tratti
    primo = int(indici[0])
    precedente = int(indici[0])
    for indice in indici[1:]:
        indice = int(indice)
        if indice - precedente > 2.0 * fps:
            tratti.append((primo, precedente))
            primo = indice
        precedente = indice
    tratti.append((primo, precedente))
    return tratti


def spezzone_dal_nome(nome: str) -> tuple:
    """"chad__1_001_1@4.0-19.5.mp4" → ("chad__1_001_1", 4.0, 19.5)."""
    chiave_video, tempi = Path(nome).stem.rsplit("@", 1)
    inizio, fine = tempi.split("-")
    return chiave_video, float(inizio), float(fine)


def finestre_anomalie(anomalo: np.ndarray, fps: float, spezzoni: list) -> list:
    """anomalo: 0/1 per fotogramma del video; spezzoni: (inizio s, fine s, tipo) segnati da Marco per questo video.
    Ritorna (inizio finestra in secondi, tipo) con tipo "normale" o uno di quelli di Marco."""
    for inizio_spezzone, fine_spezzone, tipo in spezzoni:
        if tipo == "non_idoneo":
            return []
    finestre = []
    durata = len(anomalo) / fps
    passi = int((durata - SECONDI_FINESTRA) / PASSO_ANOMALIE_SECONDI) + 1
    for passo in range(max(0, passi)):
        inizio = passo * PASSO_ANOMALIE_SECONDI
        centro = inizio + SECONDI_FINESTRA / 2
        fotogramma = min(int(round(centro * fps)), len(anomalo) - 1)
        tipo_spezzone = None
        for inizio_spezzone, fine_spezzone, tipo in spezzoni:
            if inizio_spezzone <= centro <= fine_spezzone:
                tipo_spezzone = tipo
        if tipo_spezzone is not None:
            if anomalo[fotogramma] > 0:
                if tipo_spezzone == "innocua":
                    finestre.append((round(inizio, 2), "normale"))
                else:
                    finestre.append((round(inizio, 2), tipo_spezzone))
        elif anomalo[fotogramma] == 0 and passo % FINESTRE_ANOMALE_PER_NORMALE == 0:
            finestre.append((round(inizio, 2), "normale"))
    return finestre


# ==============================================================================================================
# normali_folla (D41, D46)
# Scene NORMALI in luoghi affollati (stazioni, metro, incroci, campus, strade), tutte classe "normale": la
# confusione più grande del modello è normale↔violenza. Fonti scelte dalla ricerca del sub-agente
# (data/annotazioni/ricerca_normali/dataset_normali.md, "opzione A + HiEve", ~3 h).
# `prepara_normali` mette tutto in <uscita>/normali_folla/<fonte>/: le sequenze di frame jpg (MOT20, HT21, MOT17)
# diventano mp4 al loro frame rate, HiEve si scarica (senza registrazione) e si tengono solo le scene senza violenza,
# per gli altri si fanno collegamenti ai video scelti. Poi `elenca_clip_normali_folla` li legge come una sorgente.
#
# Uso (sul kernel):  python -m sicurezza.dati prepara-normali --input /kaggle/input --uscita /tmp --hieve
# ==============================================================================================================

SEQUENZE_SCELTE = {   # cartelle di frame: tutte le MOT20; HT21 solo quelle nuove rispetto a MOT20 (Shibuya); MOT17 a
    "MOT20": None,    # camera fissa (copie -FRCNN: le -DPM/-SDP sono gli stessi frame)
    "HT21": ("HT21-03", "HT21-04", "HT21-13", "HT21-14"),
    "MOT17": ("MOT17-01-FRCNN", "MOT17-02-FRCNN", "MOT17-03-FRCNN", "MOT17-04-FRCNN", "MOT17-08-FRCNN",
              "MOT17-09-FRCNN"),
}
FPS_SEQUENZE_PREDEFINITI = {"MOT20": 25, "HT21": 25, "MOT17": 30}   # se manca seqinfo.ini
VIDEO_SHANGHAITECH_PER_SCENA = 6       # 13 scene × 6 ≈ 75 minuti (la scena 01 da sola ha 83 video)
MINUTI_UBI_FIGHTS_NORMALI = 45
URL_HIEVE = ("http://106.54.5.179/data/HIE20.zip", "http://106.54.5.179/data/HIE20test.zip")
PAROLE_HIEVE_DA_ESCLUDERE = ("fight", "robbery", "accident", "earthquake", "jail")   # risse, rapina, incidente, ...
ESTENSIONI_VIDEO = {".mp4", ".avi", ".mov", ".webm", ".mkv"}


def fps_della_sequenza(cartella_sequenza: Path, fonte: str) -> int:
    file_info = cartella_sequenza / "seqinfo.ini"
    if file_info.exists():
        trovato = re.search(r"frameRate\s*=\s*(\d+)", file_info.read_text())
        if trovato:
            return int(trovato.group(1))
    return FPS_SEQUENZE_PREDEFINITI[fonte]


def collega_video(video: Path, cartella: Path) -> None:
    """Mette in `cartella` un collegamento (symlink) al video, senza copiarlo."""
    destinazione = cartella / video.name
    destinazione.parent.mkdir(parents=True, exist_ok=True)
    if not destinazione.exists():
        destinazione.symlink_to(video.resolve())


def e_da_escludere_hieve(nome_file: str) -> bool:
    """Le scene HiEve con risse, rapine, incidenti... non sono normali."""
    if Path(nome_file).suffix.lower() not in ESTENSIONI_VIDEO:
        return True
    nome = Path(nome_file).stem.lower()
    for parola in PAROLE_HIEVE_DA_ESCLUDERE:
        if parola in nome:
            return True
    return False


def prepara_normali(cartella_input: Path, cartella_uscita: Path, scarica_hieve: bool = False) -> Counter:
    """Ritorna quanti video ci sono per fonte (un Counter: le fonti senza video valgono 0)."""
    radice = cartella_uscita / "normali_folla"
    conteggio = Counter()

    for fonte, scelte in SEQUENZE_SCELTE.items():
        for immagini in sorted(cartella_input.glob(f"**/{fonte}/*/{fonte}-*/img1")):
            sequenza = immagini.parent
            if scelte is not None and sequenza.name not in scelte:
                continue
            destinazione = radice / fonte.lower() / f"{sequenza.name}.mp4"
            destinazione.parent.mkdir(parents=True, exist_ok=True)
            if not destinazione.exists():
                # pad a lati pari (MOT20-04 è 1545×1080 e libx264/yuv420p li vuole pari): 1 px nero a destra/sotto,
                # le coordinate dei pixel restano le stesse
                fps = fps_della_sequenza(sequenza, fonte)
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
                                "-i", str(immagini / "%06d.jpg"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(destinazione)],
                               check=True)
            conteggio[fonte.lower()] += 1

    # ShanghaiTech: i video normali del training originale hanno il nome a 3 cifre (01_001.avi)
    per_scena = {}
    for video in sorted(cartella_input.glob("**/*.avi")):
        if re.fullmatch(r"\d\d_\d{3}", video.stem) and "shanghai" in str(video).lower():
            scena = video.stem[:2]
            if scena not in per_scena:
                per_scena[scena] = []
            per_scena[scena].append(video)
    for scena in sorted(per_scena):
        for video in per_scena[scena][:VIDEO_SHANGHAITECH_PER_SCENA]:
            collega_video(video, radice / "shanghaitech")
            conteggio["shanghaitech"] += 1

    # UBI-Fights: solo i normali (N_*), presi a passo regolare finché si arriva ai minuti voluti
    normali_ubi = sorted(cartella_input.glob("**/UBI_FIGHTS/normal/N_*.mp4"))
    secondi = 0.0
    for video in normali_ubi[::6]:   # uno ogni 6
        if secondi >= MINUTI_UBI_FIGHTS_NORMALI * 60:
            break
        secondi += leggi_info_video(str(video)).durata_secondi
        collega_video(video, radice / "ubi_fights")
        conteggio["ubi_fights"] += 1

    # Avenue: solo i video di training (tutti normali)
    for video in sorted(cartella_input.glob("**/training_videos/*.avi")):
        collega_video(video, radice / "avenue")
        conteggio["avenue"] += 1

    if scarica_hieve:
        socket.setdefaulttimeout(120)   # server unico in HTTP: se si blocca non deve appendere il kernel
        for url in URL_HIEVE:
            archivio = cartella_uscita / Path(url).name
            try:   # HiEve è un di più: se il download fallisce si va avanti con le altre fonti
                urllib.request.urlretrieve(url, archivio)
                with zipfile.ZipFile(archivio) as zip_hieve:
                    for nome in zip_hieve.namelist():
                        if e_da_escludere_hieve(nome):
                            continue
                        destinazione = radice / "hieve" / Path(nome).name
                        destinazione.parent.mkdir(parents=True, exist_ok=True)
                        destinazione.write_bytes(zip_hieve.read(nome))
                        conteggio["hieve"] += 1
            except (OSError, zipfile.BadZipFile) as errore:
                print(f"HiEve saltato ({url}): {errore}", flush=True)
            archivio.unlink(missing_ok=True)
    return conteggio


def elenca_clip_normali_folla(radice_dataset) -> list:
    clip = []
    for video in sorted(Path(radice_dataset).glob("**/normali_folla/*/*")):
        if video.suffix.lower() not in ESTENSIONI_VIDEO:
            continue
        fonte = video.parent.name
        # stessa scena = stesso gruppo (mai divisa tra train e test): ShanghaiTech per numero di scena, Avenue è
        # una sola telecamera, le altre fonti hanno una scena per video
        if fonte == "shanghaitech":
            gruppo = video.stem[:2]
        elif fonte == "avenue":
            gruppo = "unica"
        else:
            gruppo = video.stem
        clip.append(ClipAnnotata(id_clip=f"normali_folla/{fonte}/{video.stem}", sorgente=f"folla_{fonte}",
                                 gruppo=f"folla_{fonte}/{gruppo}", percorso_video=str(video), classe_intera="normale"))
    return clip


def comando_prepara_normali(argomenti=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--hieve", action="store_true", help="scarica anche HiEve (1,5 GB, senza registrazione)")
    argomenti = parser.parse_args(argomenti)
    conteggio = prepara_normali(Path(argomenti.input), Path(argomenti.uscita), argomenti.hieve)
    for fonte, numero in sorted(conteggio.items()):
        print(f"{fonte:14s} {numero} video")


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
            "ucf_uca": elenca_clip_ucf_uca, "chain_snatching": elenca_clip_chain_snatching,
            "phone_snatching": elenca_clip_phone_snatching, "normali_folla": elenca_clip_normali_folla,
            "snatch10": elenca_clip_snatch10, "snatch10_normali": elenca_clip_snatch10_normali,
            "scippi_kaggle": elenca_clip_scippi_kaggle, "wvd": elenca_clip_wvd}


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
    # ordine alternato per sorgente (1a clip di ogni sorgente, poi la 2a di ognuna, ...):
    # se la sessione si ferma a metà, ogni sorgente ha già una parte estratta
    per_sorgente = {}
    for singola_clip in clip:
        if singola_clip.sorgente not in per_sorgente:
            per_sorgente[singola_clip.sorgente] = []
        per_sorgente[singola_clip.sorgente].append(singola_clip)
    clip_piu_lunga = 0
    for clip_della_sorgente in per_sorgente.values():
        clip_piu_lunga = max(clip_piu_lunga, len(clip_della_sorgente))
    elenco = []
    for indice in range(clip_piu_lunga):
        for clip_della_sorgente in per_sorgente.values():
            if indice < len(clip_della_sorgente):
                elenco.append(clip_della_sorgente[indice].da_estrarre())
    scrivi_elenco(elenco, cartella_uscita / "elenco.jsonl")

    secondi = Counter()
    for singola_clip in clip:
        if singola_clip.fine_secondi is not None:
            inizio = singola_clip.inizio_secondi or 0
            secondi[singola_clip.sorgente] += singola_clip.fine_secondi - inizio
    for sorgente, totale in sorted(secondi.items()):
        print(f"  {sorgente}: {totale / 3600:.2f} ore note in anticipo (segmenti)")
    return clip


def comando_prepara_elenchi(argomenti=None) -> None:
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


# D26: video di UCF-Crime da cui vengono le immagini di CCTV Weapon Detect (il detector armi li ha visti in
# addestramento): nel classificatore stanno sempre in train, altrimenti il test sarebbe troppo ottimista.
FILE_VIDEO_UCF_NEL_DETECTOR = CARTELLA_ANNOTAZIONI / "armi" / "video_ucf_nel_detector.txt"


def gruppi_visti_dal_detector(percorso: Path = FILE_VIDEO_UCF_NEL_DETECTOR) -> set:
    gruppi = set()
    if not Path(percorso).exists():
        return gruppi
    for riga in Path(percorso).read_text().splitlines():
        if riga.strip():
            gruppi.add(f"ucf/{riga.strip()}")
    return gruppi


def dividi_per_gruppi(gruppi: np.ndarray, etichette: np.ndarray, sorgenti: np.ndarray,
                      quote=QUOTE_PREDEFINITE, seme: int = 0, gruppi_sempre_in_train=()) -> np.ndarray:
    """Ritorna per ogni finestra l'indice dello split (0 train, 1 val, 2 test).
    I `gruppi_sempre_in_train` (D26) vanno in train e non entrano nel bilanciamento degli altri."""
    # quante finestre ha ogni gruppo da distribuire, e da quale sorgente viene
    finestre_per_gruppo = Counter()
    sorgente_gruppo = {}
    for gruppo, sorgente in zip(gruppi.tolist(), sorgenti.tolist()):
        sorgente_gruppo[gruppo] = sorgente
        if gruppo not in gruppi_sempre_in_train:
            finestre_per_gruppo[gruppo] += 1

    classe_prevalente = {}
    for gruppo in finestre_per_gruppo:
        classi_gruppo = etichette[gruppi == gruppo]
        classe_prevalente[gruppo] = int(np.bincount(classi_gruppo).argmax())

    # strato = (sorgente, classe prevalente): ogni strato si divide 70/15/15 per conto suo
    strati = {}
    for gruppo in sorted(finestre_per_gruppo):
        chiave = (sorgente_gruppo[gruppo], classe_prevalente[gruppo])
        if chiave not in strati:
            strati[chiave] = []
        strati[chiave].append(gruppo)

    generatore = np.random.default_rng(seme)
    split_del_gruppo = {}
    soglie = np.cumsum(quote)   # (0.70, 0.85, 1.0)
    for chiave in sorted(strati):
        gruppi_strato = strati[chiave]
        generatore.shuffle(gruppi_strato)
        totale = 0
        for gruppo in gruppi_strato:
            totale += finestre_per_gruppo[gruppo]
        cumulate = 0
        for gruppo in gruppi_strato:
            # il gruppo va nello split in cui cade il suo punto centrale (in finestre cumulate)
            centro = (cumulate + finestre_per_gruppo[gruppo] / 2) / totale
            # searchsorted dice in quale intervallo delle soglie cade il centro: 0 train, 1 val, 2 test
            split_del_gruppo[gruppo] = int(np.searchsorted(soglie, centro, side="right"))
            cumulate += finestre_per_gruppo[gruppo]
    for gruppo in gruppi_sempre_in_train:
        split_del_gruppo[gruppo] = 0

    split = []
    for gruppo in gruppi.tolist():
        split.append(min(split_del_gruppo[gruppo], len(quote) - 1))
    return np.array(split, np.int64)

# ==============================================================================================================
# riga di comando: python -m sicurezza.dati <comando> [opzioni]
# ==============================================================================================================
COMANDI = {"prepara-elenchi": comando_prepara_elenchi, "prepara-normali": comando_prepara_normali}


def main(argomenti=None) -> None:
    argomenti = sys.argv[1:] if argomenti is None else argomenti
    if not argomenti or argomenti[0] not in COMANDI:
        print("uso: python -m sicurezza.dati {prepara-elenchi} [opzioni]  (--help dopo il comando)")
        raise SystemExit(2)
    COMANDI[argomenti[0]](argomenti[1:])


if __name__ == "__main__":
    main()
