"""Sicurezza NN — modelli. File unico (richiesta di Marco: massimo 5 file Python)."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import sys
import time

from torch import nn
import numpy as np
import torch
import torch.nn.functional as F

from .dati import CENTRO_SCHELETRO, CLASSI, FRAME_PER_FINESTRA, NOMI_SPLIT, NUMERO_KEYPOINT, OSSA, POLSI
from .dati import SPECCHIO_KEYPOINT
from .dati import dividi_per_gruppi, etichetta_finestra, gruppi_visti_dal_detector, inizi_finestre
from .dati import leggi_annotazioni, nome_file_estratto


# ==============================================================================================================
# caratteristiche
# Da un .npz estratto alle finestre-grafo: [40 frame × 70 nodi × 5 caratteristiche] (D7, D8 nel diario).
#
# Nodi di ogni frame: 4 persone × 17 keypoint, poi 2 nodi arma.
# Caratteristiche di ogni nodo: x, y relative al centro della scena e divise per l'altezza mediana delle persone
# (così distanza e zoom della telecamera contano poco), confidenza, velocità x, velocità y.
# ==============================================================================================================

NUMERO_PERSONE = 4                      # K di D8
NUMERO_ARMI = 2
NODI_PERSONE = NUMERO_PERSONE * NUMERO_KEYPOINT
NODI_PER_FRAME = NODI_PERSONE + NUMERO_ARMI
CARATTERISTICHE_PER_NODO = 5
SOGLIA_VISIBILITA_KEYPOINT = 0.1        # sotto questa confidenza YOLO mette il keypoint a (0, 0): lo si azzera


@dataclass
class TimelineVideo:
    """Dati di un .npz riorganizzati per accesso rapido a finestre."""
    numero_frame: int
    larghezza: int
    altezza: int
    keypoint_per_traccia: dict    # id traccia → array (numero_frame, 17, 3), NaN dove la persona manca
    box_per_traccia: dict         # id traccia → array (numero_frame, 4), NaN dove la persona manca
    armi_frame: np.ndarray
    armi_box: np.ndarray
    armi_classe: np.ndarray
    armi_confidenza: np.ndarray
    # (numero_frame, 3): ingrandimento sulle mani (D44), massimo sulle persone; None se il .npz non le ha
    mani_per_frame: np.ndarray = None


def carica_timeline(percorso_npz, ricuci: bool = True) -> TimelineVideo:
    return timeline_da_dati(np.load(percorso_npz), ricuci)


# I box sono [x1, y1, x2, y2]. Queste due funzioni vanno bene per un box solo o per un array di box:
# box[..., 0] vuol dire "la coordinata x1 di ogni box", qualunque sia la forma dell'array.
def area_box(box):
    larghezza = box[..., 2] - box[..., 0]
    altezza = box[..., 3] - box[..., 1]
    return larghezza * altezza


def centro_box(box):
    x = (box[..., 0] + box[..., 2]) / 2
    y = (box[..., 1] + box[..., 3]) / 2
    return np.stack([x, y], -1)


def armi_non_piu_grandi_della_persona(dati) -> np.ndarray:
    """D38: un'arma vera non è più grande di chi la impugna. Per ogni arma rilevata si prende la persona più vicina
    (centro dei box) nello stesso frame: se il riquadro dell'arma ha area maggiore, l'arma si scarta (riquadri su
    mezza scena: reti, barche, porte). Senza persone nel frame l'arma resta: non c'è niente con cui confrontarla."""
    # np.asarray una volta sola: un .npz rilegge (e decomprime) l'array a ogni accesso
    armi_frame = np.asarray(dati["armi_frame"])
    armi_box = np.asarray(dati["armi_box"])
    # persone ordinate per frame: così quelle di un frame stanno in righe consecutive
    ordine = np.argsort(dati["persone_frame"], kind="stable")
    persone_frame = np.asarray(dati["persone_frame"])[ordine]
    persone_box = np.asarray(dati["persone_box"])[ordine]
    # per ogni arma, le righe [inizi, fini) delle persone nel suo stesso frame
    inizi = np.searchsorted(persone_frame, armi_frame, "left")
    fini = np.searchsorted(persone_frame, armi_frame, "right")

    da_tenere = np.ones(len(armi_frame), bool)
    for indice in range(len(armi_frame)):
        if fini[indice] == inizi[indice]:
            continue   # nessuna persona nel frame: l'arma resta
        box_persone = persone_box[inizi[indice]:fini[indice]]
        distanze = np.linalg.norm(centro_box(box_persone) - centro_box(armi_box[indice]), axis=-1)
        persona_vicina = box_persone[np.argmin(distanze)]
        da_tenere[indice] = area_box(armi_box[indice]) <= area_box(persona_vicina)
    return da_tenere


def timeline_da_dati(dati, ricuci: bool = True, filtra_armi_grandi: bool = True) -> TimelineVideo:
    """Come carica_timeline, ma da un dizionario già in memoria (es. il buffer della demo)."""
    numero_frame = int(dati["numero_frame"])
    # ogni array si legge UNA volta: un .npz lo decomprime a ogni accesso (con migliaia di tracce, nelle scene di
    # folla, voleva dire ore per una clip); le righe di ogni traccia si prendono già raggruppate (argsort stabile)
    id_tracce = np.asarray(dati["persone_id_traccia"])
    frame_persone = np.asarray(dati["persone_frame"])
    keypoint_persone = np.asarray(dati["persone_keypoint"])
    box_persone = np.asarray(dati["persone_box"])
    ordine = np.argsort(id_tracce, kind="stable")
    # id_distinti: gli id delle tracce; inizi: dove comincia ognuna nelle righe ordinate
    id_distinti, inizi = np.unique(id_tracce[ordine], return_index=True)
    righe_per_traccia = np.split(ordine, inizi[1:])
    keypoint_per_traccia = {}
    box_per_traccia = {}
    for id_traccia, righe in zip(id_distinti, righe_per_traccia):
        frame_traccia = frame_persone[righe]
        keypoint = np.full((numero_frame, NUMERO_KEYPOINT, 3), np.nan, np.float32)
        box = np.full((numero_frame, 4), np.nan, np.float32)
        keypoint[frame_traccia] = keypoint_persone[righe]
        box[frame_traccia] = box_persone[righe]
        keypoint_per_traccia[int(id_traccia)] = keypoint
        box_per_traccia[int(id_traccia)] = box
    if ricuci:
        keypoint_per_traccia, box_per_traccia = ricuci_tracce(keypoint_per_traccia, box_per_traccia)

    if filtra_armi_grandi:
        armi = armi_non_piu_grandi_della_persona(dati)   # True per le armi da tenere
    else:
        armi = np.ones(len(dati["armi_frame"]), bool)

    mani_per_frame = None
    if "persone_mani" in dati:   # solo se il .npz è passato da `aggiungi-armi --modello-mani`
        mani_persone = np.asarray(dati["persone_mani"], np.float32)
        mani_per_frame = np.zeros((numero_frame, mani_persone.shape[1]), np.float32)
        # per ogni frame il massimo tra le persone: np.maximum.at lo fa anche quando un frame ha più righe
        np.maximum.at(mani_per_frame, frame_persone, mani_persone)

    return TimelineVideo(
        numero_frame=numero_frame, larghezza=int(dati["larghezza"]), altezza=int(dati["altezza"]),
        keypoint_per_traccia=keypoint_per_traccia, box_per_traccia=box_per_traccia,
        armi_frame=np.asarray(dati["armi_frame"])[armi], armi_box=np.asarray(dati["armi_box"])[armi],
        armi_classe=np.asarray(dati["armi_classe"])[armi],
        armi_confidenza=np.asarray(dati["armi_confidenza"])[armi].astype(np.float32),
        mani_per_frame=mani_per_frame)


BUCO_MASSIMO_FRAME = 15            # PROVVISORIO: 0,6 s senza la persona al massimo
DISTANZA_MASSIMA_ALTEZZE = 0.5     # PROVVISORIO: il nuovo frammento deve partire entro mezza altezza-corpo
RAPPORTO_ALTEZZE_MASSIMO = 1.5     # PROVVISORIO: stessa persona = altezza del box simile


def ricuci_tracce(keypoint_per_traccia: dict, box_per_traccia: dict, buco_massimo: int = BUCO_MASSIMO_FRAME,
                  distanza_massima: float = DISTANZA_MASSIMA_ALTEZZE,
                  rapporto_altezze_massimo: float = RAPPORTO_ALTEZZE_MASSIMO):
    """D17: unisce i frammenti di traccia della stessa persona.

    Una traccia B continua una traccia A se B inizia dopo la fine di A (mai sovrapposte nel tempo), entro
    `buco_massimo` frame, vicino all'ultimo box di A e con altezza simile. Tra più candidate si sceglie la più vicina
    (a parità, la prima nell'ordine delle tracce). Le tracce sono indicizzate per frame di fine: ogni traccia nuova
    guarda solo quelle finite nei `buco_massimo` frame prima (il confronto tutte-con-tutte era quadratico: ore sulle
    scene di folla con migliaia di tracce).
    """
    # copie: i dizionari e i box di chi chiama non vengono modificati
    keypoint_per_traccia = dict(keypoint_per_traccia)
    copia_box = {}
    for id_traccia, box in box_per_traccia.items():
        copia_box[id_traccia] = box.copy()
    box_per_traccia = copia_box

    # posizione di ogni traccia nell'ordine originale (serve per scegliere a parità di distanza)
    posizione = {}
    for indice, id_traccia in enumerate(box_per_traccia):
        posizione[id_traccia] = indice

    # estremi[id] = (primo frame, ultimo frame) in cui la traccia c'è
    estremi = {}
    for id_traccia, box in box_per_traccia.items():
        presenti = np.flatnonzero(~np.isnan(box[:, 0]))
        estremi[id_traccia] = (int(presenti[0]), int(presenti[-1]))

    # finite_al_frame[f] = insieme delle tracce che finiscono al frame f
    finite_al_frame = {}
    for id_traccia, (inizio, fine) in estremi.items():
        if fine not in finite_al_frame:
            finite_al_frame[fine] = set()
        finite_al_frame[fine].add(id_traccia)

    def primo_frame(id_traccia):
        return estremi[id_traccia][0]

    # le tracce in ordine di inizio: ognuna può diventare il seguito di una traccia finita poco prima
    for id_nuova in sorted(box_per_traccia, key=primo_frame):
        inizio_nuova = estremi[id_nuova][0]
        box_inizio_nuova = box_per_traccia[id_nuova][inizio_nuova]
        centro_nuova = centro_box(box_inizio_nuova)
        altezza_nuova = box_inizio_nuova[3] - box_inizio_nuova[1]

        # candidate: le tracce finite nei `buco_massimo` frame prima che questa cominci
        candidate = []
        for frame in range(inizio_nuova - buco_massimo, inizio_nuova):
            for id_vecchia in finite_al_frame.get(frame, ()):
                candidate.append(id_vecchia)
        candidate.sort(key=posizione.get)

        migliore = None
        distanza_migliore = np.inf
        for id_vecchia in candidate:
            box_fine_vecchia = box_per_traccia[id_vecchia][estremi[id_vecchia][1]]
            centro_vecchia = centro_box(box_fine_vecchia)
            altezza_vecchia = box_fine_vecchia[3] - box_fine_vecchia[1]
            # altezze troppo diverse: non è la stessa persona
            altezza_maggiore = max(altezza_vecchia, altezza_nuova)
            altezza_minore = max(min(altezza_vecchia, altezza_nuova), 1)
            if altezza_maggiore > rapporto_altezze_massimo * altezza_minore:
                continue
            distanza = np.linalg.norm(centro_nuova - centro_vecchia) / max(altezza_vecchia, 1)
            if distanza <= distanza_massima and distanza < distanza_migliore:
                migliore = id_vecchia
                distanza_migliore = distanza

        if migliore is not None:  # la traccia nuova diventa il seguito di quella vecchia (che ora finisce dove lei)
            presenti = ~np.isnan(box_per_traccia[id_nuova][:, 0])
            box_per_traccia[migliore][presenti] = box_per_traccia[id_nuova][presenti]
            keypoint_unito = keypoint_per_traccia[migliore].copy()
            keypoint_unito[presenti] = keypoint_per_traccia[id_nuova][presenti]
            keypoint_per_traccia[migliore] = keypoint_unito
            finite_al_frame[estremi[migliore][1]].discard(migliore)
            finite_al_frame[estremi[id_nuova][1]].discard(id_nuova)
            estremi[migliore] = (estremi[migliore][0], estremi[id_nuova][1])
            finite_al_frame[estremi[migliore][1]].add(migliore)
            del box_per_traccia[id_nuova]
            del keypoint_per_traccia[id_nuova]
            del estremi[id_nuova]
    return keypoint_per_traccia, box_per_traccia


def punteggio_presenza_e_movimento(keypoint: np.ndarray, box: np.ndarray) -> float:
    """Criterio PROVVISORIO per scegliere le 4 persone (DA DECIDERE nel diario):
    frame in cui la persona c'è × (1 + movimento medio dei keypoint in altezze-corpo per frame)."""
    presente = ~np.isnan(box[:, 0])
    if presente.sum() < 2:
        return float(presente.sum())
    altezza_corpo = np.nanmedian(box[:, 3] - box[:, 1]) + 1e-6
    spostamenti = np.linalg.norm(np.diff(keypoint[:, :, :2], axis=0), axis=-1) / altezza_corpo
    if np.isfinite(spostamenti).any():
        movimento_medio = float(np.nanmean(spostamenti))
    else:
        movimento_medio = 0.0
    return float(presente.sum()) * (1.0 + movimento_medio)


CRITERI_SCELTA_PERSONE = {"presenza_e_movimento": punteggio_presenza_e_movimento}


def costruisci_finestra(timeline: TimelineVideo, inizio_finestra: int,
                        frame_per_finestra: int = FRAME_PER_FINESTRA,
                        criterio_scelta_persone: str = "presenza_e_movimento"):
    """Ritorna (nodi [T, 70, 5] float32, maschera [T, 70] bool, classe_arma [T, 2] int8 con 0 = nessuna arma)."""
    fine_finestra = inizio_finestra + frame_per_finestra
    punteggio = CRITERI_SCELTA_PERSONE[criterio_scelta_persone]

    # le persone presenti nella finestra, ognuna come (punteggio, keypoint, box)
    tracce_in_finestra = []
    for id_traccia, box in timeline.box_per_traccia.items():
        box_finestra = box[inizio_finestra:fine_finestra]
        if not np.isnan(box_finestra[:, 0]).all():
            keypoint_finestra = timeline.keypoint_per_traccia[id_traccia][inizio_finestra:fine_finestra]
            tracce_in_finestra.append((punteggio(keypoint_finestra, box_finestra), keypoint_finestra, box_finestra))

    # si tengono le 4 persone con il punteggio più alto
    def punteggio_della_traccia(traccia):
        return traccia[0]
    tracce_in_finestra.sort(key=punteggio_della_traccia, reverse=True)
    tracce_scelte = tracce_in_finestra[:NUMERO_PERSONE]

    # scala e centro della scena: altezza mediana delle persone scelte, centro medio dei loro box
    if len(tracce_scelte) > 0:
        box_scelti = []
        for punteggio_traccia, keypoint_finestra, box_finestra in tracce_scelte:
            box_scelti.append(box_finestra)
        tutti_i_box = np.concatenate(box_scelti)
        altezza_riferimento = float(np.nanmedian(tutti_i_box[:, 3] - tutti_i_box[:, 1]))
        centro_x = np.nanmean((tutti_i_box[:, 0] + tutti_i_box[:, 2]) / 2)
        centro_y = np.nanmean((tutti_i_box[:, 1] + tutti_i_box[:, 3]) / 2)
        centro_scena = np.array([centro_x, centro_y], np.float32)
    else:
        altezza_riferimento = 0.0
        centro_scena = np.array([timeline.larghezza / 2, timeline.altezza / 2], np.float32)
    if not np.isfinite(altezza_riferimento) or altezza_riferimento < 1:
        altezza_riferimento = 0.3 * timeline.altezza

    # le coordinate dei nodi si scrivono come (punto - centro_scena) / altezza_riferimento
    nodi = np.zeros((frame_per_finestra, NODI_PER_FRAME, CARATTERISTICHE_PER_NODO), np.float32)
    maschera = np.zeros((frame_per_finestra, NODI_PER_FRAME), bool)
    for posto_persona in range(len(tracce_scelte)):
        keypoint_finestra = tracce_scelte[posto_persona][1]
        # i 17 nodi di questa persona: da posto*17 a (posto+1)*17
        nodi_persona = slice(posto_persona * NUMERO_KEYPOINT, (posto_persona + 1) * NUMERO_KEYPOINT)
        presente = ~np.isnan(keypoint_finestra[:, 0, 0])                       # [frame]
        confidenza = np.nan_to_num(keypoint_finestra[:, :, 2])                 # [frame, 17]
        visibile = presente[:, None] & (confidenza >= SOGLIA_VISIBILITA_KEYPOINT)
        coordinate = (np.nan_to_num(keypoint_finestra[:, :, :2]) - centro_scena) / altezza_riferimento
        posizione = np.where(visibile[..., None], coordinate, 0.0)
        nodi[:, nodi_persona, 0:2] = posizione
        nodi[:, nodi_persona, 2] = np.where(visibile, confidenza, 0.0)
        maschera[:, nodi_persona] = presente[:, None]
        # velocità solo dove il keypoint è visibile sia in questo frame sia nel precedente
        visibile_anche_prima = visibile[1:] & visibile[:-1]
        nodi[1:, nodi_persona, 3:5] = np.where(visibile_anche_prima[..., None], np.diff(posizione, axis=0), 0.0)

    classe_arma = np.zeros((frame_per_finestra, NUMERO_ARMI), np.int8)
    nella_finestra = (timeline.armi_frame >= inizio_finestra) & (timeline.armi_frame < fine_finestra)
    for indice_arma in np.flatnonzero(nella_finestra):
        frame_relativo = timeline.armi_frame[indice_arma] - inizio_finestra
        posti_liberi = np.flatnonzero(classe_arma[frame_relativo] == 0)
        confidenza = timeline.armi_confidenza[indice_arma]
        if len(posti_liberi):
            posto = posti_liberi[0]
        else:  # già 2 armi in questo frame: tiene le 2 più sicure
            nodi_armi = nodi[frame_relativo, NODI_PERSONE:, 2]
            posto = int(np.argmin(nodi_armi))
            if confidenza <= nodi_armi[posto]:
                continue
        box = timeline.armi_box[indice_arma]
        nodo = NODI_PERSONE + posto
        centro_arma = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        nodi[frame_relativo, nodo, 0:2] = (centro_arma - centro_scena) / altezza_riferimento
        nodi[frame_relativo, nodo, 2] = confidenza
        maschera[frame_relativo, nodo] = True
        classe_arma[frame_relativo, posto] = timeline.armi_classe[indice_arma] + 1
    return nodi, maschera, classe_arma


# ==============================================================================================================
# dataset_finestre
# Dalle estrazioni (.npz) e dalle annotazioni (ClipAnnotata) al dataset di finestre etichettate.
#
# Uso:
#     python -m sicurezza.modelli dataset-finestre --annotazioni annotazioni.jsonl --estratti estratti/ --uscita finestre.npz
# ==============================================================================================================

def costruisci_dataset(clip: list, cartelle_estratti: list) -> dict:
    percorso_per_nome = {}
    for cartella in cartelle_estratti:
        for file in Path(cartella).glob("*.npz"):
            percorso_per_nome[file.name] = file

    nodi = []
    maschere = []
    classi_arma = []
    etichette = []
    gruppi = []
    sorgenti = []
    id_clip = []
    inizi = []
    clip_mancanti = 0
    for singola_clip in clip:
        percorso_npz = percorso_per_nome.get(nome_file_estratto(singola_clip.id_clip))
        if percorso_npz is None:
            clip_mancanti += 1
            continue
        timeline = carica_timeline(percorso_npz)
        eventi = singola_clip.eventi_in_frame(timeline.numero_frame)
        for inizio in inizi_finestre(timeline.numero_frame):
            inizio = int(inizio)
            classe = etichetta_finestra(inizio, eventi, singola_clip.indice_classe_sfondo())
            if classe is None:
                continue   # finestra ambigua: scartata
            nodi_finestra, maschera, classe_arma = costruisci_finestra(timeline, inizio)
            nodi.append(nodi_finestra.astype(np.float16))
            maschere.append(maschera)
            classi_arma.append(classe_arma)
            etichette.append(classe)
            gruppi.append(singola_clip.gruppo)
            sorgenti.append(singola_clip.sorgente)
            id_clip.append(singola_clip.id_clip)
            inizi.append(inizio)
    if clip_mancanti:
        print(f"ATTENZIONE: {clip_mancanti} clip senza .npz (non ancora estratte o errori di estrazione)")
    return {"nodi": np.stack(nodi), "maschera": np.stack(maschere), "classe_arma": np.stack(classi_arma),
            "etichetta": np.array(etichette, np.int64), "gruppo": np.array(gruppi), "sorgente": np.array(sorgenti),
            "id_clip": np.array(id_clip), "inizio_finestra": np.array(inizi, np.int32)}


def riepilogo(dataset: dict) -> str:
    righe = [f"{len(dataset['etichetta'])} finestre"]
    conteggio = Counter()
    for sorgente, classe in zip(dataset["sorgente"].tolist(), dataset["etichetta"].tolist()):
        conteggio[(sorgente, classe)] += 1
    for (sorgente, classe), numero in sorted(conteggio.items()):
        righe.append(f"  {sorgente:16s} {CLASSI[classe]:18s} {numero}")
    return "\n".join(righe)


def comando_dataset_finestre(argomenti=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotazioni", required=True)
    parser.add_argument("--estratti", nargs="+", required=True)
    parser.add_argument("--uscita", required=True)
    argomenti = parser.parse_args(argomenti)
    dataset = costruisci_dataset(leggi_annotazioni(argomenti.annotazioni), argomenti.estratti)
    np.savez(argomenti.uscita, **dataset)   # ogni chiave del dizionario diventa un array del .npz
    print(riepilogo(dataset))


# ==============================================================================================================
# aumenti
# Data augmentation sui lotti di finestre (su GPU). Probabilità e ampiezze PROVVISORIE.
# ==============================================================================================================

PROBABILITA_RIBALTAMENTO = 0.5
INTERVALLO_SCALA = (0.9, 1.1)
TRASLAZIONE_MASSIMA = 0.2          # in altezze-corpo
PROBABILITA_SPEGNI_KEYPOINT = 0.05


def aumenta_lotto(nodi: torch.Tensor, maschera: torch.Tensor, classe_arma: torch.Tensor):
    """nodi [B, T, 70, 5] (x, y, confidenza, vx, vy). Ritorna nuovi tensori, gli originali non vengono toccati."""
    nodi, maschera, classe_arma = nodi.clone(), maschera.clone(), classe_arma.clone()
    lotto, frame = nodi.shape[:2]
    dispositivo = nodi.device
    visibile = nodi[..., 2] > 0

    # 1. ribaltamento orizzontale: x e vx cambiano segno, sinistra e destra si scambiano
    ribalta = torch.rand(lotto, device=dispositivo) < PROBABILITA_RIBALTAMENTO
    if ribalta.any():
        specchio = torch.tensor(SPECCHIO_KEYPOINT, device=dispositivo)
        persone = nodi[ribalta, :, :NODI_PERSONE].view(-1, frame, NUMERO_PERSONE, NUMERO_KEYPOINT, 5)
        nodi[ribalta, :, :NODI_PERSONE] = persone[:, :, :, specchio].reshape(-1, frame, NODI_PERSONE, 5)
        nodi[ribalta, ..., 0] *= -1
        nodi[ribalta, ..., 3] *= -1
        visibile = nodi[..., 2] > 0

    # 2. scala e traslazione (solo sui nodi visibili: quelli invisibili restano a zero)
    scala_minima, scala_massima = INTERVALLO_SCALA
    scala = torch.empty(lotto, 1, 1, 1, device=dispositivo).uniform_(scala_minima, scala_massima)
    traslazione = torch.empty(lotto, 1, 1, 2, device=dispositivo).uniform_(-TRASLAZIONE_MASSIMA, TRASLAZIONE_MASSIMA)
    posizioni = nodi[..., 0:2] * scala + traslazione
    nodi[..., 0:2] = torch.where(visibile[..., None], posizioni, nodi[..., 0:2])
    nodi[..., 3:5] *= scala

    # 3. keypoint spenti a caso (come se YOLO non li vedesse)
    spegni = (torch.rand(lotto, frame, NODI_PERSONE, device=dispositivo) < PROBABILITA_SPEGNI_KEYPOINT)
    nodi[:, :, :NODI_PERSONE][spegni] = 0.0

    # 4. ordine delle persone casuale: il modello non deve dipendere dal "posto"
    for indice in range(lotto):
        ordine = torch.randperm(NUMERO_PERSONE, device=dispositivo)
        persone = nodi[indice, :, :NODI_PERSONE].view(frame, NUMERO_PERSONE, NUMERO_KEYPOINT, 5)
        nodi[indice, :, :NODI_PERSONE] = persone[:, ordine].reshape(frame, NODI_PERSONE, 5)
        maschera_persone = maschera[indice, :, :NODI_PERSONE].view(frame, NUMERO_PERSONE, NUMERO_KEYPOINT)
        maschera[indice, :, :NODI_PERSONE] = maschera_persone[:, ordine].reshape(frame, NODI_PERSONE)
    return nodi, maschera, classe_arma


# ==============================================================================================================
# modelli/graph_transformer
# Graph-Transformer sulle finestre-grafo (D11): relazioni tra keypoint della stessa persona, tra persone diverse
# e tra persone e armi.
#
# Ogni blocco fa:
#   1. attenzione SPAZIALE tra i 70 nodi di ogni frame, con un bias appreso per tipo di arco (osso, stessa persona,
#      persone diverse, arma-polso, ...) e un bias che penalizza la distanza per gli archi tra entità diverse;
#   2. attenzione TEMPORALE di ogni nodo lungo i 40 frame;
#   3. rete feed-forward.
# Iperparametri PROVVISORI (dimensione 128, 4 teste, 4 blocchi): da decidere con Marco dopo i primi esperimenti.
# ==============================================================================================================

# tipi di arco tra due nodi dello stesso frame
ARCO_SE_STESSO = 0
ARCO_OSSO = 1
ARCO_STESSA_PERSONA = 2
ARCO_PERSONE_DIVERSE = 3
ARCO_ARMA_POLSO = 4
ARCO_ARMA_CORPO = 5
ARCO_ARMA_ARMA = 6
NUMERO_TIPI_ARCO = 7
ARCHI_TRA_ENTITA_DIVERSE = (ARCO_PERSONE_DIVERSE, ARCO_ARMA_POLSO, ARCO_ARMA_CORPO, ARCO_ARMA_ARMA)


def matrice_tipi_arco() -> torch.Tensor:
    """[70, 70] con il tipo di arco tra ogni coppia di nodi."""
    # per ogni nodo: di quale persona è e quale keypoint è (-1 per i 2 nodi arma)
    persona_del_nodo = []
    keypoint_del_nodo = []
    for nodo in range(NODI_PERSONE):
        persona_del_nodo.append(nodo // NUMERO_KEYPOINT)
        keypoint_del_nodo.append(nodo % NUMERO_KEYPOINT)
    for arma in range(NUMERO_ARMI):
        persona_del_nodo.append(-1)
        keypoint_del_nodo.append(-1)

    tipi = torch.empty(NODI_PER_FRAME, NODI_PER_FRAME, dtype=torch.long)
    for i in range(NODI_PER_FRAME):
        for j in range(NODI_PER_FRAME):
            arma_i = persona_del_nodo[i] < 0
            arma_j = persona_del_nodo[j] < 0
            coppia = (keypoint_del_nodo[i], keypoint_del_nodo[j])
            coppia_al_contrario = (keypoint_del_nodo[j], keypoint_del_nodo[i])
            if i == j:
                tipo = ARCO_SE_STESSO
            elif arma_i and arma_j:
                tipo = ARCO_ARMA_ARMA
            elif arma_i or arma_j:
                if arma_i:
                    keypoint_persona = keypoint_del_nodo[j]
                else:
                    keypoint_persona = keypoint_del_nodo[i]
                if keypoint_persona in POLSI:
                    tipo = ARCO_ARMA_POLSO
                else:
                    tipo = ARCO_ARMA_CORPO
            elif persona_del_nodo[i] != persona_del_nodo[j]:
                tipo = ARCO_PERSONE_DIVERSE
            elif coppia in OSSA or coppia_al_contrario in OSSA:
                tipo = ARCO_OSSO
            else:
                tipo = ARCO_STESSA_PERSONA
            tipi[i, j] = tipo
    return tipi


def maschera_attenzione(chiavi_valide: torch.Tensor) -> torch.Tensor:
    """[gruppi, L] bool → [gruppi, 1, L, L] float con -inf sulle chiavi non valide.
    Ogni nodo vede sempre se stesso, così una riga tutta mascherata non produce NaN."""
    lunghezza = chiavi_valide.shape[-1]
    vede = chiavi_valide[:, None, :] | torch.eye(lunghezza, dtype=torch.bool, device=chiavi_valide.device)
    return torch.zeros(vede.shape, device=chiavi_valide.device).masked_fill(~vede, float("-inf"))[:, None]


class BloccoSpazioTempo(nn.Module):
    def __init__(self, dimensione: int, teste: int, dropout: float):
        super().__init__()
        self.teste = teste
        self.norma_spazio = nn.LayerNorm(dimensione)
        self.qkv_spazio = nn.Linear(dimensione, 3 * dimensione)
        self.uscita_spazio = nn.Linear(dimensione, dimensione)
        self.bias_tipo_arco = nn.Parameter(torch.zeros(teste, NUMERO_TIPI_ARCO))
        self.peso_distanza = nn.Parameter(torch.zeros(teste))   # softplus(0) ≈ 0,69 per testa all'inizio
        self.norma_tempo = nn.LayerNorm(dimensione)
        self.qkv_tempo = nn.Linear(dimensione, 3 * dimensione)
        self.uscita_tempo = nn.Linear(dimensione, dimensione)
        self.norma_ff = nn.LayerNorm(dimensione)
        self.feed_forward = nn.Sequential(nn.Linear(dimensione, 4 * dimensione), nn.GELU(), nn.Dropout(dropout),
                                          nn.Linear(4 * dimensione, dimensione))
        self.dropout = nn.Dropout(dropout)

    def attenzione(self, x, qkv, uscita, bias):
        gruppi, lunghezza, dimensione = x.shape
        q, k, v = qkv(x).view(gruppi, lunghezza, 3, self.teste, -1).permute(2, 0, 3, 1, 4)
        if self.training:
            probabilita_dropout = self.dropout.p
        else:
            probabilita_dropout = 0.0
        risultato = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=probabilita_dropout)
        return uscita(risultato.transpose(1, 2).reshape(gruppi, lunghezza, dimensione))

    def forward(self, x, maschera, tipi_arco, distanze):
        """x [B, T, N, D], maschera [B, T, N], tipi_arco [N, N], distanze [B, T, N, N]."""
        lotto, frame, nodi, dimensione = x.shape

        # 1. spazio: bias per tipo di arco + penalità di distanza sugli archi tra entità diverse
        bias_tipo = self.bias_tipo_arco[:, tipi_arco]                                     # [H, N, N]
        tra_entita = torch.zeros_like(tipi_arco, dtype=torch.bool)
        for tipo in ARCHI_TRA_ENTITA_DIVERSE:
            tra_entita |= tipi_arco == tipo
        penalita = -F.softplus(self.peso_distanza)[None, :, None, None] * (distanze * tra_entita).reshape(
            lotto * frame, 1, nodi, nodi)                                                # [B*T, H, N, N]
        bias = bias_tipo[None] + penalita + maschera_attenzione(maschera.reshape(lotto * frame, nodi))
        spazio = self.norma_spazio(x).reshape(lotto * frame, nodi, dimensione)
        x = x + self.dropout(self.attenzione(spazio, self.qkv_spazio, self.uscita_spazio, bias)).view_as(x)

        # 2. tempo: ogni nodo guarda se stesso negli altri frame in cui è presente
        tempo = self.norma_tempo(x).transpose(1, 2).reshape(lotto * nodi, frame, dimensione)
        bias_tempo = maschera_attenzione(maschera.transpose(1, 2).reshape(lotto * nodi, frame))
        aggiornamento = self.attenzione(tempo, self.qkv_tempo, self.uscita_tempo, bias_tempo)
        x = x + self.dropout(aggiornamento.view(lotto, nodi, frame, dimensione).transpose(1, 2))

        # 3. feed-forward
        return x + self.dropout(self.feed_forward(self.norma_ff(x)))


class GraphTransformer(nn.Module):
    def __init__(self, dimensione: int = 128, teste: int = 4, blocchi: int = 4, dropout: float = 0.1,
                 numero_classi: int = len(CLASSI), numero_classi_arma: int = 3):
        super().__init__()
        self.proiezione_ingresso = nn.Linear(CARATTERISTICHE_PER_NODO, dimensione)
        self.tipo_nodo = nn.Embedding(NUMERO_KEYPOINT + 1, dimensione)        # 17 keypoint + "arma"
        self.posto_nodo = nn.Embedding(NUMERO_PERSONE + 1, dimensione)        # 4 persone + "posto armi"
        self.classe_arma = nn.Embedding(numero_classi_arma + 1, dimensione)   # 0 = nessuna arma
        self.posizione_temporale = nn.Parameter(torch.zeros(FRAME_PER_FINESTRA, dimensione))
        for parametro in (self.posizione_temporale, self.tipo_nodo.weight, self.posto_nodo.weight,
                          self.classe_arma.weight):
            nn.init.normal_(parametro, std=0.02)
        lista_blocchi = []
        for numero in range(blocchi):
            lista_blocchi.append(BloccoSpazioTempo(dimensione, teste, dropout))
        self.blocchi = nn.ModuleList(lista_blocchi)
        self.norma_finale = nn.LayerNorm(dimensione)
        self.testa = nn.Sequential(nn.Linear(2 * dimensione, dimensione), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(dimensione, numero_classi))

        indice_nodo = torch.arange(NODI_PER_FRAME)
        self.register_buffer("tipo_di_ogni_nodo", torch.where(indice_nodo < NODI_PERSONE,
                                                              indice_nodo % NUMERO_KEYPOINT, NUMERO_KEYPOINT))
        self.register_buffer("posto_di_ogni_nodo", torch.where(indice_nodo < NODI_PERSONE,
                                                               indice_nodo // NUMERO_KEYPOINT, NUMERO_PERSONE))
        self.register_buffer("tipi_arco", matrice_tipi_arco())

    def forward(self, nodi: torch.Tensor, maschera: torch.Tensor, classe_arma: torch.Tensor) -> torch.Tensor:
        """nodi [B, T, 70, 5], maschera [B, T, 70] bool, classe_arma [B, T, 2] → logit [B, classi]."""
        lotto, frame, _, _ = nodi.shape
        x = self.proiezione_ingresso(nodi) + self.tipo_nodo(self.tipo_di_ogni_nodo) \
            + self.posto_nodo(self.posto_di_ogni_nodo) + self.posizione_temporale[:frame, None, :]
        classe_per_nodo = torch.zeros(lotto, frame, NODI_PER_FRAME, dtype=torch.long, device=nodi.device)
        classe_per_nodo[:, :, NODI_PERSONE:] = classe_arma.long()
        x = x + self.classe_arma(classe_per_nodo) * (classe_per_nodo > 0)[..., None]

        posizioni = nodi[..., :2]
        distanze = torch.cdist(posizioni.reshape(lotto * frame, NODI_PER_FRAME, 2),
                               posizioni.reshape(lotto * frame, NODI_PER_FRAME, 2)).view(
            lotto, frame, NODI_PER_FRAME, NODI_PER_FRAME)
        for blocco in self.blocchi:
            x = blocco(x, maschera, self.tipi_arco, distanze)

        x = self.norma_finale(x)
        peso = maschera[..., None].float()
        media = (x * peso).sum((1, 2)) / peso.sum((1, 2)).clamp(min=1.0)
        massimo = x.masked_fill(~maschera[..., None], float("-inf")).amax((1, 2))
        massimo = torch.where(torch.isfinite(massimo), massimo, torch.zeros_like(massimo))  # scena vuota
        return self.testa(torch.cat([media, massimo], dim=-1))


# ==============================================================================================================
# modelli/stgcn
# ST-GCN (Yan et al., AAAI 2018) in PyTorch puro, come termine di confronto per il Graph-Transformer (D11).
#
# Grafo dello scheletro COCO per ogni persona, partizione "spaziale" (radice, verso il centro, verso l'esterno),
# 10 blocchi con convoluzione temporale 9, importanza degli archi appresa. Le 4 persone sono elaborate separatamente
# e poi unite con max-pool (le persone assenti sono escluse); le armi passano da un ramo a parte e vengono aggiunte
# prima del classificatore. Iperparametri del paper originale; PROVVISORI per questo progetto.
# ==============================================================================================================

def distanze_in_salti(numero_nodi: int, archi) -> np.ndarray:
    adiacenza = np.eye(numero_nodi, dtype=bool)
    for i, j in archi:
        adiacenza[i, j] = adiacenza[j, i] = True
    distanze = np.full((numero_nodi, numero_nodi), np.inf)
    raggiunti = np.eye(numero_nodi, dtype=bool)
    distanze[raggiunti] = 0
    for salti in range(1, numero_nodi):
        raggiunti = (raggiunti.astype(int) @ adiacenza.astype(int)) > 0
        distanze[raggiunti & np.isinf(distanze)] = salti
    return distanze


def adiacenza_partizione_spaziale() -> np.ndarray:
    """[3, 17, 17]: sottoinsiemi radice / centripeto / centrifugo, normalizzati per grado."""
    salti = distanze_in_salti(NUMERO_KEYPOINT, OSSA)
    vicini = salti <= 1
    grado = vicini.sum(0)
    normalizzata = vicini / grado[None, :]  # colonna j normalizzata per il grado di j
    distanza_dal_centro = salti[:, CENTRO_SCHELETRO]
    partizioni = np.zeros((3, NUMERO_KEYPOINT, NUMERO_KEYPOINT), np.float32)
    for i in range(NUMERO_KEYPOINT):
        for j in range(NUMERO_KEYPOINT):
            if not vicini[i, j]:
                continue
            if distanza_dal_centro[j] == distanza_dal_centro[i]:
                partizioni[0, j, i] = normalizzata[j, i]
            elif distanza_dal_centro[j] > distanza_dal_centro[i]:
                partizioni[1, j, i] = normalizzata[j, i]
            else:
                partizioni[2, j, i] = normalizzata[j, i]
    return partizioni


class BloccoStGcn(nn.Module):
    def __init__(self, canali_ingresso: int, canali_uscita: int, numero_partizioni: int, passo: int = 1,
                 dropout: float = 0.0, residuo: bool = True, kernel_temporale: int = 9):
        super().__init__()
        self.numero_partizioni = numero_partizioni
        self.convoluzione_grafo = nn.Conv2d(canali_ingresso, canali_uscita * numero_partizioni, kernel_size=1)
        self.convoluzione_tempo = nn.Sequential(
            nn.BatchNorm2d(canali_uscita), nn.ReLU(inplace=True),
            nn.Conv2d(canali_uscita, canali_uscita, (kernel_temporale, 1), (passo, 1),
                      ((kernel_temporale - 1) // 2, 0)),
            nn.BatchNorm2d(canali_uscita), nn.Dropout(dropout, inplace=True))
        # il residuo è ciò che si somma all'uscita del blocco: niente, l'ingresso così com'è, oppure l'ingresso
        # passato da una convoluzione 1×1 quando cambiano i canali o il passo
        if not residuo:
            self.tipo_residuo = "nessuno"
        elif canali_ingresso == canali_uscita and passo == 1:
            self.tipo_residuo = "identita"
        else:
            self.tipo_residuo = "convoluzione"
            self.residuo = nn.Sequential(nn.Conv2d(canali_ingresso, canali_uscita, 1, (passo, 1)),
                                         nn.BatchNorm2d(canali_uscita))
        self.attivazione = nn.ReLU(inplace=True)

    def forward(self, x, adiacenza):
        """x [N, C, T, V], adiacenza [K, V, V]."""
        if self.tipo_residuo == "nessuno":
            residuo = 0
        elif self.tipo_residuo == "identita":
            residuo = x
        else:
            residuo = self.residuo(x)
        y = self.convoluzione_grafo(x)
        lotto, _, frame, nodi = y.shape
        y = y.view(lotto, self.numero_partizioni, -1, frame, nodi)
        y = torch.einsum("nkctv,kvw->nctw", y, adiacenza)
        return self.attivazione(self.convoluzione_tempo(y) + residuo)


class StGcn(nn.Module):
    def __init__(self, dropout: float = 0.5, numero_classi: int = len(CLASSI), numero_classi_arma: int = 3,
                 dimensione_ramo_armi: int = 64):
        super().__init__()
        self.register_buffer("adiacenza", torch.from_numpy(adiacenza_partizione_spaziale()))
        partizioni = self.adiacenza.shape[0]
        self.normalizzazione_dati = nn.BatchNorm1d(CARATTERISTICHE_PER_NODO * NUMERO_KEYPOINT)
        configurazione = [(CARATTERISTICHE_PER_NODO, 64, 1, False), (64, 64, 1, True), (64, 64, 1, True),
                          (64, 64, 1, True), (64, 128, 2, True), (128, 128, 1, True), (128, 128, 1, True),
                          (128, 256, 2, True), (256, 256, 1, True), (256, 256, 1, True)]
        lista_blocchi = []
        lista_importanze = []
        for numero, (ingresso, uscita, passo, residuo) in enumerate(configurazione):
            if numero == 0:
                dropout_blocco = 0.0   # niente dropout sul primo blocco
            else:
                dropout_blocco = dropout
            lista_blocchi.append(BloccoStGcn(ingresso, uscita, partizioni, passo, dropout_blocco, residuo))
            lista_importanze.append(nn.Parameter(torch.ones_like(self.adiacenza)))
        self.blocchi = nn.ModuleList(lista_blocchi)
        self.importanza_archi = nn.ParameterList(lista_importanze)

        self.classe_arma = nn.Embedding(numero_classi_arma + 1, 16)
        self.ramo_armi = nn.Sequential(nn.Linear(CARATTERISTICHE_PER_NODO + 16, dimensione_ramo_armi), nn.ReLU(),
                                       nn.Linear(dimensione_ramo_armi, dimensione_ramo_armi))
        self.classificatore = nn.Linear(256 + 2 * dimensione_ramo_armi, numero_classi)

    def forward(self, nodi: torch.Tensor, maschera: torch.Tensor, classe_arma: torch.Tensor) -> torch.Tensor:
        """Stessi ingressi del Graph-Transformer: nodi [B, T, 70, 5], maschera [B, T, 70], classe_arma [B, T, 2]."""
        lotto, frame, _, canali = nodi.shape
        persone = nodi[:, :, :NODI_PERSONE].reshape(lotto, frame, NUMERO_PERSONE, NUMERO_KEYPOINT, canali)
        x = persone.permute(0, 2, 3, 4, 1).reshape(lotto * NUMERO_PERSONE, NUMERO_KEYPOINT * canali, frame)
        x = self.normalizzazione_dati(x)
        x = x.view(lotto * NUMERO_PERSONE, NUMERO_KEYPOINT, canali, frame).permute(0, 2, 3, 1)  # [N*M, C, T, V]
        for blocco, importanza in zip(self.blocchi, self.importanza_archi):
            x = blocco(x, self.adiacenza * importanza)
        per_persona = x.mean((2, 3)).view(lotto, NUMERO_PERSONE, -1)
        persona_presente = maschera[:, :, :NODI_PERSONE].reshape(lotto, frame, NUMERO_PERSONE, -1).any(-1).any(1)
        per_persona = per_persona.masked_fill(~persona_presente[..., None], float("-inf")).amax(1)
        per_persona = torch.where(torch.isfinite(per_persona), per_persona, torch.zeros_like(per_persona))

        armi = torch.cat([nodi[:, :, NODI_PERSONE:], self.classe_arma(classe_arma.long())], dim=-1)
        armi = self.ramo_armi(armi) * maschera[:, :, NODI_PERSONE:, None]                  # [B, T, 2, D]
        armi_presenti = maschera[:, :, NODI_PERSONE:].float().sum((1, 2)).clamp(min=1.0)[:, None]
        sintesi_armi = torch.cat([armi.sum((1, 2)) / armi_presenti, armi.amax((1, 2))], dim=-1)
        return self.classificatore(torch.cat([per_persona, sintesi_armi], dim=-1))


# ==============================================================================================================
# modelli/__init__
# Classificatori delle finestre: stesso ingresso (nodi, maschera, classe_arma), stessa uscita (logit 4 classi).
# ==============================================================================================================

MODELLI = {"graph_transformer": GraphTransformer, "stgcn": StGcn}


def crea_modello(nome: str, **iperparametri):
    return MODELLI[nome](**iperparametri)


def conta_parametri(modello) -> int:
    totale = 0
    for parametro in modello.parameters():
        if parametro.requires_grad:
            totale += parametro.numel()
    return totale


# ==============================================================================================================
# valutazione
# Metriche (macro-F1, recall per classe, matrice di confusione) e latenza su CPU.
# ==============================================================================================================

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
    if presenti.any():
        macro_f1 = float(f1[presenti].mean())
    else:
        macro_f1 = 0.0
    return {
        "accuratezza": float(veri_positivi.sum() / max(matrice.sum(), 1)),
        "macro_f1": macro_f1,
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


# ==============================================================================================================
# addestramento
# Addestramento di un classificatore di finestre (Graph-Transformer o ST-GCN) con split per video.
#
# Uso:
#     python -m sicurezza.modelli addestra --finestre finestre.npz --modello graph_transformer --uscita runs/gt
#
# Tutti gli iperparametri predefiniti sono PROVVISORI (DA DECIDERE con Marco dopo i primi esperimenti).
# ==============================================================================================================

def carica_finestre(percorso) -> dict:
    finestre = {}
    with np.load(percorso, allow_pickle=False) as dati:
        for chiave in dati.files:
            finestre[chiave] = dati[chiave]
    return finestre


def lotti(indici: np.ndarray, dimensione_lotto: int, mescola: bool, generatore) -> list:
    """Divide gli indici in pezzi da `dimensione_lotto` (l'ultimo può essere più corto)."""
    if mescola:
        ordine = generatore.permutation(indici)
    else:
        ordine = indici
    elenco_lotti = []
    for partenza in range(0, len(ordine), dimensione_lotto):
        elenco_lotti.append(ordine[partenza:partenza + dimensione_lotto])
    return elenco_lotti


def tensori_lotto(finestre, indici, dispositivo):
    nodi = torch.from_numpy(finestre["nodi"][indici].astype(np.float32)).to(dispositivo)
    maschera = torch.from_numpy(finestre["maschera"][indici]).to(dispositivo)
    classe_arma = torch.from_numpy(finestre["classe_arma"][indici].astype(np.int64)).to(dispositivo)
    etichetta = torch.from_numpy(finestre["etichetta"][indici]).to(dispositivo)
    return nodi, maschera, classe_arma, etichetta


@torch.no_grad()
def predici(modello, finestre, indici, dispositivo, dimensione_lotto=256):
    """Ritorna (probabilità [finestre, classi], etichette vere)."""
    modello.eval()
    probabilita = []
    for indici_lotto in lotti(indici, dimensione_lotto, False, None):
        nodi, maschera, classe_arma, _ = tensori_lotto(finestre, indici_lotto, dispositivo)
        with torch.autocast(dispositivo.type, enabled=dispositivo.type == "cuda"):
            logit = modello(nodi, maschera, classe_arma).float()
        probabilita.append(torch.softmax(logit, -1).cpu().numpy())
    if len(probabilita) > 0:
        probabilita = np.concatenate(probabilita)
    else:
        probabilita = np.empty((0, len(CLASSI)))
    return probabilita, finestre["etichetta"][indici]


def indici_degli_split(finestre: dict, seme: int = 0) -> dict:
    """Split P4 per video (D33) + D26. Deterministico: rifatto sulle stesse finestre dà gli stessi indici.
    D68: i video sintetici di WVD vanno sempre in train (val e test restano solo video veri)."""
    sempre_in_train = set(gruppi_visti_dal_detector())
    for gruppo, sorgente in zip(finestre["gruppo"].tolist(), finestre["sorgente"].tolist()):
        if sorgente == "wvd":
            sempre_in_train.add(gruppo)
    split = dividi_per_gruppi(finestre["gruppo"], finestre["etichetta"], finestre["sorgente"], seme=seme,
                              gruppi_sempre_in_train=sempre_in_train)
    indici_split = {}
    for numero, nome in enumerate(NOMI_SPLIT):   # 0 train, 1 val, 2 test
        indici_split[nome] = np.flatnonzero(split == numero)
    return indici_split


def salva_split(finestre: dict, percorso_uscita: Path, seme: int = 0) -> dict:
    """split.npz: indici delle finestre per split + i video di ogni split (servono a scegliere i video della demo tra
    quelli MAI visti in addestramento, e a tarare il WARNING ARMA sul validation senza addestrare niente)."""
    indici_split = indici_degli_split(finestre, seme)
    da_salvare = {}
    for nome, indici in indici_split.items():
        da_salvare[nome] = indici
    for nome, indici in indici_split.items():
        da_salvare[f"clip_{nome}"] = np.unique(finestre["id_clip"][indici])
    np.savez(percorso_uscita, **da_salvare)
    return indici_split


def addestra(finestre: dict, nome_modello: str, cartella_uscita: Path, epoche: int = 50, dimensione_lotto: int = 64,
             tasso_apprendimento: float = 5e-4, decadimento_pesi: float = 0.05, pazienza: int = 10,
             bilanciamento: str = "pesi", seme: int = 0, iperparametri_modello=None,
             ore_massime=None) -> dict:
    """ore_massime: tetto di tempo (quota GPU di Kaggle). Si controlla a fine epoca: poi si valuta il migliore."""
    torch.manual_seed(seme)
    inizio_addestramento = time.perf_counter()
    generatore = np.random.default_rng(seme)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cartella_uscita.mkdir(parents=True, exist_ok=True)

    indici_split = salva_split(finestre, cartella_uscita / "split.npz", seme)

    if iperparametri_modello is None:
        iperparametri_modello = {}
    modello = crea_modello(nome_modello, **iperparametri_modello).to(dispositivo)
    conteggio_train = np.bincount(finestre["etichetta"][indici_split["train"]], minlength=len(CLASSI))
    if bilanciamento == "pesi":  # pesi inversamente proporzionali alla frequenza (classi assenti: peso 0)
        pesi_classi = conteggio_train.sum() / np.maximum(conteggio_train, 1) / len(CLASSI)
        pesi = np.where(conteggio_train > 0, pesi_classi, 0)
        funzione_perdita = nn.CrossEntropyLoss(weight=torch.tensor(pesi, dtype=torch.float32, device=dispositivo))
    else:
        funzione_perdita = nn.CrossEntropyLoss()
    ottimizzatore = torch.optim.AdamW(modello.parameters(), lr=tasso_apprendimento, weight_decay=decadimento_pesi)
    passi_totali = epoche * math.ceil(len(indici_split["train"]) / dimensione_lotto)
    passi_riscaldamento = max(1, passi_totali // 20)

    def fattore_tasso_apprendimento(passo):
        """Il tasso sale piano nel primo 5% dei passi (riscaldamento), poi scende a coseno fino a 0."""
        riscaldamento = (passo + 1) / passi_riscaldamento
        coseno = 0.5 * (1 + math.cos(math.pi * min(passo / passi_totali, 1.0)))
        return min(riscaldamento, coseno)

    pianificatore = torch.optim.lr_scheduler.LambdaLR(ottimizzatore, fattore_tasso_apprendimento)
    scalatore = torch.amp.GradScaler(enabled=dispositivo.type == "cuda")

    numeri_finestre = []
    for nome, indici in indici_split.items():
        numeri_finestre.append(f"{nome}={len(indici)}")
    print(f"{nome_modello}: {conta_parametri(modello):,} parametri, dispositivo {dispositivo}, finestre "
          + ", ".join(numeri_finestre), flush=True)
    storia = []
    miglior_f1 = -1.0
    epoche_senza_miglioramento = 0
    for epoca in range(1, epoche + 1):
        modello.train()
        inizio = time.perf_counter()
        perdita_totale = 0.0
        indici_train = indici_split["train"]
        if bilanciamento == "campionamento":  # ogni classe estratta con la stessa probabilità
            peso_finestra = 1.0 / conteggio_train[finestre["etichetta"][indici_train]]
            indici_train = generatore.choice(indici_train, len(indici_train), p=peso_finestra / peso_finestra.sum())
        for indici_lotto in lotti(indici_train, dimensione_lotto, True, generatore):
            nodi, maschera, classe_arma, etichetta = tensori_lotto(finestre, indici_lotto, dispositivo)
            nodi, maschera, classe_arma = aumenta_lotto(nodi, maschera, classe_arma)
            with torch.autocast(dispositivo.type, enabled=dispositivo.type == "cuda"):
                perdita = funzione_perdita(modello(nodi, maschera, classe_arma).float(), etichetta)
            ottimizzatore.zero_grad(set_to_none=True)
            scalatore.scale(perdita).backward()
            scalatore.unscale_(ottimizzatore)
            nn.utils.clip_grad_norm_(modello.parameters(), 1.0)
            scalatore.step(ottimizzatore)
            scalatore.update()
            pianificatore.step()
            perdita_totale += perdita.item() * len(indici_lotto)

        probabilita_val, vere_val = predici(modello, finestre, indici_split["val"], dispositivo)
        metriche_val = metriche(vere_val, probabilita_val.argmax(1))
        storia.append({"epoca": epoca, "perdita_train": perdita_totale / max(len(indici_train), 1),
                       "macro_f1_val": metriche_val["macro_f1"], "secondi": time.perf_counter() - inizio})
        print(f"epoca {epoca}: perdita {storia[-1]['perdita_train']:.4f}, macro-F1 val {metriche_val['macro_f1']:.4f}"
              f" ({storia[-1]['secondi']:.0f} s)", flush=True)
        if metriche_val["macro_f1"] > miglior_f1:
            miglior_f1 = metriche_val["macro_f1"]
            epoche_senza_miglioramento = 0
            torch.save({"nome_modello": nome_modello, "iperparametri_modello": iperparametri_modello,
                        "classi": CLASSI, "stato": modello.state_dict(), "epoca": epoca}, cartella_uscita / "migliore.pt")
        else:
            epoche_senza_miglioramento += 1
            if epoche_senza_miglioramento >= pazienza:
                print(f"stop anticipato: {pazienza} epoche senza miglioramento", flush=True)
                break
        if ore_massime and time.perf_counter() - inizio_addestramento > ore_massime * 3600:
            print(f"stop: superate {ore_massime} h (tetto di tempo)", flush=True)
            break

    salvato = torch.load(cartella_uscita / "migliore.pt", map_location=dispositivo)
    modello.load_state_dict(salvato["stato"])
    impostazioni = {"epoche": epoche, "dimensione_lotto": dimensione_lotto,
                    "tasso_apprendimento": tasso_apprendimento, "decadimento_pesi": decadimento_pesi,
                    "bilanciamento": bilanciamento, "seme": seme}
    impostazioni.update(iperparametri_modello)
    risultati = {"modello": nome_modello, "parametri": conta_parametri(modello), "epoca_migliore": salvato["epoca"],
                 "storia": storia, "impostazioni": impostazioni}
    for nome in ("val", "test"):
        probabilita, vere = predici(modello, finestre, indici_split[nome], dispositivo)
        if len(vere) > 0:
            risultati[nome] = metriche(vere, probabilita.argmax(1))
        else:
            risultati[nome] = None
        np.save(cartella_uscita / f"probabilita_{nome}.npy", probabilita)
        # per analizzare gli errori per sorgente e per video senza il file delle finestre (resta su Kaggle)
        indici = indici_split[nome]
        np.savez(cartella_uscita / f"finestre_{nome}.npz", etichetta=finestre["etichetta"][indici],
                 sorgente=finestre["sorgente"][indici], id_clip=finestre["id_clip"][indici],
                 inizio_finestra=finestre["inizio_finestra"][indici])
    risultati["latenza_cpu_ms_per_finestra"] = latenza_cpu_millisecondi(modello)
    (cartella_uscita / "risultati.json").write_text(json.dumps(risultati, indent=2, ensure_ascii=False))
    return risultati


def comando_addestra(argomenti=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finestre", required=True)
    parser.add_argument("--modello", choices=["graph_transformer", "stgcn"], required=True)
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--epoche", type=int, default=50)
    parser.add_argument("--lotto", type=int, default=64)
    parser.add_argument("--tasso-apprendimento", type=float, default=5e-4)
    parser.add_argument("--bilanciamento", choices=["pesi", "campionamento", "nessuno"], default="pesi")
    parser.add_argument("--seme", type=int, default=0)
    parser.add_argument("--pazienza", type=int, default=10, help="epoche senza miglioramento prima di fermarsi")
    parser.add_argument("--decadimento-pesi", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=None, help="se omesso, quello predefinito del modello")
    parser.add_argument("--ore-massime", type=float, default=None, help="tetto di tempo (quota GPU)")
    argomenti = parser.parse_args(argomenti)
    iperparametri_modello = {}
    if argomenti.dropout is not None:
        iperparametri_modello["dropout"] = argomenti.dropout
    risultati = addestra(carica_finestre(argomenti.finestre), argomenti.modello, Path(argomenti.uscita),
                         argomenti.epoche, argomenti.lotto, argomenti.tasso_apprendimento,
                         decadimento_pesi=argomenti.decadimento_pesi, pazienza=argomenti.pazienza,
                         bilanciamento=argomenti.bilanciamento, seme=argomenti.seme,
                         iperparametri_modello=iperparametri_modello, ore_massime=argomenti.ore_massime)
    da_mostrare = {"val": risultati["val"], "test": risultati["test"],
                   "latenza_cpu_ms_per_finestra": risultati["latenza_cpu_ms_per_finestra"]}
    print(json.dumps(da_mostrare, indent=2, ensure_ascii=False))

# ==============================================================================================================
# analisi_predizioni
# Dove sbagliano i modelli: metriche per sorgente e classe prevalente per video, dalle probabilità salvate da
# `addestra` (probabilita_test.npy) e dalle stesse finestre (lo split si ricalcola identico). Serve anche a
# scegliere i video della demo tra quelli del test (mai visti in addestramento).
#
# Uso:
#     python -m sicurezza.modelli analizza-predizioni --finestre finestre.npz \
#         --probabilita stgcn=runs/.../probabilita_test.npy graph_transformer=... --uscita analisi_test.json
# ==============================================================================================================

def conta_classi(indici_classe) -> dict:
    """Quante finestre per classe, con i nomi delle classi: {"normale": 12, "violenza": 3}."""
    conteggio = {}
    for indice in indici_classe:
        nome = CLASSI[indice]
        conteggio[nome] = conteggio.get(nome, 0) + 1
    return conteggio


def analizza_predizioni(finestre: dict, probabilita_per_modello: dict, nome_split: str = "test",
                        seme: int = 0) -> dict:
    indici = indici_degli_split(finestre, seme)[nome_split]
    vere = finestre["etichetta"][indici]
    sorgenti = finestre["sorgente"][indici]
    id_clip = finestre["id_clip"][indici]
    analisi = {"split": nome_split, "finestre": len(indici), "modelli": {}}
    for nome_modello, probabilita in probabilita_per_modello.items():
        if len(probabilita) != len(indici):
            raise ValueError(f"{nome_modello}: {len(probabilita)} probabilità per {len(indici)} finestre di {nome_split}"
                             " (finestre diverse da quelle dell'addestramento?)")
        predette = probabilita.argmax(1)
        per_sorgente = {}
        for sorgente in sorted(set(sorgenti.tolist())):
            della_sorgente = sorgenti == sorgente
            misure = metriche(vere[della_sorgente], predette[della_sorgente])
            per_sorgente[sorgente] = {"finestre": int(della_sorgente.sum()), "macro_f1": round(misure["macro_f1"], 4),
                                      "accuratezza": round(misure["accuratezza"], 4), "recall": misure["recall"],
                                      "matrice_confusione": misure["matrice_confusione"]}
        per_clip = []
        for clip in sorted(set(id_clip.tolist())):
            della_clip = id_clip == clip
            per_clip.append({"id_clip": clip, "finestre": int(della_clip.sum()),
                             "classi_vere": conta_classi(vere[della_clip].tolist()),
                             "classi_predette": conta_classi(predette[della_clip].tolist()),
                             "corrette": round(float((vere[della_clip] == predette[della_clip]).mean()), 3)})
        analisi["modelli"][nome_modello] = {"per_sorgente": per_sorgente, "per_clip": per_clip}
    return analisi


def comando_analizza_predizioni(argomenti=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finestre", required=True)
    parser.add_argument("--probabilita", nargs="+", required=True, help="nome_modello=probabilita_test.npy")
    parser.add_argument("--split", default="test", choices=NOMI_SPLIT)
    parser.add_argument("--uscita", required=True)
    argomenti = parser.parse_args(argomenti)
    probabilita = {}
    for voce in argomenti.probabilita:   # "nome_modello=percorso.npy"
        nome_modello, percorso = voce.split("=", 1)
        probabilita[nome_modello] = np.load(percorso)
    analisi = analizza_predizioni(carica_finestre(argomenti.finestre), probabilita, argomenti.split)
    Path(argomenti.uscita).write_text(json.dumps(analisi, indent=1, ensure_ascii=False))
    for nome_modello, dati_modello in analisi["modelli"].items():
        for sorgente, misure in dati_modello["per_sorgente"].items():
            print(f"{nome_modello:18s} {sorgente:16s} finestre {misure['finestre']:6d}  macro-F1 {misure['macro_f1']:.3f}  "
                  f"accuratezza {misure['accuratezza']:.3f}")

# ==============================================================================================================
# riga di comando: python -m sicurezza.modelli <comando> [opzioni]
# ==============================================================================================================
def comando_salva_split(argomenti=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finestre", required=True)
    parser.add_argument("--uscita", required=True, help="split.npz (lo stesso che scrive `addestra`)")
    parser.add_argument("--seme", type=int, default=0)
    argomenti = parser.parse_args(argomenti)
    indici_split = salva_split(carica_finestre(argomenti.finestre), Path(argomenti.uscita), argomenti.seme)
    numeri = {}
    for nome, indici in indici_split.items():
        numeri[nome] = len(indici)
    print(numeri)


def comando_predici(argomenti=None) -> None:
    """Probabilità di un classificatore salvato (migliore.pt) sulle finestre di uno split: serve a confrontare sulle
    STESSE finestre modelli addestrati su dati diversi (es. l'addestramento 1 sui dati dell'addestramento 2)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--finestre", required=True)
    parser.add_argument("--checkpoint", required=True, help="migliore.pt di `addestra`")
    parser.add_argument("--split", required=True, help="split.npz con gli indici delle finestre")
    parser.add_argument("--nome-split", default="test")
    parser.add_argument("--uscita", required=True, help="file .npy [finestre, classi]")
    argomenti = parser.parse_args(argomenti)
    salvato = torch.load(argomenti.checkpoint, map_location="cpu")
    modello = crea_modello(salvato["nome_modello"], **salvato["iperparametri_modello"])
    modello.load_state_dict(salvato["stato"])
    indici = np.load(argomenti.split)[argomenti.nome_split]
    probabilita, _ = predici(modello, carica_finestre(argomenti.finestre), indici, torch.device("cpu"))
    np.save(argomenti.uscita, probabilita)
    print(f"{salvato['nome_modello']} (epoca {salvato.get('epoca')}): {len(indici)} finestre → {argomenti.uscita}")


COMANDI = {"dataset-finestre": comando_dataset_finestre, "addestra": comando_addestra,
           "analizza-predizioni": comando_analizza_predizioni, "salva-split": comando_salva_split,
           "predici": comando_predici}


def main(argomenti=None) -> None:
    argomenti = sys.argv[1:] if argomenti is None else argomenti
    if not argomenti or argomenti[0] not in COMANDI:
        print("uso: python -m sicurezza.modelli {dataset-finestre|addestra} [opzioni]  (--help dopo il comando)")
        raise SystemExit(2)
    COMANDI[argomenti[0]](argomenti[1:])


if __name__ == "__main__":
    main()
