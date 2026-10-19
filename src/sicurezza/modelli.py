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

from .dati import gruppi_visti_dal_detector, CENTRO_SCHELETRO, CLASSI, ClipAnnotata, FRAME_PER_FINESTRA, NOMI_SPLIT, NUMERO_KEYPOINT, OSSA, POLSI, SPECCHIO_KEYPOINT, dividi_per_gruppi, etichetta_finestra, inizi_finestre, leggi_annotazioni, nome_file_estratto


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
    keypoint_per_traccia: dict[int, np.ndarray]   # id traccia → (numero_frame, 17, 3), NaN dove la persona manca
    box_per_traccia: dict[int, np.ndarray]        # id traccia → (numero_frame, 4), NaN dove la persona manca
    armi_frame: np.ndarray
    armi_box: np.ndarray
    armi_classe: np.ndarray
    armi_confidenza: np.ndarray


def carica_timeline(percorso_npz, ricuci: bool = True) -> TimelineVideo:
    return timeline_da_dati(np.load(percorso_npz), ricuci)


def armi_non_piu_grandi_della_persona(dati) -> np.ndarray:
    """D38: un'arma vera non è più grande di chi la impugna. Per ogni arma rilevata si prende la persona più vicina
    (centro dei box) nello stesso frame: se il riquadro dell'arma ha area maggiore, l'arma si scarta (riquadri su
    mezza scena: reti, barche, porte). Senza persone nel frame l'arma resta: non c'è niente con cui confrontarla."""
    area = lambda box: (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])
    centro = lambda box: np.stack([(box[..., 0] + box[..., 2]) / 2, (box[..., 1] + box[..., 3]) / 2], -1)
    armi_frame, armi_box = np.asarray(dati["armi_frame"]), np.asarray(dati["armi_box"])  # un .npz rilegge a ogni accesso
    ordine = np.argsort(dati["persone_frame"], kind="stable")
    persone_frame, persone_box = np.asarray(dati["persone_frame"])[ordine], np.asarray(dati["persone_box"])[ordine]
    inizi, fini = np.searchsorted(persone_frame, armi_frame, "left"), np.searchsorted(persone_frame, armi_frame, "right")
    da_tenere = np.ones(len(armi_frame), bool)
    for indice in np.flatnonzero(fini > inizi):
        box_persone = persone_box[inizi[indice]:fini[indice]]
        distanze = np.linalg.norm(centro(box_persone) - centro(armi_box[indice]), axis=-1)
        da_tenere[indice] = area(armi_box[indice]) <= area(box_persone[np.argmin(distanze)])
    return da_tenere


def timeline_da_dati(dati, ricuci: bool = True, filtra_armi_grandi: bool = True) -> TimelineVideo:
    """Come carica_timeline, ma da un dizionario già in memoria (es. il buffer della demo)."""
    numero_frame = int(dati["numero_frame"])
    keypoint_per_traccia, box_per_traccia = {}, {}
    for id_traccia in np.unique(dati["persone_id_traccia"]):
        righe = dati["persone_id_traccia"] == id_traccia
        frame_traccia = dati["persone_frame"][righe]
        keypoint = np.full((numero_frame, NUMERO_KEYPOINT, 3), np.nan, np.float32)
        box = np.full((numero_frame, 4), np.nan, np.float32)
        keypoint[frame_traccia] = dati["persone_keypoint"][righe]
        box[frame_traccia] = dati["persone_box"][righe]
        keypoint_per_traccia[int(id_traccia)] = keypoint
        box_per_traccia[int(id_traccia)] = box
    if ricuci:
        keypoint_per_traccia, box_per_traccia = ricuci_tracce(keypoint_per_traccia, box_per_traccia)
    armi = armi_non_piu_grandi_della_persona(dati) if filtra_armi_grandi else slice(None)
    return TimelineVideo(numero_frame, int(dati["larghezza"]), int(dati["altezza"]), keypoint_per_traccia,
                         box_per_traccia, np.asarray(dati["armi_frame"])[armi], np.asarray(dati["armi_box"])[armi],
                         np.asarray(dati["armi_classe"])[armi], np.asarray(dati["armi_confidenza"])[armi].astype(np.float32))


BUCO_MASSIMO_FRAME = 15            # PROVVISORIO: 0,6 s senza la persona al massimo
DISTANZA_MASSIMA_ALTEZZE = 0.5     # PROVVISORIO: il nuovo frammento deve partire entro mezza altezza-corpo
RAPPORTO_ALTEZZE_MASSIMO = 1.5     # PROVVISORIO: stessa persona = altezza del box simile


def ricuci_tracce(keypoint_per_traccia: dict, box_per_traccia: dict, buco_massimo: int = BUCO_MASSIMO_FRAME,
                  distanza_massima: float = DISTANZA_MASSIMA_ALTEZZE,
                  rapporto_altezze_massimo: float = RAPPORTO_ALTEZZE_MASSIMO):
    """D17: unisce i frammenti di traccia della stessa persona.

    Una traccia B continua una traccia A se B inizia dopo la fine di A (mai sovrapposte nel tempo), entro
    `buco_massimo` frame, vicino all'ultimo box di A e con altezza simile. Tra più candidate si sceglie la più vicina.
    """
    def estremi(box):
        presenti = np.flatnonzero(~np.isnan(box[:, 0]))
        return presenti[0], presenti[-1]

    def centro_e_altezza(riga_box):
        return np.array([(riga_box[0] + riga_box[2]) / 2, (riga_box[1] + riga_box[3]) / 2]), riga_box[3] - riga_box[1]

    keypoint_per_traccia = dict(keypoint_per_traccia)
    box_per_traccia = {id_traccia: box.copy() for id_traccia, box in box_per_traccia.items()}
    ordine_per_inizio = sorted(box_per_traccia, key=lambda id_traccia: estremi(box_per_traccia[id_traccia])[0])
    for id_nuova in ordine_per_inizio:
        if id_nuova not in box_per_traccia:
            continue
        inizio_nuova, _ = estremi(box_per_traccia[id_nuova])
        centro_nuova, altezza_nuova = centro_e_altezza(box_per_traccia[id_nuova][inizio_nuova])
        migliore, distanza_migliore = None, np.inf
        for id_vecchia, box_vecchia in box_per_traccia.items():
            if id_vecchia == id_nuova:
                continue
            _, fine_vecchia = estremi(box_vecchia)
            if not (0 < inizio_nuova - fine_vecchia <= buco_massimo):
                continue
            centro_vecchia, altezza_vecchia = centro_e_altezza(box_vecchia[fine_vecchia])
            if max(altezza_vecchia, altezza_nuova) > rapporto_altezze_massimo * max(min(altezza_vecchia, altezza_nuova), 1):
                continue
            distanza = np.linalg.norm(centro_nuova - centro_vecchia) / max(altezza_vecchia, 1)
            if distanza <= distanza_massima and distanza < distanza_migliore:
                migliore, distanza_migliore = id_vecchia, distanza
        if migliore is not None:  # la traccia nuova diventa il seguito di quella vecchia
            presenti = ~np.isnan(box_per_traccia[id_nuova][:, 0])
            box_per_traccia[migliore][presenti] = box_per_traccia[id_nuova][presenti]
            keypoint_unito = keypoint_per_traccia[migliore].copy()
            keypoint_unito[presenti] = keypoint_per_traccia[id_nuova][presenti]
            keypoint_per_traccia[migliore] = keypoint_unito
            del box_per_traccia[id_nuova], keypoint_per_traccia[id_nuova]
    return keypoint_per_traccia, box_per_traccia


def punteggio_presenza_e_movimento(keypoint: np.ndarray, box: np.ndarray) -> float:
    """Criterio PROVVISORIO per scegliere le 4 persone (DA DECIDERE nel diario):
    frame in cui la persona c'è × (1 + movimento medio dei keypoint in altezze-corpo per frame)."""
    presente = ~np.isnan(box[:, 0])
    if presente.sum() < 2:
        return float(presente.sum())
    altezza_corpo = np.nanmedian(box[:, 3] - box[:, 1]) + 1e-6
    spostamenti = np.linalg.norm(np.diff(keypoint[:, :, :2], axis=0), axis=-1) / altezza_corpo
    movimento_medio = float(np.nanmean(spostamenti)) if np.isfinite(spostamenti).any() else 0.0
    return float(presente.sum()) * (1.0 + movimento_medio)


CRITERI_SCELTA_PERSONE = {"presenza_e_movimento": punteggio_presenza_e_movimento}


def costruisci_finestra(timeline: TimelineVideo, inizio_finestra: int,
                        frame_per_finestra: int = FRAME_PER_FINESTRA,
                        criterio_scelta_persone: str = "presenza_e_movimento"):
    """Ritorna (nodi [T, 70, 5] float32, maschera [T, 70] bool, classe_arma [T, 2] int8 con 0 = nessuna arma)."""
    fine_finestra = inizio_finestra + frame_per_finestra
    punteggio = CRITERI_SCELTA_PERSONE[criterio_scelta_persone]

    tracce_in_finestra = []
    for id_traccia, box in timeline.box_per_traccia.items():
        box_finestra = box[inizio_finestra:fine_finestra]
        if not np.isnan(box_finestra[:, 0]).all():
            keypoint_finestra = timeline.keypoint_per_traccia[id_traccia][inizio_finestra:fine_finestra]
            tracce_in_finestra.append((punteggio(keypoint_finestra, box_finestra), keypoint_finestra, box_finestra))
    tracce_scelte = sorted(tracce_in_finestra, key=lambda traccia: -traccia[0])[:NUMERO_PERSONE]

    # scala e centro della scena: altezza mediana delle persone scelte, centro medio dei loro box
    if tracce_scelte:
        tutti_i_box = np.concatenate([box for _, _, box in tracce_scelte])
        altezza_riferimento = float(np.nanmedian(tutti_i_box[:, 3] - tutti_i_box[:, 1]))
        centro_scena = np.array([np.nanmean((tutti_i_box[:, 0] + tutti_i_box[:, 2]) / 2),
                                 np.nanmean((tutti_i_box[:, 1] + tutti_i_box[:, 3]) / 2)], np.float32)
    else:
        altezza_riferimento, centro_scena = 0.0, np.array([timeline.larghezza / 2, timeline.altezza / 2], np.float32)
    if not np.isfinite(altezza_riferimento) or altezza_riferimento < 1:
        altezza_riferimento = 0.3 * timeline.altezza

    def normalizza(coordinate):
        return (coordinate - centro_scena) / altezza_riferimento

    nodi = np.zeros((frame_per_finestra, NODI_PER_FRAME, CARATTERISTICHE_PER_NODO), np.float32)
    maschera = np.zeros((frame_per_finestra, NODI_PER_FRAME), bool)
    for posto_persona, (_, keypoint_finestra, _) in enumerate(tracce_scelte):
        nodi_persona = slice(posto_persona * NUMERO_KEYPOINT, (posto_persona + 1) * NUMERO_KEYPOINT)
        presente = ~np.isnan(keypoint_finestra[:, 0, 0])
        visibile = presente[:, None] & (np.nan_to_num(keypoint_finestra[:, :, 2]) >= SOGLIA_VISIBILITA_KEYPOINT)
        posizione = np.where(visibile[..., None], normalizza(np.nan_to_num(keypoint_finestra[:, :, :2])), 0.0)
        nodi[:, nodi_persona, 0:2] = posizione
        nodi[:, nodi_persona, 2] = np.where(visibile, np.nan_to_num(keypoint_finestra[:, :, 2]), 0.0)
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
        nodi[frame_relativo, nodo, 0:2] = normalizza(np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]))
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

def finestre_di_una_clip(clip: ClipAnnotata, percorso_npz: Path):
    timeline = carica_timeline(percorso_npz)
    eventi = clip.eventi_in_frame(timeline.numero_frame)
    for inizio in inizi_finestre(timeline.numero_frame):
        classe = etichetta_finestra(int(inizio), eventi, clip.indice_classe_sfondo())
        if classe is not None:
            yield int(inizio), classe, costruisci_finestra(timeline, int(inizio))


def costruisci_dataset(clip: list[ClipAnnotata], cartelle_estratti: list[Path]) -> dict[str, np.ndarray]:
    percorso_per_nome = {file.name: file for cartella in cartelle_estratti for file in Path(cartella).glob("*.npz")}
    nodi, maschere, classi_arma, etichette, gruppi, sorgenti, id_clip, inizi = [], [], [], [], [], [], [], []
    clip_mancanti = 0
    for singola_clip in clip:
        percorso_npz = percorso_per_nome.get(nome_file_estratto(singola_clip.id_clip))
        if percorso_npz is None:
            clip_mancanti += 1
            continue
        for inizio, classe, (nodi_finestra, maschera, classe_arma) in finestre_di_una_clip(singola_clip, percorso_npz):
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


def riepilogo(dataset: dict[str, np.ndarray]) -> str:
    righe = [f"{len(dataset['etichetta'])} finestre"]
    conteggio = Counter(zip(dataset["sorgente"].tolist(), dataset["etichetta"].tolist()))
    for (sorgente, classe), numero in sorted(conteggio.items()):
        righe.append(f"  {sorgente:16s} {CLASSI[classe]:18s} {numero}")
    return "\n".join(righe)


def comando_dataset_finestre(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotazioni", required=True)
    parser.add_argument("--estratti", nargs="+", required=True)
    parser.add_argument("--uscita", required=True)
    argomenti = parser.parse_args(argomenti)
    dataset = costruisci_dataset(leggi_annotazioni(argomenti.annotazioni), [Path(c) for c in argomenti.estratti])
    np.savez(argomenti.uscita, **dataset)
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
    scala = torch.empty(lotto, 1, 1, 1, device=dispositivo).uniform_(*INTERVALLO_SCALA)
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
ARCO_SE_STESSO, ARCO_OSSO, ARCO_STESSA_PERSONA, ARCO_PERSONE_DIVERSE, ARCO_ARMA_POLSO, ARCO_ARMA_CORPO, \
    ARCO_ARMA_ARMA = range(7)
NUMERO_TIPI_ARCO = 7
ARCHI_TRA_ENTITA_DIVERSE = (ARCO_PERSONE_DIVERSE, ARCO_ARMA_POLSO, ARCO_ARMA_CORPO, ARCO_ARMA_ARMA)


def matrice_tipi_arco() -> torch.Tensor:
    """[70, 70] con il tipo di arco tra ogni coppia di nodi."""
    persona_del_nodo = [nodo // NUMERO_KEYPOINT for nodo in range(NODI_PERSONE)] + [-1] * NUMERO_ARMI
    keypoint_del_nodo = [nodo % NUMERO_KEYPOINT for nodo in range(NODI_PERSONE)] + [-1] * NUMERO_ARMI
    ossa = {frozenset(osso) for osso in OSSA}
    tipi = torch.empty(NODI_PER_FRAME, NODI_PER_FRAME, dtype=torch.long)
    for i in range(NODI_PER_FRAME):
        for j in range(NODI_PER_FRAME):
            arma_i, arma_j = persona_del_nodo[i] < 0, persona_del_nodo[j] < 0
            if i == j:
                tipo = ARCO_SE_STESSO
            elif arma_i and arma_j:
                tipo = ARCO_ARMA_ARMA
            elif arma_i or arma_j:
                keypoint_persona = keypoint_del_nodo[j] if arma_i else keypoint_del_nodo[i]
                tipo = ARCO_ARMA_POLSO if keypoint_persona in POLSI else ARCO_ARMA_CORPO
            elif persona_del_nodo[i] != persona_del_nodo[j]:
                tipo = ARCO_PERSONE_DIVERSE
            elif frozenset((keypoint_del_nodo[i], keypoint_del_nodo[j])) in ossa:
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
        risultato = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                                   dropout_p=self.dropout.p if self.training else 0.0)
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
        self.blocchi = nn.ModuleList(BloccoSpazioTempo(dimensione, teste, dropout) for _ in range(blocchi))
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
        if not residuo:
            self.residuo = lambda x: 0
        elif canali_ingresso == canali_uscita and passo == 1:
            self.residuo = lambda x: x
        else:
            self.residuo = nn.Sequential(nn.Conv2d(canali_ingresso, canali_uscita, 1, (passo, 1)),
                                         nn.BatchNorm2d(canali_uscita))
        self.attivazione = nn.ReLU(inplace=True)

    def forward(self, x, adiacenza):
        """x [N, C, T, V], adiacenza [K, V, V]."""
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
        self.blocchi = nn.ModuleList(
            BloccoStGcn(ingresso, uscita, partizioni, passo, dropout if numero else 0.0, residuo)
            for numero, (ingresso, uscita, passo, residuo) in enumerate(configurazione))
        self.importanza_archi = nn.ParameterList(nn.Parameter(torch.ones_like(self.adiacenza))
                                                 for _ in self.blocchi)

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
    return sum(parametro.numel() for parametro in modello.parameters() if parametro.requires_grad)


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
    return {
        "accuratezza": float(veri_positivi.sum() / max(matrice.sum(), 1)),
        "macro_f1": float(f1[presenti].mean()) if presenti.any() else 0.0,
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

def carica_finestre(percorso: str | Path) -> dict[str, np.ndarray]:
    with np.load(percorso, allow_pickle=False) as dati:
        return {chiave: dati[chiave] for chiave in dati.files}


def lotti(indici: np.ndarray, dimensione_lotto: int, mescola: bool, generatore: np.random.Generator):
    ordine = generatore.permutation(indici) if mescola else indici
    for partenza in range(0, len(ordine), dimensione_lotto):
        yield ordine[partenza:partenza + dimensione_lotto]


def tensori_lotto(finestre, indici, dispositivo):
    return (torch.from_numpy(finestre["nodi"][indici].astype(np.float32)).to(dispositivo),
            torch.from_numpy(finestre["maschera"][indici]).to(dispositivo),
            torch.from_numpy(finestre["classe_arma"][indici].astype(np.int64)).to(dispositivo),
            torch.from_numpy(finestre["etichetta"][indici]).to(dispositivo))


@torch.no_grad()
def predici(modello, finestre, indici, dispositivo, dimensione_lotto=256) -> tuple[np.ndarray, np.ndarray]:
    modello.eval()
    probabilita = []
    for indici_lotto in lotti(indici, dimensione_lotto, False, None):
        nodi, maschera, classe_arma, _ = tensori_lotto(finestre, indici_lotto, dispositivo)
        with torch.autocast(dispositivo.type, enabled=dispositivo.type == "cuda"):
            probabilita.append(torch.softmax(modello(nodi, maschera, classe_arma).float(), -1).cpu().numpy())
    probabilita = np.concatenate(probabilita) if probabilita else np.empty((0, len(CLASSI)))
    return probabilita, finestre["etichetta"][indici]


def addestra(finestre: dict, nome_modello: str, cartella_uscita: Path, epoche: int = 50, dimensione_lotto: int = 64,
             tasso_apprendimento: float = 5e-4, decadimento_pesi: float = 0.05, pazienza: int = 10,
             bilanciamento: str = "pesi", seme: int = 0, iperparametri_modello: dict | None = None) -> dict:
    torch.manual_seed(seme)
    generatore = np.random.default_rng(seme)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cartella_uscita.mkdir(parents=True, exist_ok=True)

    split = dividi_per_gruppi(finestre["gruppo"], finestre["etichetta"], finestre["sorgente"], seme=seme,
                              gruppi_sempre_in_train=gruppi_visti_dal_detector())
    indici_split = {nome: np.flatnonzero(split == numero) for numero, nome in enumerate(NOMI_SPLIT)}
    # anche i video di ogni split: servono a scegliere i video della demo tra quelli MAI visti in addestramento
    np.savez(cartella_uscita / "split.npz", **indici_split,
             **{f"clip_{nome}": np.unique(finestre["id_clip"][indici]) for nome, indici in indici_split.items()})

    iperparametri_modello = iperparametri_modello or {}
    modello = crea_modello(nome_modello, **iperparametri_modello).to(dispositivo)
    conteggio_train = np.bincount(finestre["etichetta"][indici_split["train"]], minlength=len(CLASSI))
    if bilanciamento == "pesi":  # pesi inversamente proporzionali alla frequenza (classi assenti: peso 0)
        pesi = np.where(conteggio_train > 0, conteggio_train.sum() / np.maximum(conteggio_train, 1) / len(CLASSI), 0)
        funzione_perdita = nn.CrossEntropyLoss(weight=torch.tensor(pesi, dtype=torch.float32, device=dispositivo))
    else:
        funzione_perdita = nn.CrossEntropyLoss()
    ottimizzatore = torch.optim.AdamW(modello.parameters(), lr=tasso_apprendimento, weight_decay=decadimento_pesi)
    passi_totali = epoche * math.ceil(len(indici_split["train"]) / dimensione_lotto)
    passi_riscaldamento = max(1, passi_totali // 20)
    pianificatore = torch.optim.lr_scheduler.LambdaLR(ottimizzatore, lambda passo: min(
        (passo + 1) / passi_riscaldamento, 0.5 * (1 + math.cos(math.pi * min(passo / passi_totali, 1.0)))))
    scalatore = torch.amp.GradScaler(enabled=dispositivo.type == "cuda")

    print(f"{nome_modello}: {conta_parametri(modello):,} parametri, dispositivo {dispositivo}, finestre "
          + ", ".join(f"{nome}={len(indici)}" for nome, indici in indici_split.items()), flush=True)
    storia, miglior_f1, epoche_senza_miglioramento = [], -1.0, 0
    for epoca in range(1, epoche + 1):
        modello.train()
        inizio, perdita_totale = time.perf_counter(), 0.0
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
            miglior_f1, epoche_senza_miglioramento = metriche_val["macro_f1"], 0
            torch.save({"nome_modello": nome_modello, "iperparametri_modello": iperparametri_modello,
                        "classi": CLASSI, "stato": modello.state_dict(), "epoca": epoca}, cartella_uscita / "migliore.pt")
        else:
            epoche_senza_miglioramento += 1
            if epoche_senza_miglioramento >= pazienza:
                print(f"stop anticipato: {pazienza} epoche senza miglioramento", flush=True)
                break

    salvato = torch.load(cartella_uscita / "migliore.pt", map_location=dispositivo)
    modello.load_state_dict(salvato["stato"])
    risultati = {"modello": nome_modello, "parametri": conta_parametri(modello), "epoca_migliore": salvato["epoca"],
                 "storia": storia,
                 "impostazioni": {"epoche": epoche, "dimensione_lotto": dimensione_lotto,
                                  "tasso_apprendimento": tasso_apprendimento, "decadimento_pesi": decadimento_pesi,
                                  "bilanciamento": bilanciamento, "seme": seme, **iperparametri_modello}}
    for nome in ("val", "test"):
        probabilita, vere = predici(modello, finestre, indici_split[nome], dispositivo)
        risultati[nome] = metriche(vere, probabilita.argmax(1)) if len(vere) else None
        np.save(cartella_uscita / f"probabilita_{nome}.npy", probabilita)
        # per analizzare gli errori per sorgente e per video senza il file delle finestre (resta su Kaggle)
        indici = indici_split[nome]
        np.savez(cartella_uscita / f"finestre_{nome}.npz", etichetta=finestre["etichetta"][indici],
                 sorgente=finestre["sorgente"][indici], id_clip=finestre["id_clip"][indici],
                 inizio_finestra=finestre["inizio_finestra"][indici])
    risultati["latenza_cpu_ms_per_finestra"] = latenza_cpu_millisecondi(modello)
    (cartella_uscita / "risultati.json").write_text(json.dumps(risultati, indent=2, ensure_ascii=False))
    return risultati


def comando_addestra(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finestre", required=True)
    parser.add_argument("--modello", choices=["graph_transformer", "stgcn"], required=True)
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--epoche", type=int, default=50)
    parser.add_argument("--lotto", type=int, default=64)
    parser.add_argument("--tasso-apprendimento", type=float, default=5e-4)
    parser.add_argument("--bilanciamento", choices=["pesi", "campionamento", "nessuno"], default="pesi")
    parser.add_argument("--seme", type=int, default=0)
    argomenti = parser.parse_args(argomenti)
    risultati = addestra(carica_finestre(argomenti.finestre), argomenti.modello, Path(argomenti.uscita),
                         argomenti.epoche, argomenti.lotto, argomenti.tasso_apprendimento,
                         bilanciamento=argomenti.bilanciamento, seme=argomenti.seme)
    print(json.dumps({chiave: risultati[chiave] for chiave in ("val", "test", "latenza_cpu_ms_per_finestra")},
                     indent=2, ensure_ascii=False))

# ==============================================================================================================
# riga di comando: python -m sicurezza.modelli <comando> [opzioni]
# ==============================================================================================================
COMANDI = {"dataset-finestre": comando_dataset_finestre, "addestra": comando_addestra}


def main(argomenti: list[str] | None = None) -> None:
    argomenti = sys.argv[1:] if argomenti is None else argomenti
    if not argomenti or argomenti[0] not in COMANDI:
        print("uso: python -m sicurezza.modelli {dataset-finestre|addestra} [opzioni]  (--help dopo il comando)")
        raise SystemExit(2)
    COMANDI[argomenti[0]](argomenti[1:])


if __name__ == "__main__":
    main()
