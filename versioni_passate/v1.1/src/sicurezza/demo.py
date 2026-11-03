"""Argo — demo e pannello delle telecamere. File unico (richiesta di Marco: massimo 5 file Python)."""

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import csv
import json
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.parse

import cv2
import numpy as np
import torch

from .dati import CLASSI, FPS_ANALISI, FRAME_PER_FINESTRA, INDICE_CLASSE, OSSA, PASSO_FINESTRE, POLSI
from .dati import leggi_frame, leggi_info_video, nome_file_estratto
from .modelli import TimelineVideo, carica_timeline, costruisci_finestra, crea_modello, metriche, timeline_da_dati
from .visione import MODELLO_POSE_PREDEFINITO, TRACKER_PREDEFINITO, Estrattore, impacchetta_estrazione


# ==============================================================================================================
# allarme_arma
# WARNING ARMA (D6) e regola finale "violenza + arma confermata = violenza con arma".
#
# Soglie PROVVISORIE dal piano (≥ 0,5 di confidenza in ≥ 5 frame su 40, vicino a una persona): da tarare sul
# validation set. "Vicino" = centro dell'arma dentro il box della persona allargato di MARGINE_VICINANZA (PROVVISORIO).
# ==============================================================================================================

SOGLIA_CONFIDENZA_ARMA = 0.5
FRAME_MINIMI_CON_ARMA = 5
MARGINE_VICINANZA = 0.5


def arma_vicina_a_persona(box_arma: np.ndarray, box_persone: np.ndarray, margine: float = MARGINE_VICINANZA) -> bool:
    if len(box_persone) == 0:
        return False
    centro_x = (box_arma[0] + box_arma[2]) / 2
    centro_y = (box_arma[1] + box_arma[3]) / 2
    larghezza = box_persone[:, 2] - box_persone[:, 0]
    altezza = box_persone[:, 3] - box_persone[:, 1]
    # box di ogni persona allargato di `margine` volte la sua larghezza/altezza per lato
    sinistra = box_persone[:, 0] - margine * larghezza
    destra = box_persone[:, 2] + margine * larghezza
    sopra = box_persone[:, 1] - margine * altezza
    sotto = box_persone[:, 3] + margine * altezza
    dentro = (centro_x >= sinistra) & (centro_x <= destra) & (centro_y >= sopra) & (centro_y <= sotto)
    return bool(dentro.any())


# 19/10: la demo ha mostrato scarpe nere prese per pistole (vicine alla persona, ma ai piedi). Secondo criterio:
# "polso:d" = centro dell'arma entro d × altezza della persona da uno dei suoi polsi (un'arma vera è in mano).
# Quale criterio usare lo decide la taratura (D35) sul validation set, insieme a soglia e frame minimi.
# 19/10 (D44): "mani" = probabilità del classificatore sui ritagli intorno ai polsi; "polso:0.2+mani" = il più
# alto tra detector vicino al polso e mani.
VICINANZA_PREDEFINITA = f"persona:{MARGINE_VICINANZA}"
VICINANZE_DA_PROVARE = ("persona:0.25", "persona:0.5", "persona:1.0", "polso:0.1", "polso:0.2", "polso:0.35",
                        "mani", "polso:0.2+mani", "persona:0.5+mani")


def persone_nel_frame(timeline: TimelineVideo, frame: int):
    """(box [n, 4], polsi [n, 2, 3]) delle persone tracciate presenti nel frame."""
    box = []
    polsi = []
    for id_traccia, box_traccia in timeline.box_per_traccia.items():
        if np.isnan(box_traccia[frame, 0]):
            continue   # la persona non c'è in questo frame
        box.append(box_traccia[frame])
        polsi.append(timeline.keypoint_per_traccia[id_traccia][frame][list(POLSI)])
    # reshape: con zero persone gli array devono avere comunque la forma [0, 4] e [0, 2, 3]
    box = np.array(box, np.float32).reshape(-1, 4)
    polsi = np.array(polsi, np.float32).reshape(-1, 2, 3)
    return box, polsi


def arma_vicina(box_arma: np.ndarray, box_persone: np.ndarray, polsi_persone: np.ndarray, vicinanza: str) -> bool:
    """vicinanza "persona:0.5" = dentro il box della persona allargato; "polso:0.2" = entro 0,2 altezze da un polso."""
    criterio, valore = vicinanza.split(":")
    if criterio == "persona":
        return arma_vicina_a_persona(box_arma, box_persone, float(valore))
    if len(box_persone) == 0:
        return False
    centro = np.array([(box_arma[0] + box_arma[2]) / 2, (box_arma[1] + box_arma[3]) / 2])
    distanze = np.linalg.norm(polsi_persone[:, :, :2] - centro, axis=-1)          # [persone, 2 polsi]
    distanze = np.where(np.isnan(distanze), np.inf, distanze).min(axis=1)   # polso non visto = lontanissimo
    altezze = box_persone[:, 3] - box_persone[:, 1]
    return bool((distanze <= float(valore) * altezze).any())


def frame_con_arma(timeline: TimelineVideo, inizio_finestra: int, frame_per_finestra: int = FRAME_PER_FINESTRA,
                   soglia_confidenza: float = SOGLIA_CONFIDENZA_ARMA, vicinanza: str = VICINANZA_PREDEFINITA) -> int:
    """Quanti frame della finestra hanno almeno un'arma sicura vicino a una persona tracciata."""
    confidenze = confidenza_arma_per_frame(timeline, inizio_finestra, vicinanza, frame_per_finestra)
    return int((confidenze >= soglia_confidenza).sum())


def warning_arma(timeline: TimelineVideo, inizio_finestra: int, frame_minimi: int = FRAME_MINIMI_CON_ARMA,
                 soglia_confidenza: float = SOGLIA_CONFIDENZA_ARMA, vicinanza: str = VICINANZA_PREDEFINITA) -> bool:
    frame = frame_con_arma(timeline, inizio_finestra, soglia_confidenza=soglia_confidenza, vicinanza=vicinanza)
    return frame >= frame_minimi


def classe_finale(probabilita: np.ndarray, arma_confermata: bool) -> str:
    """Classe più probabile, con la regola di D6: violenza + arma confermata → violenza con arma."""
    classe = CLASSI[int(np.argmax(probabilita))]
    if classe == "violenza" and arma_confermata:
        return "violenza_con_arma"
    return classe


# ==============================================================================================================
# disegno
# Disegno dell'overlay sui frame della demo.
#
# >>> SEGNAPOSTO DESIGN <<<  Colori, font, posizioni e testi sono provvisori: la grafica la decide Marco.
# Tutto lo stile è in STILE_OVERLAY e TESTI_CLASSI, così si cambia senza toccare la logica.
# ==============================================================================================================

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
TESTI_CLASSI = {"normale": "Normal", "violenza": "Violence", "violenza_con_arma": "Armed violence",
                "scippo": "Snatching"}  # D64: testi visibili in inglese
TESTO_WARNING_ARMA = "WEAPON WARNING"
TESTO_ALLARME_VIOLENZA = "VIOLENCE ALARM"
NOMI_ARMI = ("pistol", "knife", "rifle")


def disegna_persone(frame: np.ndarray, persone: dict) -> None:
    stile = STILE_OVERLAY
    for id_traccia, box, keypoint in zip(persone["id_traccia"], persone["box"], persone["keypoint"]):
        visibile = keypoint[:, 2] >= stile["soglia_visibilita_keypoint"]
        for inizio, fine in OSSA:
            if visibile[inizio] and visibile[fine]:
                punto_inizio = (int(keypoint[inizio, 0]), int(keypoint[inizio, 1]))
                punto_fine = (int(keypoint[fine, 0]), int(keypoint[fine, 1]))
                cv2.line(frame, punto_inizio, punto_fine, stile["colore_scheletro"], stile["spessore"], cv2.LINE_AA)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), stile["colore_scheletro"], 1)
        cv2.putText(frame, f"ID {int(id_traccia)}", (x1, max(15, y1 - 5)), stile["font"], stile["scala_font"],
                    stile["colore_testo"], 1, cv2.LINE_AA)


def disegna_armi(frame: np.ndarray, armi: dict, soglia_confidenza: float) -> None:
    stile = STILE_OVERLAY
    for box, classe, confidenza in zip(armi["box"], armi["classe"], armi["confidenza"]):
        if confidenza < soglia_confidenza:
            continue
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), stile["colore_arma"], stile["spessore"])
        cv2.putText(frame, f"{NOMI_ARMI[int(classe)]} {confidenza:.2f}", (x1, max(15, y1 - 5)), stile["font"],
                    stile["scala_font"], stile["colore_arma"], 1, cv2.LINE_AA)


def disegna_pannello(frame: np.ndarray, probabilita, classe, warning: bool,
                     allarme: bool = False, attesa: str = "waiting for 40 frames...") -> None:
    """Pannello in alto a sinistra: classe decisa, barre delle probabilità, WARNING ARMA, ALLARME VIOLENZA.
    probabilita e classe sono None finché non c'è una decisione."""
    stile = STILE_OVERLAY
    larghezza_pannello = 330
    altezza_riga = 24
    inizio_barre = 160
    altezza_pannello = altezza_riga * (len(CLASSI) + 3) + 10
    sovrapposizione = frame.copy()
    cv2.rectangle(sovrapposizione, (0, 0), (larghezza_pannello, altezza_pannello), stile["sfondo_pannello"], -1)
    cv2.addWeighted(sovrapposizione, stile["opacita_pannello"], frame, 1 - stile["opacita_pannello"], 0, frame)
    if classe is None:
        cv2.putText(frame, attesa, (10, altezza_riga), stile["font"], stile["scala_font"],
                    stile["colore_testo"], 1, cv2.LINE_AA)
        if warning:   # modalità scheletri con il detector: il warning non ha bisogno del classificatore
            cv2.putText(frame, TESTO_WARNING_ARMA, (10, altezza_riga * (len(CLASSI) + 2)), stile["font"],
                        stile["scala_font"] * 1.2, stile["colore_arma"], stile["spessore"], cv2.LINE_AA)
        return
    cv2.putText(frame, TESTI_CLASSI[classe], (10, altezza_riga), stile["font"], stile["scala_font"] * 1.3,
                stile["colore_per_classe"][classe], stile["spessore"], cv2.LINE_AA)
    for indice, nome_classe in enumerate(CLASSI):
        valore = probabilita[indice]
        y = altezza_riga * (indice + 2)   # la prima riga è per la classe decisa
        cv2.rectangle(frame, (inizio_barre, y - 14), (inizio_barre + int(160 * valore), y - 2),
                      stile["colore_per_classe"][nome_classe], -1)
        cv2.putText(frame, TESTI_CLASSI[nome_classe], (10, y - 3), stile["font"], 0.45,
                    stile["colore_testo"], 1, cv2.LINE_AA)
    if warning:
        y = altezza_riga * (len(CLASSI) + 2)
        cv2.putText(frame, TESTO_WARNING_ARMA, (10, y), stile["font"], stile["scala_font"] * 1.2,
                    stile["colore_arma"], stile["spessore"], cv2.LINE_AA)
    if allarme:
        y = altezza_riga * (len(CLASSI) + 3)
        cv2.putText(frame, TESTO_ALLARME_VIOLENZA, (10, y), stile["font"], stile["scala_font"] * 1.2,
                    stile["colore_per_classe"]["violenza_con_arma"], stile["spessore"], cv2.LINE_AA)


# ==============================================================================================================
# inferenza
# Demo: un video → video annotato + eventi.json + probabilita.csv + keypoint.jsonl (D12).
#
# Ogni 10 frame, appena ci sono 40 frame, la finestra appena conclusa passa al classificatore. Il pannello mostra
# l'ultima decisione fino alla successiva. Sul portatile va più lento del tempo reale (~2,7 frame/s), ma il video
# prodotto è a 25 fps e fluido.
#
# Uso:
#     python -m sicurezza.demo analizza-video --video ingresso.mp4 --classificatore runs/gt/migliore.pt \
#         --modello-armi runs/armi/yolo26s_armi/weights/best.pt --uscita demo/
# ==============================================================================================================

def carica_classificatore(percorso_checkpoint: str, dispositivo: str = "cpu"):
    salvato = torch.load(percorso_checkpoint, map_location=dispositivo)
    modello = crea_modello(salvato["nome_modello"], **salvato["iperparametri_modello"])
    modello.load_state_dict(salvato["stato"])
    return modello.to(dispositivo).eval()


# DA DECIDERE 22 d: con più checkpoint la probabilità è la media delle softmax (sulle clip mai viste, add. 1 + add. 2:
# macro-F1 0,409 contro 0,395 del solo add. 1). Con un solo checkpoint è il classificatore di sempre.
def probabilita_media(classificatori: list, nodi: np.ndarray, maschera: np.ndarray, classe_arma: np.ndarray,
                      dispositivo: str = "cpu") -> np.ndarray:
    # [None] aggiunge la dimensione del lotto: un lotto da una finestra sola
    nodi = torch.from_numpy(nodi)[None].to(dispositivo)
    maschera = torch.from_numpy(maschera)[None].to(dispositivo)
    classe_arma = torch.from_numpy(classe_arma.astype(np.int64))[None].to(dispositivo)
    probabilita = []
    for modello in classificatori:
        logit = modello(nodi, maschera, classe_arma)
        probabilita.append(torch.softmax(logit, -1)[0].cpu().numpy())
    return np.mean(probabilita, axis=0)


# Ramo pixel (D48), attivo con --testa-pixel: R(2+1)D-18 di torchvision preaddestrato su
# Kinetics-400 e CONGELATO → vettore da 512 → regressione logistica violenza sì/no (testa allenata sul train
# dell'add. 2, `experiments_and_future_work/testa_rgb.npz`). Stessa preparazione dei kernel `prova-rgb`: 16 frame
# dei 40, 171×128, ritaglio centrale 112×112. Fusione: media tra la probabilità dei pixel e quella di violenza degli
# scheletri; sulle clip intere mai viste 0,82 per pezzi da 30 s e 0,75 per finestra (scheletri da soli 0,78 e 0,71).
# ~1 s di CPU per finestra.
LATO_PIXEL = (171, 128)                 # larghezza, altezza prima del ritaglio centrale 112×112
FRAME_PIXEL = 16
MEDIA_KINETICS = np.array([0.43216, 0.394666, 0.37645], np.float32)
DEVIAZIONE_KINETICS = np.array([0.22803, 0.22145, 0.216989], np.float32)


def riduci_per_pixel(frame: np.ndarray) -> np.ndarray:
    """Frame BGR del lettore → RGB 171×128 (come `scale=171:128` di ffmpeg nei kernel, che è bicubico)."""
    ridotto = cv2.resize(frame, LATO_PIXEL, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(ridotto, cv2.COLOR_BGR2RGB)


class RamoPixel:
    """percorso_rete: pesi della rete fine-tunata sulla violenza (D53, `rete_violenza.pt` del kernel finetune-rgb,
    con la sua `testa_violenza.npz`); senza, la rete Kinetics congelata di D48."""
    def __init__(self, percorso_testa: str, dispositivo: str = "cpu", percorso_rete=None):
        import torchvision
        if percorso_rete:
            pesi = None   # i pesi arrivano dal file della rete fine-tunata
        else:
            pesi = torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1
        self.rete = torchvision.models.video.r2plus1d_18(weights=pesi)
        self.rete.fc = torch.nn.Identity()   # senza l'ultimo strato: la rete dà il vettore da 512
        if percorso_rete:
            stato = torch.load(percorso_rete, map_location="cpu")
            stato_float = {}
            for chiave, valore in stato.items():
                stato_float[chiave] = valore.float()
            self.rete.load_state_dict(stato_float)
        self.rete = self.rete.to(dispositivo).eval()
        self.dispositivo = dispositivo
        self.testa = {}
        with np.load(percorso_testa) as testa:
            for chiave in ("media", "scala", "pesi", "bias"):
                self.testa[chiave] = testa[chiave]

    @torch.no_grad()
    def vettore(self, frame_ridotti: list) -> np.ndarray:
        # 16 frame distribuiti in modo uniforme tra i 40, poi il ritaglio centrale 112×112
        scelti = np.linspace(0, len(frame_ridotti) - 1, FRAME_PIXEL).round().astype(int)
        frame_scelti = []
        for indice in scelti:
            frame_scelti.append(frame_ridotti[indice])
        ritagli = np.stack(frame_scelti)[:, 8:120, 29:141]
        ingresso = (ritagli.astype(np.float32) / 255 - MEDIA_KINETICS) / DEVIAZIONE_KINETICS   # [16, 112, 112, 3]
        # la rete vuole [lotto, canali, frame, altezza, larghezza]
        ingresso = torch.from_numpy(ingresso).permute(3, 0, 1, 2)[None].to(self.dispositivo)
        return self.rete(ingresso)[0].cpu().numpy()

    def probabilita_violenza(self, frame_ridotti: list) -> float:
        testa = self.testa
        logit = ((self.vettore(frame_ridotti) - testa["media"]) / testa["scala"]) @ testa["pesi"] + testa["bias"]
        return float(1 / (1 + np.exp(-logit)))


# Rete video a 5 uscite (D70; addestrata in D68, kernel experiments_and_future_work/rete_video del branch
# esperimenti-armi-scippi): R(2+1)D-18 fine-tunata con violenza, arma, fuoco, coltello, scippo sì/no. Due copie:
# b1 (112 px) decide lo SCIPPO, chiamata ogni 0,8 s; b2 (224 px) dà il WARNING ARMA al posto del detector, chiamata
# ogni 4 s (sulle armi non perde niente rispetto a ogni 0,4 s; gli scippi durano 2-3 s e vogliono b1 spesso).
# Pezzi da 10 s mai visti, misura incrociata: b2 arma 0,78 / fuoco 0,75 / coltello 0,84; b1 scippo 0,82.
# Soglie scelte sul val alle stesse cadenze (runs/esperimenti_armi_scippi/diradate/confronto_b1.json e _b2.json).
# Preparazione identica al kernel: frame a (lato·8/7) di altezza in 4:3, 16 dei 40, ritaglio centrale lato×lato.
USCITE_RETE_VIDEO = ("violenza", "arma", "fuoco", "coltello", "scippo")
PASSO_RETE_SCIPPI = 20     # frame: b1 ogni 0,8 s
PASSO_RETE_ARMI = 100      # frame: b2 ogni 4 s
SOGLIA_SCIPPO_B1 = 0.0715  # test 0,91, recall 0,94, falsi allarmi sulle folle 5% dei pezzi da 10 s
SOGLIA_ARMA_B2 = 0.3467    # test 0,77, recall 0,64, falsi allarmi sulle folle 14%


class ReteVideoArmiScippi:
    def __init__(self, percorso_rete: str, lato: int, dispositivo: str = "cpu"):
        import torchvision
        self.rete = torchvision.models.video.r2plus1d_18(weights=None)
        self.rete.fc = torch.nn.Linear(512, len(USCITE_RETE_VIDEO))
        stato_float = {}
        for chiave, valore in torch.load(percorso_rete, map_location="cpu").items():
            stato_float[chiave] = valore.float()   # il kernel salva in mezza precisione
        self.rete.load_state_dict(stato_float)
        self.rete = self.rete.to(dispositivo).eval()
        self.dispositivo = dispositivo
        self.lato = lato
        self.altezza = lato * 8 // 7                                 # 128 per 112, 256 per 224
        self.larghezza = self.altezza * 4 // 3 + (self.altezza * 4 // 3) % 2   # 170 per 112, 342 per 224

    def riduci(self, frame: np.ndarray) -> np.ndarray:
        """Frame BGR del lettore → RGB alla misura del kernel (bicubico come `scale=` di ffmpeg)."""
        ridotto = cv2.resize(frame, (self.larghezza, self.altezza), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(ridotto, cv2.COLOR_BGR2RGB)

    @torch.no_grad()
    def probabilita(self, frame_ridotti: list) -> dict:
        scelti = np.linspace(0, len(frame_ridotti) - 1, FRAME_PIXEL).round().astype(int)
        frame_scelti = []
        for indice in scelti:
            frame_scelti.append(frame_ridotti[indice])
        y = (self.altezza - self.lato) // 2
        x = (self.larghezza - self.lato) // 2
        ritagli = np.stack(frame_scelti)[:, y:y + self.lato, x:x + self.lato]
        ingresso = (ritagli.astype(np.float32) / 255 - MEDIA_KINETICS) / DEVIAZIONE_KINETICS
        ingresso = torch.from_numpy(ingresso).permute(3, 0, 1, 2)[None].to(self.dispositivo)
        valori = torch.sigmoid(self.rete(ingresso))[0].cpu().numpy()
        risultato = {}
        for numero, nome in enumerate(USCITE_RETE_VIDEO):
            risultato[nome] = float(valori[numero])
        return risultato


def classe_con_scippo(classe: str, allarme_scippo: bool) -> str:
    """D70: lo scippo lo decide b1; la violenza con arma resta (è più grave)."""
    if allarme_scippo and classe != "violenza_con_arma":
        return "scippo"
    return classe


def fondi_con_pixel(probabilita: np.ndarray, probabilita_pixel: float) -> np.ndarray:
    """4 classi con la violenza = media tra pixel e scheletri; dentro i due gruppi (classi violente / non violente)
    restano le proporzioni degli scheletri."""
    violente = np.isin(CLASSI, CLASSI_VIOLENTE)
    totale_violenza = (probabilita[violente].sum() + probabilita_pixel) / 2
    fuse = probabilita.astype(np.float64)
    for gruppo, totale in ((violente, totale_violenza), (~violente, 1 - totale_violenza)):
        somma = fuse[gruppo].sum()
        if somma > 0:
            fuse[gruppo] = fuse[gruppo] / somma * totale
        else:
            fuse[gruppo] = totale / gruppo.sum()   # tutto a zero: il totale diviso in parti uguali
    return fuse.astype(np.float32)


def classe_finale_con_pixel(probabilita_fuse: np.ndarray, arma_confermata: bool) -> str:
    """Prima violenza sì/no (gruppo violento > 0,5, come nella misura della fusione), poi la classe più probabile del
    gruppo; infine la regola di D6 come in `classe_finale`."""
    violente = np.isin(CLASSI, CLASSI_VIOLENTE)
    if probabilita_fuse[violente].sum() > 0.5:
        gruppo = np.flatnonzero(violente)
    else:
        gruppo = np.flatnonzero(~violente)
    classe = CLASSI[int(gruppo[np.argmax(probabilita_fuse[gruppo])])]
    if classe == "violenza" and arma_confermata:
        return "violenza_con_arma"
    return classe


# ALLARME VIOLENZA (D52): la stessa regola con cui si misura il sistema sui pezzi di video (`valuta-allarme`,
# `runs/finetune_rgb/fusione_finetune.json`): probabilità mediate sulle ultime MEDIA_MOBILE_ALLARME finestre, allarme
# quando FINESTRE_DI_FILA_ALLARME finestre di fila sono violente. 7 e 1 = scelti sul validation per la rete video
# fine-tunata (D53) + scheletri a pezzi da 30 s (0,90 sul test). Con la rete congelata di D48 erano 5 e 5 (0,82).
# Con i pixel "violenta" = gruppo violento > 0,5 (come nella misura della fusione); senza, la classe più probabile
# (come `allarmi_per_video`).
FINESTRE_DI_FILA_ALLARME = 7
MEDIA_MOBILE_ALLARME = 1


class AllarmeViolenza:
    def __init__(self, finestre_di_fila: int = FINESTRE_DI_FILA_ALLARME, media: int = MEDIA_MOBILE_ALLARME,
                 con_pixel: bool = True):
        self.finestre_di_fila = finestre_di_fila
        self.con_pixel = con_pixel
        self.ultime = deque(maxlen=media)   # una coda che tiene solo le ultime `media` probabilità
        self.di_fila = 0

    def aggiorna(self, probabilita: np.ndarray) -> bool:
        self.ultime.append(probabilita)
        media = np.mean(self.ultime, axis=0)
        if self.con_pixel:
            violenta = classe_finale_con_pixel(media, False) in CLASSI_VIOLENTE
        else:
            violenta = CLASSI[int(np.argmax(media))] in CLASSI_VIOLENTE
        if violenta:
            self.di_fila += 1
        else:
            self.di_fila = 0
        return self.di_fila >= self.finestre_di_fila


class ScrittoreVideo:
    def __init__(self, percorso: Path, larghezza: int, altezza: int, fps: int = FPS_ANALISI):
        self.processo = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{larghezza}x{altezza}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(percorso)],
            stdin=subprocess.PIPE)

    def scrivi(self, frame: np.ndarray) -> None:
        self.processo.stdin.write(np.ascontiguousarray(frame).tobytes())

    def chiudi(self) -> None:
        self.processo.stdin.close()
        self.processo.wait()


class AnalisiFlusso:
    """Il sistema su un frame alla volta (D56): lo usano la demo su file (`analizza_video`) e il pannello di Argo.

    Senza classificatore ("modalità scheletri", per chi non ha i pesi) restano scheletri, tracking e, se c'è il
    detector, armi e WARNING ARMA. taratura: la combinazione "migliore" di `tara-warning` (soglia, frame minimi,
    vicinanza); senza, i valori provvisori del piano. tracker: lo stesso usato per estrarre i dati di addestramento
    del classificatore (D3). percorso_classificatore: un checkpoint, oppure una lista di checkpoint di cui fare la
    media (DA DECIDERE 22 d). percorso_testa_pixel: con la testa RGB, fusione pixel + scheletri sulla violenza (D48).
    finestre_di_fila, media_mobile_allarme: regola dell'ALLARME VIOLENZA (D52). solo_scheletri: disegno su fondo nero
    (video pubblici dei dataset, D58). pannello_nel_frame: False nel pannello web, che mostra le classi per conto suo.
    percorso_rete_scippi, percorso_rete_armi: le reti video b1 e b2 (D70): b1 decide lo scippo, b2 dà il WARNING
    ARMA al posto del detector (che resta per i riquadri e per i nodi arma degli scheletri)."""

    def __init__(self, percorso_classificatore=None, percorso_modello_armi=None,
                 percorso_modello_pose: str = MODELLO_POSE_PREDEFINITO, dispositivo: str = "cpu",
                 taratura=None, percorso_modello_mani=None,
                 tracker: str = TRACKER_PREDEFINITO, percorso_testa_pixel=None,
                 finestre_di_fila: int = FINESTRE_DI_FILA_ALLARME, media_mobile_allarme: int = MEDIA_MOBILE_ALLARME,
                 percorso_rete_pixel=None, solo_scheletri: bool = False, pannello_nel_frame: bool = True,
                 percorso_rete_scippi=None, percorso_rete_armi=None):
        # regola del WARNING ARMA: i valori provvisori del piano, sostituiti da quelli della taratura se c'è
        self.regola = {"soglia_confidenza": SOGLIA_CONFIDENZA_ARMA, "frame_minimi": FRAME_MINIMI_CON_ARMA,
                       "vicinanza": VICINANZA_PREDEFINITA}
        if taratura is not None:
            for chiave in self.regola:
                if chiave in taratura:
                    self.regola[chiave] = taratura[chiave]
        self.dispositivo = dispositivo
        self.estrattore = Estrattore(percorso_modello_pose, percorso_modello_armi, dispositivo, tracker=tracker,
                                     percorso_modello_mani=percorso_modello_mani)   # D44: anche le mani, se c'è il modello

        # percorso_classificatore: nessuno, un file, oppure una lista di file
        if percorso_classificatore is None:
            percorsi = []
        elif isinstance(percorso_classificatore, str):
            percorsi = [percorso_classificatore]
        else:
            percorsi = percorso_classificatore
        self.classificatori = []
        for percorso in percorsi:
            self.classificatori.append(carica_classificatore(percorso, dispositivo))

        self.ramo_pixel = None
        if percorso_testa_pixel and len(self.classificatori) > 0:
            self.ramo_pixel = RamoPixel(percorso_testa_pixel, dispositivo, percorso_rete_pixel)
        self.allarme_violenza = AllarmeViolenza(finestre_di_fila, media_mobile_allarme,
                                                con_pixel=self.ramo_pixel is not None)
        # D70: le reti video b1 (scippi) e b2 (armi), con i loro ultimi 40 frame ridotti e l'ultimo risultato
        self.rete_scippi = None
        self.rete_armi = None
        if percorso_rete_scippi and len(self.classificatori) > 0:
            self.rete_scippi = ReteVideoArmiScippi(percorso_rete_scippi, 112, dispositivo)
        if percorso_rete_armi and len(self.classificatori) > 0:
            self.rete_armi = ReteVideoArmiScippi(percorso_rete_armi, 224, dispositivo)
        self.ridotti_scippi = deque(maxlen=FRAME_PER_FINESTRA)
        self.ridotti_armi = deque(maxlen=FRAME_PER_FINESTRA)
        self.ultima_scippi = None
        self.ultima_armi = None
        self.solo_scheletri = solo_scheletri
        self.pannello_nel_frame = pannello_nel_frame
        self.ultimi_frame = deque(maxlen=FRAME_PER_FINESTRA)   # (persone, armi) degli ultimi 40 frame
        self.ultimi_ridotti = deque(maxlen=FRAME_PER_FINESTRA)  # gli stessi frame a 171×128 RGB, solo per il ramo pixel
        self.indice_frame = 0
        self.decisione = {"probabilita": None, "classe": None, "warning": False, "allarme": False}

    @torch.no_grad()
    def analizza(self, frame: np.ndarray):
        """→ (frame annotato, persone, armi, evento della finestra appena conclusa oppure None)."""
        persone, armi = self.estrattore.analizza_frame(frame, primo_frame_del_video=self.indice_frame == 0)
        self.ultimi_frame.append((persone, armi))
        if self.ramo_pixel:
            self.ultimi_ridotti.append(riduci_per_pixel(frame))
        if self.rete_scippi:
            self.ridotti_scippi.append(self.rete_scippi.riduci(frame))
        if self.rete_armi:
            self.ridotti_armi.append(self.rete_armi.riduci(frame))
        inizio_finestra = self.indice_frame - FRAME_PER_FINESTRA + 1
        evento = None
        if inizio_finestra >= 0 and inizio_finestra % PASSO_FINESTRE == 0:
            evento = self.decidi(inizio_finestra, frame.shape[1], frame.shape[0])
        if self.solo_scheletri:
            annotato = np.zeros_like(frame)   # fondo nero
        else:
            annotato = frame.copy()
        disegna_persone(annotato, persone)
        disegna_armi(annotato, armi, self.regola["soglia_confidenza"])
        if self.pannello_nel_frame:
            if len(self.classificatori) > 0:
                attesa = "waiting for 40 frames..."
            else:
                attesa = "skeletons only: weights not found"
            decisione = self.decisione
            disegna_pannello(annotato, decisione["probabilita"], decisione["classe"], decisione["warning"],
                             decisione["allarme"], attesa)
        self.indice_frame += 1
        return annotato, persone, armi, evento

    def decidi(self, inizio_finestra: int, larghezza: int, altezza: int) -> dict:
        # gli ultimi 40 frame diventano una piccola timeline, come quelle dell'addestramento
        righe_persone = []
        righe_armi = []
        for numero, (dati_persone, dati_armi) in enumerate(self.ultimi_frame):
            righe_persone.append((numero, dati_persone))
            righe_armi.append((numero, dati_armi))
        dati = impacchetta_estrazione(righe_persone, righe_armi, FRAME_PER_FINESTRA, larghezza, altezza, 0.0)
        timeline = timeline_da_dati(dati)
        warning = warning_arma(timeline, 0, self.regola["frame_minimi"], self.regola["soglia_confidenza"],
                               self.regola["vicinanza"])
        evento = {"inizio_secondi": inizio_finestra / FPS_ANALISI, "fine_secondi": (self.indice_frame + 1) / FPS_ANALISI}
        if len(self.classificatori) == 0:   # modalità scheletri: solo il WARNING ARMA
            self.decisione = {"probabilita": None, "classe": None, "warning": warning, "allarme": False}
            evento.update({"classe": None, "warning_arma": warning, "allarme_violenza": False, "probabilita": None})
            return evento
        nodi, maschera, classe_arma = costruisci_finestra(timeline, 0)
        probabilita = probabilita_media(self.classificatori, nodi, maschera, classe_arma, self.dispositivo)
        # D70: b2 ogni 4 s e b1 ogni 0,8 s; tra una chiamata e l'altra vale l'ultimo risultato
        if self.rete_armi and inizio_finestra % PASSO_RETE_ARMI == 0:
            self.ultima_armi = self.rete_armi.probabilita(list(self.ridotti_armi))
        if self.rete_scippi and inizio_finestra % PASSO_RETE_SCIPPI == 0:
            self.ultima_scippi = self.rete_scippi.probabilita(list(self.ridotti_scippi))
        if self.ultima_armi is not None:
            warning = self.ultima_armi["arma"] >= SOGLIA_ARMA_B2
        allarme_scippo = self.ultima_scippi is not None and self.ultima_scippi["scippo"] >= SOGLIA_SCIPPO_B1
        if self.ramo_pixel:
            probabilita_pixel = self.ramo_pixel.probabilita_violenza(list(self.ultimi_ridotti))
            probabilita = fondi_con_pixel(probabilita, probabilita_pixel)
        # con b1 lo scippo dello ST-GCN non sceglie più la classe (le probabilità salvate restano quelle vere)
        per_la_classe = probabilita
        if self.rete_scippi:
            per_la_classe = probabilita.copy()
            per_la_classe[INDICE_CLASSE["scippo"]] = 0
        if self.ramo_pixel:
            classe = classe_finale_con_pixel(per_la_classe, warning)
        else:
            classe = classe_finale(per_la_classe, warning)
        if self.rete_scippi:
            classe = classe_con_scippo(classe, allarme_scippo)
        allarme = self.allarme_violenza.aggiorna(probabilita)
        self.decisione = {"probabilita": probabilita, "classe": classe, "warning": warning, "allarme": allarme}
        arrotondate = np.round(probabilita, 4).tolist()
        probabilita_per_classe = {}
        for indice, nome in enumerate(CLASSI):
            probabilita_per_classe[nome] = arrotondate[indice]
        evento.update({"classe": classe, "warning_arma": warning, "allarme_violenza": allarme,
                       "probabilita": probabilita_per_classe})
        if self.ramo_pixel:
            evento["probabilita_violenza_pixel"] = round(probabilita_pixel, 4)
        if self.ultima_armi is not None:
            evento["probabilita_arma_video"] = round(self.ultima_armi["arma"], 4)
        if self.rete_scippi:
            evento["allarme_scippo"] = allarme_scippo
            if self.ultima_scippi is not None:
                evento["probabilita_scippo_video"] = round(self.ultima_scippi["scippo"], 4)
        return evento


def riga_keypoint(indice_frame: int, persone: dict, armi: dict) -> dict:
    """Una riga di keypoint.jsonl: chi c'è nel frame, dove, e le armi viste (in pixel del frame)."""
    lista_persone = []
    for id_traccia, box, keypoint in zip(persone["id_traccia"], persone["box"], persone["keypoint"]):
        lista_persone.append({"id": int(id_traccia), "box": np.round(np.asarray(box, float), 1).tolist(),
                              "keypoint": np.round(np.asarray(keypoint, float), 3).tolist()})
    lista_armi = []
    for box, classe, confidenza in zip(armi["box"], armi["classe"], armi["confidenza"]):
        lista_armi.append({"classe": NOMI_ARMI[int(classe)], "confidenza": round(float(confidenza), 3),
                           "box": np.round(np.asarray(box, float), 1).tolist()})
    return {"frame": indice_frame, "secondi": round(indice_frame / FPS_ANALISI, 2),
            "persone": lista_persone, "armi": lista_armi}


def salva_eventi(cartella_uscita: Path, eventi: list) -> None:
    (cartella_uscita / "eventi.json").write_text(json.dumps(eventi, indent=2, ensure_ascii=False))
    with open(cartella_uscita / "probabilita.csv", "w", newline="") as file_csv:
        scrittore_csv = csv.writer(file_csv)
        scrittore_csv.writerow(["inizio_secondi", "fine_secondi"] + list(CLASSI)
                               + ["classe", "warning_arma", "allarme_violenza"])
        for evento in eventi:
            probabilita = []
            for classe in CLASSI:
                if evento["probabilita"]:
                    probabilita.append(evento["probabilita"][classe])
                else:
                    probabilita.append("")   # modalità scheletri: niente probabilità
            scrittore_csv.writerow([evento["inizio_secondi"], evento["fine_secondi"]] + probabilita
                                   + [evento["classe"], evento["warning_arma"], evento["allarme_violenza"]])


def analizza_video(percorso_video: str, percorso_classificatore, cartella_uscita,
                   percorso_modello_armi=None, percorso_modello_pose: str = MODELLO_POSE_PREDEFINITO,
                   dispositivo: str = "cpu", taratura=None,
                   percorso_modello_mani=None, tracker: str = TRACKER_PREDEFINITO,
                   percorso_testa_pixel=None, finestre_di_fila: int = FINESTRE_DI_FILA_ALLARME,
                   media_mobile_allarme: int = MEDIA_MOBILE_ALLARME, percorso_rete_pixel=None,
                   solo_scheletri: bool = False, pannello_nel_frame: bool = True,
                   percorso_rete_scippi=None, percorso_rete_armi=None) -> list:
    """Un video → video_annotato.mp4, eventi.json, probabilita.csv e keypoint.jsonl (opzioni come `AnalisiFlusso`)."""
    cartella_uscita = Path(cartella_uscita)
    cartella_uscita.mkdir(parents=True, exist_ok=True)
    info = leggi_info_video(percorso_video)
    analisi = AnalisiFlusso(percorso_classificatore, percorso_modello_armi, percorso_modello_pose, dispositivo, taratura,
                            percorso_modello_mani, tracker, percorso_testa_pixel, finestre_di_fila, media_mobile_allarme,
                            percorso_rete_pixel, solo_scheletri, pannello_nel_frame, percorso_rete_scippi,
                            percorso_rete_armi)
    scrittore = ScrittoreVideo(cartella_uscita / "video_annotato.mp4", info.larghezza, info.altezza)
    eventi = []
    with open(cartella_uscita / "keypoint.jsonl", "w") as file_keypoint:
        for indice_frame, frame in enumerate(leggi_frame(percorso_video)):
            annotato, persone, armi, evento = analisi.analizza(frame)
            file_keypoint.write(json.dumps(riga_keypoint(indice_frame, persone, armi)) + "\n")
            if evento:
                eventi.append(evento)
            scrittore.scrivi(annotato)
    scrittore.chiudi()
    salva_eventi(cartella_uscita, eventi)
    return eventi


def comando_analizza_video(argomenti=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--classificatore", required=True, nargs="+",
                        help="migliore.pt di `python -m sicurezza.modelli addestra`; con più file, media delle probabilità")
    parser.add_argument("--modello-armi", default=None)
    parser.add_argument("--modello-pose", default=MODELLO_POSE_PREDEFINITO)
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--dispositivo", default="cpu")
    parser.add_argument("--taratura", default=None, help="taratura_warning.json di `tara-warning` (D35)")
    parser.add_argument("--modello-mani", default=None, help="classificatore dell'ingrandimento sulle mani (D44)")
    parser.add_argument("--tracker", default=TRACKER_PREDEFINITO, help="lo stesso dell'estrazione dei dati (D3)")
    parser.add_argument("--testa-pixel", default=None,
                        help="testa_violenza.npz (o la vecchia experiments_and_future_work/testa_rgb.npz): "
                             "fusione pixel + scheletri sulla violenza (D48)")
    parser.add_argument("--rete-pixel", default=None,
                        help="rete_violenza.pt fine-tunata (D53), da usare con la sua --testa-pixel testa_violenza.npz")
    parser.add_argument("--rete-scippi", default=None, help="rete_scippi_b1.pt: decide lo scippo (D70)")
    parser.add_argument("--rete-armi", default=None, help="rete_armi_b2.pt: WARNING ARMA al posto del detector (D70)")
    parser.add_argument("--finestre-di-fila", type=int, default=FINESTRE_DI_FILA_ALLARME,
                        help="ALLARME VIOLENZA dopo tante finestre violente di fila (D52)")
    parser.add_argument("--media-mobile", type=int, default=MEDIA_MOBILE_ALLARME,
                        help="finestre su cui mediare le probabilità per l'allarme (D52)")
    parser.add_argument("--solo-scheletri", action="store_true",
                        help="skeletons and weapons on black, without the video pixels")
    parser.add_argument("--senza-pannello", action="store_true",
                        help="no class box in the video (draw it yourself from eventi.json)")
    argomenti = parser.parse_args(argomenti)
    taratura = None
    if argomenti.taratura:
        taratura = json.loads(Path(argomenti.taratura).read_text())["migliore"]
    eventi = analizza_video(argomenti.video, argomenti.classificatore, argomenti.uscita, argomenti.modello_armi,
                            argomenti.modello_pose, argomenti.dispositivo, taratura, argomenti.modello_mani,
                            argomenti.tracker, argomenti.testa_pixel, argomenti.finestre_di_fila,
                            argomenti.media_mobile, argomenti.rete_pixel, argomenti.solo_scheletri,
                            not argomenti.senza_pannello, argomenti.rete_scippi, argomenti.rete_armi)
    print(f"{len(eventi)} finestre analizzate → {argomenti.uscita}")

# ==============================================================================================================
# taratura del WARNING ARMA (D35: massimo F1 sul validation set)
# Riferimento: una finestra "ha un'arma" se la sua etichetta è violenza_con_arma (le altre classi non hanno armi
# annotate). Per ogni finestra si calcola, per frame, la confidenza massima di un'arma vicina a una persona; poi si
# provano tutte le combinazioni di soglia di confidenza, frame minimi e criterio di vicinanza.
# ==============================================================================================================
SOGLIE_DA_PROVARE = (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
FRAME_MINIMI_DA_PROVARE = tuple(range(1, 16))


def confidenza_arma_per_frame(timeline: TimelineVideo, inizio_finestra: int, vicinanza: str,
                              frame_per_finestra: int = FRAME_PER_FINESTRA) -> np.ndarray:
    """[frame_per_finestra]: confidenza massima di un'arma vicina a una persona in ogni frame (0 se nessuna).
    `vicinanza` può unire più criteri con "+": vale il più alto."""
    confidenze = np.zeros(frame_per_finestra, np.float32)
    fine_finestra = inizio_finestra + frame_per_finestra
    criteri = vicinanza.split("+")
    if "mani" in criteri and timeline.mani_per_frame is not None:
        mani = timeline.mani_per_frame[inizio_finestra:fine_finestra].max(axis=1)
        confidenze[:len(mani)] = mani
    criteri_detector = []
    for criterio in criteri:
        if criterio != "mani":
            criteri_detector.append(criterio)
    if len(criteri_detector) == 0:
        return confidenze
    for indice in range(len(timeline.armi_frame)):
        frame = int(timeline.armi_frame[indice])
        if frame < inizio_finestra or frame >= fine_finestra:
            continue   # arma fuori dalla finestra
        box_persone, polsi_persone = persone_nel_frame(timeline, frame)
        vicina = False
        for criterio in criteri_detector:
            if arma_vicina(timeline.armi_box[indice], box_persone, polsi_persone, criterio):
                vicina = True
                break
        if vicina:
            posizione = frame - inizio_finestra
            confidenze[posizione] = max(confidenze[posizione], float(timeline.armi_confidenza[indice]))
    return confidenze


def tara_warning_arma(confidenze_per_vicinanza: dict, arma_vera: np.ndarray) -> dict:
    """confidenze_per_vicinanza: criterio di vicinanza → [finestre, frame]; arma_vera: [finestre] bool. Ritorna la
    combinazione con F1 massimo e la tabella completa."""
    tabella = []
    for vicinanza, confidenze in confidenze_per_vicinanza.items():
        for soglia in SOGLIE_DA_PROVARE:
            frame_sopra_soglia = (confidenze >= soglia).sum(axis=1)
            for frame_minimi in FRAME_MINIMI_DA_PROVARE:
                previsto = frame_sopra_soglia >= frame_minimi
                veri_positivi = int((previsto & arma_vera).sum())
                precisione = veri_positivi / max(int(previsto.sum()), 1)
                recall = veri_positivi / max(int(arma_vera.sum()), 1)
                f1 = 2 * precisione * recall / max(precisione + recall, 1e-12)
                tabella.append({"vicinanza": vicinanza, "soglia_confidenza": soglia, "frame_minimi": frame_minimi,
                                "precisione": round(precisione, 4), "recall": round(recall, 4), "f1": round(f1, 4)})
    # la migliore: F1 più alto; a parità, precisione più alta; a parità ancora, la prima della tabella
    migliore = tabella[0]
    for riga in tabella:
        if (riga["f1"], riga["precisione"]) > (migliore["f1"], migliore["precisione"]):
            migliore = riga
    return {"migliore": migliore, "tabella": tabella}


def percorsi_degli_estratti(cartelle: list) -> dict:
    """nome del file → percorso, per tutti i .npz delle cartelle."""
    percorso_per_nome = {}
    for cartella in cartelle:
        for file in Path(cartella).glob("*.npz"):
            percorso_per_nome[file.name] = file
    return percorso_per_nome


def comando_tara_warning(argomenti=None) -> None:
    parser = argparse.ArgumentParser(description="Taratura del WARNING ARMA (D35) sulle finestre di validazione")
    parser.add_argument("--finestre", required=True, help="finestre.npz (id_clip, inizio_finestra, etichetta)")
    parser.add_argument("--split", required=True, help="split.npz prodotto dall'addestramento (indici 'val')")
    parser.add_argument("--estratti", nargs="+", required=True)
    parser.add_argument("--uscita", required=True, help="file .json con migliore combinazione e tabella")
    argomenti = parser.parse_args(argomenti)

    with np.load(argomenti.finestre) as finestre:
        id_clip = finestre["id_clip"]
        inizi = finestre["inizio_finestra"]
        etichette = finestre["etichetta"]
    indici_val = np.load(argomenti.split)["val"]
    percorso_per_nome = percorsi_degli_estratti(argomenti.estratti)
    confidenze = {}
    for vicinanza in VICINANZE_DA_PROVARE:
        confidenze[vicinanza] = []
    # le finestre di una clip sono una dopo l'altra: si tiene in memoria una timeline alla volta
    nome_in_memoria = None
    timeline = None
    for indice in indici_val:
        nome = nome_file_estratto(str(id_clip[indice]))
        if nome != nome_in_memoria:
            timeline = carica_timeline(percorso_per_nome[nome])
            nome_in_memoria = nome
        for vicinanza in VICINANZE_DA_PROVARE:
            confidenze[vicinanza].append(confidenza_arma_per_frame(timeline, int(inizi[indice]), vicinanza))
    arma_vera = etichette[indici_val] == INDICE_CLASSE["violenza_con_arma"]
    for vicinanza in confidenze:
        confidenze[vicinanza] = np.stack(confidenze[vicinanza])
    risultato = tara_warning_arma(confidenze, arma_vera)
    Path(argomenti.uscita).write_text(json.dumps(risultato, indent=1, ensure_ascii=False))
    print("migliore:", risultato["migliore"])


# ==============================================================================================================
# valuta_sistema
# Il sistema intero sul test: classificatore + WARNING ARMA tarato + regola D6 ("rissa riconosciuta + warning arma
# = rissa con arma", come chiesto da Marco il 19/10). Confronta le metriche del solo modello con quelle del sistema.
#
# Uso:
#     python -m sicurezza.demo valuta-sistema --finestre finestre.npz --split split.npz --probabilita probabilita_test.npy
#         --taratura taratura_warning.json --estratti estratti/ --uscita sistema.json
# ==============================================================================================================

def valuta_sistema(probabilita: np.ndarray, etichette: np.ndarray, warning: np.ndarray) -> dict:
    predette_sistema = []
    for indice in range(len(probabilita)):
        classe = classe_finale(probabilita[indice], warning[indice])
        predette_sistema.append(INDICE_CLASSE[classe])
    predette_sistema = np.array(predette_sistema)
    con_arma = etichette == INDICE_CLASSE["violenza_con_arma"]
    if con_arma.any():
        warning_con_arma = round(float(warning[con_arma].mean()), 4)
    else:
        warning_con_arma = None
    return {"solo_modello": metriche(etichette, probabilita.argmax(1)),
            "modello_piu_regola": metriche(etichette, predette_sistema),
            "finestre_con_warning": int(warning.sum()), "finestre": len(etichette),
            "warning_su_finestre_con_arma": warning_con_arma,
            "warning_su_finestre_senza_arma": round(float(warning[~con_arma].mean()), 4)}


def comando_valuta_sistema(argomenti=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finestre", required=True)
    parser.add_argument("--split", required=True, help="split.npz dell'addestramento (indici 'test')")
    parser.add_argument("--probabilita", required=True, help="probabilita_test.npy dello stesso addestramento")
    parser.add_argument("--taratura", required=True, help="taratura_warning.json (combinazione 'migliore')")
    parser.add_argument("--estratti", nargs="+", required=True)
    parser.add_argument("--uscita", required=True)
    argomenti = parser.parse_args(argomenti)

    regola = json.loads(Path(argomenti.taratura).read_text())["migliore"]
    soglia_confidenza = regola.get("soglia_confidenza", SOGLIA_CONFIDENZA_ARMA)
    vicinanza = regola.get("vicinanza", VICINANZA_PREDEFINITA)
    with np.load(argomenti.finestre) as finestre:
        id_clip = finestre["id_clip"]
        inizi = finestre["inizio_finestra"]
        etichette = finestre["etichetta"]
    indici_test = np.load(argomenti.split)["test"]
    percorso_per_nome = percorsi_degli_estratti(argomenti.estratti)
    warning = []
    # le finestre di una clip sono una dopo l'altra: si tiene in memoria una timeline alla volta
    nome_in_memoria = None
    timeline = None
    for indice in indici_test:
        nome = nome_file_estratto(str(id_clip[indice]))
        if nome != nome_in_memoria:
            timeline = carica_timeline(percorso_per_nome[nome])
            nome_in_memoria = nome
        warning.append(warning_arma(timeline, int(inizi[indice]), regola["frame_minimi"], soglia_confidenza,
                                    vicinanza))
    risultato = valuta_sistema(np.load(argomenti.probabilita), etichette[indici_test], np.array(warning, bool))
    Path(argomenti.uscita).write_text(json.dumps(risultato, indent=1, ensure_ascii=False))
    for nome in ("solo_modello", "modello_piu_regola"):
        print(f"{nome:20s} macro-F1 {risultato[nome]['macro_f1']:.3f}  recall {risultato[nome]['recall']}")


# ==============================================================================================================
# Allarme di violenza per video (DA DECIDERE 23): media mobile delle probabilità sulle ultime K finestre della stessa
# clip (si può fare in diretta) e allarme quando almeno N finestre di fila sono violenza o violenza con arma.
# Un video è "violento" se ha almeno una finestra etichettata violenza o violenza con arma (indipendente da N).
# N e K si scelgono sul validation (massima accuratezza per video, a parità meno falsi allarmi) e si misurano sul test.
#
# Uso (le probabilità di più file vengono mediate, DA DECIDERE 22 d):
#     python -m sicurezza.demo valuta-allarme --finestre-val finestre_val.npz --probabilita-val a_val.npy b_val.npy
#         --finestre-test finestre_test.npz --probabilita-test a_test.npy b_test.npy --uscita allarme.json
# ==============================================================================================================
CLASSI_VIOLENTE = ("violenza", "violenza_con_arma")
FINESTRE_DI_FILA_DA_PROVARE = tuple(range(1, 9))
MEDIE_MOBILI_DA_PROVARE = (1, 3, 5)


def media_mobile(probabilita: np.ndarray, id_clip: np.ndarray, inizi: np.ndarray, finestre: int) -> np.ndarray:
    """Per ogni finestra, la media delle probabilità sue e delle (finestre - 1) precedenti della stessa clip."""
    uscita = np.asarray(probabilita, np.float64).copy()
    if finestre <= 1:
        return uscita
    ordine = np.lexsort((inizi, id_clip))
    _, primi = np.unique(np.asarray(id_clip)[ordine], return_index=True)
    for righe in np.split(ordine, np.sort(primi)[1:]):
        somme = np.vstack([np.zeros((1, uscita.shape[1])), np.cumsum(probabilita[righe], axis=0)])
        fine = np.arange(1, len(righe) + 1)
        inizio = np.maximum(0, fine - finestre)
        uscita[righe] = (somme[fine] - somme[inizio]) / (fine - inizio)[:, None]
    return uscita


def pezzi_di_video(id_clip: np.ndarray, inizi: np.ndarray, secondi) -> np.ndarray:
    """I video lunghi contano come tanti pezzi da `secondi`: un video lungo con una rissa da qualche parte farebbe
    scattare l'allarme quasi sempre, e "per video" gonfierebbe l'accuratezza (CCTV-Fights: tutti i video hanno una rissa)."""
    if not secondi:
        return np.asarray(id_clip)
    # "clip" diventa "clip#0", "clip#1", ...: il numero del pezzo da `secondi` in cui comincia la finestra
    frame_per_pezzo = int(round(secondi * FPS_ANALISI))
    pezzi = []
    for clip, inizio in zip(np.asarray(id_clip).tolist(), np.asarray(inizi).tolist()):
        pezzi.append(f"{clip}#{inizio // frame_per_pezzo}")
    return np.array(pezzi)


def finestre_di_fila_massime(valori: np.ndarray) -> int:
    """La serie più lunga di valori veri consecutivi."""
    massimo = 0
    corrente = 0
    for valore in valori:
        if valore:
            corrente += 1
        else:
            corrente = 0
        massimo = max(massimo, corrente)
    return massimo


def quota(maschera: np.ndarray, valori: np.ndarray):
    """Frazione di valori veri tra quelli scelti dalla maschera (None se la maschera non sceglie niente)."""
    if not maschera.any():
        return None
    return round(float(valori[maschera].mean()), 4)


def allarmi_per_video(probabilita: np.ndarray, etichette: np.ndarray, id_clip: np.ndarray, inizi: np.ndarray,
                      finestre_di_fila: int, media: int = 1) -> dict:
    indici_violenti = [CLASSI.index("violenza"), CLASSI.index("violenza_con_arma")]
    predette = np.isin(media_mobile(probabilita, id_clip, inizi, media).argmax(1), indici_violenti)
    vere = np.isin(etichette, indici_violenti)
    video_veri = []
    video_allarme = []
    for clip in np.unique(id_clip):
        righe = np.where(id_clip == clip)[0]
        righe = righe[np.argsort(inizi[righe], kind="stable")]   # finestre della clip in ordine di tempo
        video_veri.append(vere[righe].any())
        video_allarme.append(finestre_di_fila_massime(predette[righe]) >= finestre_di_fila)
    video_veri = np.array(video_veri)
    video_allarme = np.array(video_allarme)
    return {"finestre_di_fila": finestre_di_fila, "media_mobile": media, "video": len(video_veri),
            "video_violenti": int(video_veri.sum()),
            "accuratezza": round(float((video_veri == video_allarme).mean()), 4),
            "recall": quota(video_veri, video_allarme), "falsi_allarmi": quota(~video_veri, video_allarme),
            "precisione": quota(video_allarme, video_veri)}


def tara_allarme(probabilita: np.ndarray, etichette: np.ndarray, id_clip: np.ndarray, inizi: np.ndarray) -> dict:
    tabella = []
    for media in MEDIE_MOBILI_DA_PROVARE:
        for di_fila in FINESTRE_DI_FILA_DA_PROVARE:
            tabella.append(allarmi_per_video(probabilita, etichette, id_clip, inizi, di_fila, media))

    # la migliore: accuratezza più alta; a parità, meno falsi allarmi; a parità ancora, la prima della tabella
    def punteggio(riga):
        falsi_allarmi = riga["falsi_allarmi"] or 0
        return (riga["accuratezza"], -falsi_allarmi)
    migliore = tabella[0]
    for riga in tabella:
        if punteggio(riga) > punteggio(migliore):
            migliore = riga
    return {"migliore": migliore, "tabella": tabella}


def comando_valuta_allarme(argomenti=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finestre-val", required=True, help="finestre_val.npz di `addestra` (etichetta, id_clip, ...)")
    parser.add_argument("--probabilita-val", required=True, nargs="+", help="uno o più .npy: si fa la media")
    parser.add_argument("--finestre-test", required=True)
    parser.add_argument("--probabilita-test", required=True, nargs="+")
    parser.add_argument("--secondi-pezzo", type=float, default=None,
                        help="misura per pezzi di video di questa durata invece che per video interi (es. 10)")
    parser.add_argument("--uscita", required=True)
    argomenti = parser.parse_args(argomenti)
    val = carica_insieme(argomenti.finestre_val, argomenti.probabilita_val, argomenti.secondi_pezzo)
    test = carica_insieme(argomenti.finestre_test, argomenti.probabilita_test, argomenti.secondi_pezzo)
    taratura = tara_allarme(val["probabilita"], val["etichette"], val["pezzi"], val["inizi"])
    scelta = taratura["migliore"]
    risultato_test = allarmi_per_video(test["probabilita"], test["etichette"], test["pezzi"], test["inizi"],
                                       scelta["finestre_di_fila"], scelta["media_mobile"])
    risultato_prima_finestra = allarmi_per_video(test["probabilita"], test["etichette"], test["pezzi"],
                                                 test["inizi"], 1, 1)
    risultato = {"secondi_pezzo": argomenti.secondi_pezzo, "scelta_sul_val": scelta, "tabella_val": taratura["tabella"],
                 "test": risultato_test, "test_allarme_alla_prima_finestra": risultato_prima_finestra}
    Path(argomenti.uscita).write_text(json.dumps(risultato, indent=1, ensure_ascii=False))
    print("scelta sul validation:", scelta)
    print("test:", risultato["test"])


# ==============================================================================================================
# pannello
# Argo dal vivo (D56, D57): un pannello web locale collegato alle telecamere. Per ogni telecamera un ffmpeg registra
# il flusso a segmenti da 10 s a 25 fps ("videoregistratore", in sessioni/<telecamera>/<data>/) e un thread li
# analizza in ordine con `AnalisiFlusso`, il sistema intero. Sulla CPU del portatile l'analisi va più lenta del
# tempo reale e resta indietro (in differita, il ritardo è nel pannello); con una GPU tiene il passo. I frame non
# stanno in memoria: aspettano su disco, e la registrazione resta come archivio.
#
# Uscite per sessione: video_annotato.mp4, eventi.jsonl (una riga per finestra, come eventi.json), keypoint.jsonl
# (persone con box e 17 keypoint, armi), e alla fine eventi.json + probabilita.csv come `analizza-video`.
# Pagina: src/sicurezza/pannello/ (telecamere, registro degli eventi, archivio delle sessioni).
#
# Uso:
#     uv run argo scarica-pesi
#     uv run argo pannello --telecamera ingresso=rtsp://utente:password@192.168.1.20:554/stream1 --telecamera webcam=/dev/video0
#     uv run argo pannello --telecamera prova=video.mp4 --come-telecamera      (un file letto a velocità reale)
#     poi apri http://127.0.0.1:8765
# ==============================================================================================================
def carica_insieme(percorso_finestre: str, percorsi_probabilita: list, secondi_pezzo) -> dict:
    """Le finestre di uno split e la media delle probabilità di uno o più modelli."""
    probabilita = []
    for percorso in percorsi_probabilita:
        probabilita.append(np.load(percorso))
    with np.load(percorso_finestre) as finestre:
        return {"probabilita": np.mean(probabilita, axis=0), "etichette": finestre["etichetta"],
                "pezzi": pezzi_di_video(finestre["id_clip"], finestre["inizio_finestra"], secondi_pezzo),
                "inizi": finestre["inizio_finestra"]}


CARTELLA_PAGINA = Path(__file__).parent / "pannello"
FILE_PAGINA = {"/": ("index.html", "text/html; charset=utf-8"), "/pannello.css": ("pannello.css", "text/css"),
               "/pannello.js": ("pannello.js", "text/javascript"), "/logo.svg": ("logo.svg", "image/svg+xml")}
FILE_ARCHIVIO = {"video_annotato.mp4": "video/mp4", "eventi.json": "application/json", "probabilita.csv": "text/csv"}
SECONDI_SEGMENTO = 10
# I pesi del sistema consegnato (D60): allegati alla Release del repo, si scaricano con `argo scarica-pesi`.
# D70: Release v1.1 = quelli della v1.0 + le reti video b1 (scippi) e b2 (armi).
FILE_PESI = {"classificatore": "stgcn.pt", "modello_armi": "armi_v3.pt", "rete_pixel": "rete_violenza.pt",
             "testa_pixel": "testa_violenza.npz", "taratura": "taratura_warning.json",
             "rete_scippi": "rete_scippi_b1.pt", "rete_armi": "rete_armi_b2.pt"}
URL_PESI = "https://github.com/marcozambrella/argo/releases/download/v1.1/"


def ingresso_ffmpeg(sorgente: str, come_telecamera: bool = False) -> list[str]:
    """Opzioni d'ingresso di ffmpeg per una telecamera: RTSP, http(s) (MJPEG/HLS, anche un telefono usato come
    telecamera IP), webcam (/dev/videoN) o un file (con come_telecamera letto a velocità reale)."""
    if sorgente.startswith("rtsp://"):
        return ["-rtsp_transport", "tcp", "-i", sorgente]
    if sorgente.startswith("/dev/video"):
        return ["-f", "v4l2", "-i", sorgente]
    if re.match(r"^[a-z][a-z0-9+.-]*://", sorgente):
        return ["-i", sorgente]
    if come_telecamera:
        return ["-re", "-i", sorgente]   # -re: legge il file alla sua velocità reale
    return ["-i", sorgente]


def sorgente_senza_password(sorgente: str) -> str:
    """rtsp://utente:password@host/… → rtsp://utente:***@host/… (nel pannello e nei log non deve comparire)."""
    return re.sub(r"(://[^:/@]+):[^@/]*@", r"\1:***@", sorgente)


def opzioni_dai_pesi(cartella: Path) -> dict:
    """Le opzioni di `AnalisiFlusso` per i pesi presenti nella cartella. Senza classificatore: modalità scheletri."""
    presenti = {}
    for chiave, nome in FILE_PESI.items():
        if (cartella / nome).exists():
            presenti[chiave] = cartella / nome
    opzioni = {}
    if "classificatore" in presenti:
        opzioni["percorso_classificatore"] = str(presenti["classificatore"])
        if "rete_pixel" in presenti and "testa_pixel" in presenti:
            opzioni["percorso_rete_pixel"] = str(presenti["rete_pixel"])
            opzioni["percorso_testa_pixel"] = str(presenti["testa_pixel"])
        if "rete_scippi" in presenti:
            opzioni["percorso_rete_scippi"] = str(presenti["rete_scippi"])
        if "rete_armi" in presenti:
            opzioni["percorso_rete_armi"] = str(presenti["rete_armi"])
    if "modello_armi" in presenti:
        opzioni["percorso_modello_armi"] = str(presenti["modello_armi"])
    if "taratura" in presenti:
        opzioni["taratura"] = json.loads(presenti["taratura"].read_text())["migliore"]
    return opzioni


def descrivi_modalita(opzioni: dict) -> str:
    if "percorso_classificatore" not in opzioni:
        return "skeletons only"
    if "percorso_rete_pixel" in opzioni:
        return "full system"
    return "skeletons + ST-GCN"


class Videoregistratore:
    """ffmpeg che registra una telecamera a segmenti chiusi da `secondi_segmento` (D57)."""
    def __init__(self, sorgente: str, cartella: Path, come_telecamera: bool = False,
                 secondi_segmento: int = SECONDI_SEGMENTO):
        (cartella / "segmenti").mkdir(parents=True, exist_ok=True)
        self.elenco = cartella / "segmenti.csv"
        self.cartella_segmenti = cartella / "segmenti"
        self.inizio = time.time()
        comando = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
        comando += ingresso_ffmpeg(sorgente, come_telecamera)
        comando += ["-an", "-vf", f"fps={FPS_ANALISI},scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-force_key_frames", "expr:gte(t,n_forced*1)",
                    "-f", "segment", "-segment_time", str(secondi_segmento), "-reset_timestamps", "1",
                    "-segment_list", str(self.elenco), "-segment_list_type", "csv",
                    str(self.cartella_segmenti / "%05d.mp4")]
        with open(cartella / "ffmpeg.log", "w") as log:
            self.processo = subprocess.Popen(comando, stderr=log)

    def segmenti_chiusi(self) -> list:
        """I segmenti finiti, in ordine: ffmpeg aggiunge una riga all'elenco solo quando chiude un segmento."""
        if not self.elenco.exists():
            return []
        segmenti = []
        for riga in self.elenco.read_text().splitlines():
            if riga.strip():
                nome_file = Path(riga.split(",")[0]).name   # la riga è "00001.mp4,inizio,fine"
                segmenti.append(self.cartella_segmenti / nome_file)
        return segmenti

    def finito(self) -> bool:
        return self.processo.poll() is not None

    def ferma(self) -> None:
        if not self.finito():
            self.processo.send_signal(signal.SIGINT)   # come Ctrl+C: ffmpeg chiude bene l'ultimo segmento
            try:
                self.processo.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.processo.kill()


class Telecamera:
    def __init__(self, nome: str, sorgente: str, cartella_sessioni: Path, opzioni_analisi: dict,
                 come_telecamera: bool = False, al_evento=None):
        self.nome = nome
        self.cartella = Path(cartella_sessioni) / nome / time.strftime("%Y%m%d-%H%M%S")
        self.cartella.mkdir(parents=True, exist_ok=True)
        self.opzioni_analisi = opzioni_analisi
        self.al_evento = al_evento   # funzione chiamata a ogni evento (il pannello la usa per il registro)
        # un file senza --come-telecamera si analizza tutto e subito: non è "dal vivo" e il ritardo non ha senso
        self.dal_vivo = come_telecamera or bool(re.match(r"^(/dev/video|[a-z][a-z0-9+.-]*://)", sorgente))
        self.stato = {"nome": nome, "sorgente": sorgente_senza_password(sorgente), "dal_vivo": self.dal_vivo,
                      "fase": "in avvio", "ritardo_secondi": None, "frame_al_secondo": None,
                      "secondi_analizzati": 0.0, "ultimo_evento": None, "errore": None}
        self.jpeg = None
        self.nuovo_frame = threading.Condition()
        self.fermati = threading.Event()
        self.registratore = Videoregistratore(sorgente, self.cartella, come_telecamera)
        self.thread = threading.Thread(target=self.lavora, daemon=True, name=f"argo-{nome}")
        self.thread.start()

    def lavora(self) -> None:
        try:
            self.analizza_segmenti()
            self.stato["fase"] = "finita"
        except Exception as errore:   # una telecamera che si rompe non deve fermare le altre
            self.stato["fase"] = "errore"
            self.stato["errore"] = f"{type(errore).__name__}: {errore}"
        with self.nuovo_frame:
            self.nuovo_frame.notify_all()

    def segmento_successivo(self, fatti: int):
        """Aspetta il prossimo segmento chiuso; None quando la registrazione è finita e non ne restano."""
        while not self.fermati.is_set():
            segmenti = self.registratore.segmenti_chiusi()
            if len(segmenti) > fatti:
                return segmenti[fatti]
            if self.registratore.finito():
                segmenti = self.registratore.segmenti_chiusi()    # l'ultimo si chiude quando ffmpeg esce
                if len(segmenti) > fatti:
                    return segmenti[fatti]
                if fatti == 0:
                    log = (self.cartella / "ffmpeg.log").read_text().strip()
                    if log:
                        motivo = log.splitlines()[-1]
                    else:
                        motivo = "ffmpeg exited"
                    raise RuntimeError(f"no video from the source ({motivo})")
                return None
            time.sleep(0.3)
        return None

    def analizza_segmenti(self) -> None:
        # **: ogni chiave del dizionario diventa l'argomento con lo stesso nome
        analisi = AnalisiFlusso(**self.opzioni_analisi, pannello_nel_frame=False)
        self.stato["fase"] = "in attesa del primo segmento"
        scrittore = None
        eventi = []
        fatti = 0           # segmenti già analizzati
        indice_frame = 0
        velocita = None     # secondi per frame, media mobile
        with open(self.cartella / "keypoint.jsonl", "w") as file_keypoint, \
                open(self.cartella / "eventi.jsonl", "w") as file_eventi:
            while True:
                segmento = self.segmento_successivo(fatti)
                if segmento is None:
                    break
                fatti += 1
                self.stato["fase"] = "in analisi"
                for frame in leggi_frame(str(segmento)):
                    if self.fermati.is_set():
                        break
                    inizio = time.perf_counter()
                    if scrittore is None:
                        scrittore = ScrittoreVideo(self.cartella / "video_annotato.mp4", frame.shape[1], frame.shape[0])
                    annotato, persone, armi, evento = analisi.analizza(frame)
                    scrittore.scrivi(annotato)
                    file_keypoint.write(json.dumps(riga_keypoint(indice_frame, persone, armi)) + "\n")
                    if evento:
                        eventi.append(evento)
                        file_eventi.write(json.dumps(evento, ensure_ascii=False) + "\n")
                        file_eventi.flush()
                        self.stato["ultimo_evento"] = evento
                        if self.al_evento is not None:
                            self.al_evento(self.nome, evento)
                    indice_frame += 1

                    durata = time.perf_counter() - inizio
                    if velocita is None:
                        velocita = durata
                    else:
                        velocita = 0.9 * velocita + 0.1 * durata
                    self.stato["frame_al_secondo"] = round(1 / max(velocita, 1e-6), 1)
                    self.stato["secondi_analizzati"] = round(indice_frame / FPS_ANALISI, 1)
                    if self.dal_vivo:
                        # ritardo = adesso - momento in cui la telecamera ha ripreso questo frame
                        ripreso_alle = self.registratore.inizio + indice_frame / FPS_ANALISI
                        self.stato["ritardo_secondi"] = round(max(0.0, time.time() - ripreso_alle), 1)
                    jpeg = cv2.imencode(".jpg", annotato, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                    with self.nuovo_frame:
                        self.jpeg = jpeg
                        self.nuovo_frame.notify_all()
        if scrittore:
            scrittore.chiudi()
        salva_eventi(self.cartella, eventi)

    def ferma(self) -> None:
        self.registratore.ferma()
        self.fermati.set()
        self.thread.join(timeout=30)


def data_e_nome(percorso: Path):
    """Chiave per ordinare le sessioni: data di modifica, poi nome."""
    return (percorso.stat().st_mtime, str(percorso))


def sessioni_in_archivio(cartelle: list) -> list:
    """Le cartelle con un'analisi finita (video_annotato.mp4 + eventi.json): sessioni del pannello e `analizza-video`.
    Le più recenti prima."""
    trovate = set()
    for cartella in cartelle:
        if not Path(cartella).exists():
            continue
        for file in Path(cartella).rglob("eventi.json"):
            if (file.parent / "video_annotato.mp4").exists():
                trovate.add(file.parent)
    return sorted(trovate, key=data_e_nome, reverse=True)


def riassunto_sessione(numero: int, cartella: Path, radici: list) -> dict:
    eventi = json.loads((cartella / "eventi.json").read_text())
    # la radice dell'archivio che contiene questa sessione: il nome si mostra a partire da lì
    radice = cartella.parent
    for possibile_radice in radici:
        if Path(cartella).is_relative_to(Path(possibile_radice)):
            radice = Path(possibile_radice)
            break
    if cartella != radice:
        nome = str(cartella.relative_to(radice))
    else:
        nome = cartella.name
    durata = 0.0
    if len(eventi) > 0:
        durata = eventi[-1]["fine_secondi"]
    finestre_con_allarme = 0
    finestre_con_warning = 0
    for evento in eventi:
        if evento["allarme_violenza"]:
            finestre_con_allarme += 1
        if evento["warning_arma"]:
            finestre_con_warning += 1
    return {"numero": numero, "nome": nome, "durata_secondi": durata,
            "finestre_con_allarme": finestre_con_allarme, "finestre_con_warning": finestre_con_warning}


class Pannello:
    def __init__(self, telecamere: list, cartella_sessioni: Path, cartelle_archivio: list,
                 opzioni_analisi: dict, come_telecamera: bool = False):
        """telecamere: lista di (nome, sorgente)."""
        self.cartella_sessioni = Path(cartella_sessioni)
        self.cartelle_archivio = [self.cartella_sessioni]
        for cartella in cartelle_archivio:
            self.cartelle_archivio.append(Path(cartella))
        self.modalita = descrivi_modalita(opzioni_analisi)
        self.eventi = []
        self.blocco = threading.Lock()   # le telecamere aggiungono eventi da thread diversi
        self.telecamere = {}
        for nome, sorgente in telecamere:
            self.telecamere[nome] = Telecamera(nome, sorgente, self.cartella_sessioni, opzioni_analisi,
                                               come_telecamera, self.aggiungi_evento)

    def aggiungi_evento(self, nome: str, evento: dict) -> None:
        with self.blocco:
            evento_del_registro = {"numero": len(self.eventi), "telecamera": nome, "ora": time.strftime("%H:%M:%S")}
            evento_del_registro.update(evento)
            self.eventi.append(evento_del_registro)

    def eventi_dopo(self, numero: int) -> list:
        with self.blocco:
            return self.eventi[max(numero, 0):]

    def stato(self) -> dict:
        stati = []
        for telecamera in self.telecamere.values():
            stati.append(dict(telecamera.stato))
        return {"modalita": self.modalita, "telecamere": stati}

    def archivio(self) -> list:
        return sessioni_in_archivio(self.cartelle_archivio)

    def ferma(self) -> None:
        for telecamera in self.telecamere.values():
            telecamera.ferma()


class GestorePannello(BaseHTTPRequestHandler):
    pannello: Pannello = None   # assegnato da `crea_server`

    def log_message(self, formato, *argomenti) -> None:   # niente righe a ogni richiesta (MJPEG e interrogazioni)
        pass

    def do_GET(self) -> None:
        indirizzo = urllib.parse.urlparse(self.path)
        parametri = urllib.parse.parse_qs(indirizzo.query)
        try:
            if indirizzo.path in FILE_PAGINA:
                nome, tipo = FILE_PAGINA[indirizzo.path]
                self.invia_file(CARTELLA_PAGINA / nome, tipo)
            elif indirizzo.path == "/api/stato":
                self.invia_json(self.pannello.stato())
            elif indirizzo.path == "/api/eventi":
                self.invia_json(self.pannello.eventi_dopo(int(parametri.get("dopo", ["0"])[0])))
            elif indirizzo.path == "/api/archivio":
                riassunti = []
                for numero, cartella in enumerate(self.pannello.archivio()):
                    riassunti.append(riassunto_sessione(numero, cartella, self.pannello.cartelle_archivio))
                self.invia_json(riassunti)
            elif indirizzo.path.startswith("/flusso/"):
                self.invia_flusso(indirizzo.path)
            elif indirizzo.path.startswith("/archivio/"):
                self.invia_file_archivio(indirizzo.path)
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass   # il browser ha chiuso la connessione

    def invia_flusso(self, percorso: str) -> None:
        """/flusso/<telecamera>.mjpg: l'immagine annotata dal vivo."""
        trovato = re.fullmatch(r"/flusso/(.+)\.mjpg", percorso)
        if trovato is None:
            self.send_error(404)
            return
        nome = urllib.parse.unquote(trovato[1])
        if nome not in self.pannello.telecamere:
            self.send_error(404)
            return
        self.invia_mjpeg(self.pannello.telecamere[nome])

    def invia_file_archivio(self, percorso: str) -> None:
        """/archivio/<numero della sessione>/<file>: solo numeri e nomi noti, nessun percorso arriva dal browser."""
        trovato = re.fullmatch(r"/archivio/(\d+)/([a-z0-9_.]+)", percorso)
        if trovato is None or trovato[2] not in FILE_ARCHIVIO:
            self.send_error(404)
            return
        numero = int(trovato[1])
        nome_file = trovato[2]
        sessioni = self.pannello.archivio()
        if numero >= len(sessioni):
            self.send_error(404)
            return
        self.invia_file(sessioni[numero] / nome_file, FILE_ARCHIVIO[nome_file])

    def invia_json(self, dati) -> None:
        corpo = json.dumps(dati, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(corpo)

    def invia_file(self, percorso: Path, tipo: str) -> None:
        """Con le richieste Range (206): il <video> dell'archivio può saltare avanti e indietro."""
        if not percorso.is_file():
            self.send_error(404)
            return
        dimensione = percorso.stat().st_size
        inizio = 0
        fine = dimensione - 1
        # Range: "bytes=100-199" (dal byte 100 al 199), "bytes=100-" (dal 100 alla fine), "bytes=-500" (ultimi 500)
        intervallo = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
        if intervallo and (intervallo[1] or intervallo[2]):
            if intervallo[1]:
                inizio = int(intervallo[1])
                if intervallo[2]:
                    fine = min(int(intervallo[2]), fine)
            else:
                inizio = max(0, dimensione - int(intervallo[2]))
            if inizio > fine:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{dimensione}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {inizio}-{fine}/{dimensione}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", tipo)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(fine - inizio + 1))
        self.end_headers()
        with open(percorso, "rb") as file:
            file.seek(inizio)
            restanti = fine - inizio + 1
            while restanti > 0:
                pezzo = file.read(min(restanti, 65536))   # a pezzi da 64 KB
                if not pezzo:
                    break
                self.wfile.write(pezzo)
                restanti -= len(pezzo)

    def invia_mjpeg(self, telecamera: Telecamera) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=argo")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        ultimo = None
        while True:
            with telecamera.nuovo_frame:
                # aspetta un frame nuovo (o la fine della telecamera), al massimo 5 s
                finita = telecamera.stato["fase"] in ("finita", "errore")
                if telecamera.jpeg is ultimo and not finita:
                    telecamera.nuovo_frame.wait(timeout=5)
                jpeg = telecamera.jpeg
            if jpeg is not None and jpeg is not ultimo:
                self.wfile.write(b"--argo\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                ultimo = jpeg
            elif telecamera.stato["fase"] in ("finita", "errore"):
                return


def crea_server(pannello: Pannello, host: str = "127.0.0.1", porta: int = 8765) -> ThreadingHTTPServer:
    # una sottoclasse per server: così ogni server ha il suo pannello (nei test ne girano più di uno)
    class GestoreDiQuestoPannello(GestorePannello):
        pass
    GestoreDiQuestoPannello.pannello = pannello
    server = ThreadingHTTPServer((host, porta), GestoreDiQuestoPannello)
    server.daemon_threads = True
    return server


def leggi_telecamera(testo: str, numero: int):
    """"ingresso=rtsp://…" → ("ingresso", "rtsp://…"); senza nome: "cam1", "cam2", …"""
    nome, uguale, sorgente = testo.partition("=")
    if uguale and re.fullmatch(r"[\w-]+", nome):
        return nome, sorgente
    return f"cam{numero}", testo


def comando_pannello(argomenti=None) -> None:
    parser = argparse.ArgumentParser(description="ARGO: web panel for the cameras")
    parser.add_argument("--telecamera", action="append", default=[], metavar="NAME=SOURCE",
                        help="rtsp://…, http(s)://…, /dev/video0 or a file; repeat for more cameras")
    parser.add_argument("--come-telecamera", action="store_true",
                        help="read files at real speed, like a camera (for testing)")
    parser.add_argument("--pesi", default="pesi", help="weights folder (`argo scarica-pesi`)")
    parser.add_argument("--sessioni", default="sessioni", help="where to store camera recordings and analyses")
    parser.add_argument("--archivio", nargs="*", default=[],
                        help="more folders with analyses to review (e.g. runs/demo_finale)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--porta", type=int, default=8765)
    if torch.cuda.is_available():
        dispositivo_predefinito = "cuda"
    else:
        dispositivo_predefinito = "cpu"
    parser.add_argument("--dispositivo", default=dispositivo_predefinito)
    parser.add_argument("--modello-pose", default=MODELLO_POSE_PREDEFINITO)
    parser.add_argument("--tracker", default=TRACKER_PREDEFINITO)
    argomenti = parser.parse_args(argomenti)
    opzioni = opzioni_dai_pesi(Path(argomenti.pesi))
    opzioni["percorso_modello_pose"] = argomenti.modello_pose
    opzioni["tracker"] = argomenti.tracker
    opzioni["dispositivo"] = argomenti.dispositivo
    if "percorso_classificatore" not in opzioni:
        print(f"No weights in {argomenti.pesi}/: skeletons-only mode. For the full system: uv run argo scarica-pesi")
    if argomenti.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: the panel is reachable from other computers on {argomenti.host} (camera images, no password)")
    telecamere = []
    for numero, testo in enumerate(argomenti.telecamera, start=1):
        telecamere.append(leggi_telecamera(testo, numero))
    pannello = Pannello(telecamere, Path(argomenti.sessioni), argomenti.archivio, opzioni, argomenti.come_telecamera)
    server = crea_server(pannello, argomenti.host, argomenti.porta)
    print(f"ARGO ({pannello.modalita}, {argomenti.dispositivo}): http://{argomenti.host}:{argomenti.porta}"
          "  — Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the cameras and saving the sessions…")
    finally:
        server.server_close()
        pannello.ferma()


def comando_scarica_pesi(argomenti=None) -> None:
    from .strumenti import scarica_con_riprova
    parser = argparse.ArgumentParser(description="Download the trained weights from the repo Release")
    parser.add_argument("--cartella", default="pesi")
    parser.add_argument("--url", default=URL_PESI)
    argomenti = parser.parse_args(argomenti)
    cartella = Path(argomenti.cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    for nome in FILE_PESI.values():
        if (cartella / nome).exists():
            print(f"{nome}: already here")
            continue
        print(f"{nome}: downloading…", flush=True)
        parziale = cartella / (nome + ".parziale")
        scarica_con_riprova(argomenti.url + nome, parziale)
        parziale.rename(cartella / nome)
    print(f"Weights in {cartella}/ (research only, non-commercial use: see the README)")


# ==============================================================================================================
# riga di comando: python -m sicurezza.demo <comando> [opzioni]
# ==============================================================================================================
COMANDI = {"pannello": comando_pannello, "scarica-pesi": comando_scarica_pesi,
           "analizza-video": comando_analizza_video, "tara-warning": comando_tara_warning,
           "valuta-sistema": comando_valuta_sistema, "valuta-allarme": comando_valuta_allarme}


def main(argomenti=None) -> None:
    argomenti = sys.argv[1:] if argomenti is None else argomenti
    if not argomenti or argomenti[0] not in COMANDI:
        print("usage: argo {pannello,scarica-pesi,analizza-video} [options]  (--help after the command)")
        raise SystemExit(2)
    COMANDI[argomenti[0]](argomenti[1:])


if __name__ == "__main__":
    main()
