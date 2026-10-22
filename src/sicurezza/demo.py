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

from .dati import CLASSI, FPS_ANALISI, FRAME_PER_FINESTRA, OSSA, PASSO_FINESTRE, POLSI, leggi_frame, leggi_info_video
from .modelli import TimelineVideo, costruisci_finestra, crea_modello, timeline_da_dati
from .visione import Estrattore, MODELLO_POSE_PREDEFINITO, TRACKER_PREDEFINITO, impacchetta_estrazione


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
    centro_x, centro_y = (box_arma[0] + box_arma[2]) / 2, (box_arma[1] + box_arma[3]) / 2
    larghezza = box_persone[:, 2] - box_persone[:, 0]
    altezza = box_persone[:, 3] - box_persone[:, 1]
    dentro = ((centro_x >= box_persone[:, 0] - margine * larghezza) & (centro_x <= box_persone[:, 2] + margine * larghezza)
              & (centro_y >= box_persone[:, 1] - margine * altezza) & (centro_y <= box_persone[:, 3] + margine * altezza))
    return bool(dentro.any())


# 19/10: la demo ha mostrato scarpe nere prese per pistole (vicine alla persona, ma ai piedi). Secondo criterio:
# "polso:d" = centro dell'arma entro d × altezza della persona da uno dei suoi polsi (un'arma vera è in mano).
# Quale criterio usare lo decide la taratura (D35) sul validation set, insieme a soglia e frame minimi.
# 19/10 (D44): "mani" = probabilità del classificatore sui ritagli intorno ai polsi; "polso:0.2+mani" = il più
# alto tra detector vicino al polso e mani.
VICINANZA_PREDEFINITA = f"persona:{MARGINE_VICINANZA}"
VICINANZE_DA_PROVARE = ("persona:0.25", "persona:0.5", "persona:1.0", "polso:0.1", "polso:0.2", "polso:0.35",
                        "mani", "polso:0.2+mani", "persona:0.5+mani")


def persone_nel_frame(timeline: TimelineVideo, frame: int) -> tuple[np.ndarray, np.ndarray]:
    """(box [n, 4], polsi [n, 2, 3]) delle persone tracciate presenti nel frame."""
    presenti = [id_traccia for id_traccia, box in timeline.box_per_traccia.items() if not np.isnan(box[frame, 0])]
    box = np.array([timeline.box_per_traccia[i][frame] for i in presenti], np.float32).reshape(-1, 4)
    polsi = np.array([timeline.keypoint_per_traccia[i][frame][list(POLSI)] for i in presenti], np.float32).reshape(-1, 2, 3)
    return box, polsi


def arma_vicina(box_arma: np.ndarray, box_persone: np.ndarray, polsi_persone: np.ndarray, vicinanza: str) -> bool:
    criterio, valore = vicinanza.split(":")
    if criterio == "persona":
        return arma_vicina_a_persona(box_arma, box_persone, float(valore))
    if len(box_persone) == 0:
        return False
    centro = np.array([(box_arma[0] + box_arma[2]) / 2, (box_arma[1] + box_arma[3]) / 2])
    distanze = np.linalg.norm(polsi_persone[:, :, :2] - centro, axis=-1)          # [persone, 2 polsi]
    distanze = np.where(np.isnan(distanze), np.inf, distanze).min(axis=1)
    return bool((distanze <= float(valore) * (box_persone[:, 3] - box_persone[:, 1])).any())


def frame_con_arma(timeline: TimelineVideo, inizio_finestra: int, frame_per_finestra: int = FRAME_PER_FINESTRA,
                   soglia_confidenza: float = SOGLIA_CONFIDENZA_ARMA, vicinanza: str = VICINANZA_PREDEFINITA) -> int:
    """Quanti frame della finestra hanno almeno un'arma sicura vicino a una persona tracciata."""
    confidenze = confidenza_arma_per_frame(timeline, inizio_finestra, vicinanza, frame_per_finestra)
    return int((confidenze >= soglia_confidenza).sum())


def warning_arma(timeline: TimelineVideo, inizio_finestra: int, frame_minimi: int = FRAME_MINIMI_CON_ARMA,
                 **opzioni) -> bool:
    return frame_con_arma(timeline, inizio_finestra, **opzioni) >= frame_minimi


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
TESTI_CLASSI = {"normale": "Normale", "violenza": "Violenza", "violenza_con_arma": "Violenza con arma",
                "scippo": "Scippo"}  # SEGNAPOSTO DESIGN
TESTO_WARNING_ARMA = "WARNING ARMA"     # SEGNAPOSTO DESIGN
TESTO_ALLARME_VIOLENZA = "ALLARME VIOLENZA"   # SEGNAPOSTO DESIGN
NOMI_ARMI = ("pistola", "coltello", "fucile")


def disegna_persone(frame: np.ndarray, persone: dict) -> None:
    stile = STILE_OVERLAY
    for id_traccia, box, keypoint in zip(persone["id_traccia"], persone["box"], persone["keypoint"]):
        visibile = keypoint[:, 2] >= stile["soglia_visibilita_keypoint"]
        for inizio, fine in OSSA:
            if visibile[inizio] and visibile[fine]:
                cv2.line(frame, tuple(map(int, keypoint[inizio, :2])), tuple(map(int, keypoint[fine, :2])),
                         stile["colore_scheletro"], stile["spessore"], cv2.LINE_AA)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), stile["colore_scheletro"], 1)
        cv2.putText(frame, f"ID {int(id_traccia)}", (x1, max(15, y1 - 5)), stile["font"], stile["scala_font"],
                    stile["colore_testo"], 1, cv2.LINE_AA)


def disegna_armi(frame: np.ndarray, armi: dict, soglia_confidenza: float) -> None:
    stile = STILE_OVERLAY
    for box, classe, confidenza in zip(armi["box"], armi["classe"], armi["confidenza"]):
        if confidenza < soglia_confidenza:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), stile["colore_arma"], stile["spessore"])
        cv2.putText(frame, f"{NOMI_ARMI[int(classe)]} {confidenza:.2f}", (x1, max(15, y1 - 5)), stile["font"],
                    stile["scala_font"], stile["colore_arma"], 1, cv2.LINE_AA)


def disegna_pannello(frame: np.ndarray, probabilita: np.ndarray | None, classe: str | None, warning: bool,
                     allarme: bool = False, attesa: str = "in attesa di 40 frame...") -> None:
    """Pannello in alto a sinistra: classe decisa, barre delle probabilità, WARNING ARMA, ALLARME VIOLENZA."""
    stile = STILE_OVERLAY
    larghezza_pannello, altezza_riga, inizio_barre = 330, 24, 160
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
    for riga, (nome_classe, valore) in enumerate(zip(CLASSI, probabilita), start=1):
        y = altezza_riga * (riga + 1)
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
#     python -m sicurezza.demo analizza-video --video ingresso.mp4 --classificatore runs/gt/migliore.pt         --modello-armi runs/armi/yolo26s_armi/weights/best.pt --uscita demo/
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
    ingressi = (torch.from_numpy(nodi)[None].to(dispositivo), torch.from_numpy(maschera)[None].to(dispositivo),
                torch.from_numpy(classe_arma.astype(np.int64))[None].to(dispositivo))
    return np.mean([torch.softmax(modello(*ingressi), -1)[0].cpu().numpy() for modello in classificatori], axis=0)


# Ramo pixel (D48), attivo con --testa-pixel: R(2+1)D-18 di torchvision preaddestrato su
# Kinetics-400 e CONGELATO → vettore da 512 → regressione logistica violenza sì/no (testa allenata sul train
# dell'add. 2, `pesi/testa_rgb.npz`). Stessa preparazione dei kernel `prova-rgb`: 16 frame dei 40, 171×128,
# ritaglio centrale 112×112. Fusione: media tra la probabilità dei pixel e quella di violenza degli scheletri; sulle
# clip intere mai viste 0,82 per pezzi da 30 s e 0,75 per finestra (scheletri da soli 0,78 e 0,71). ~1 s di CPU per finestra.
LATO_PIXEL = (171, 128)                 # larghezza, altezza prima del ritaglio centrale 112×112
FRAME_PIXEL = 16
MEDIA_KINETICS = np.array([0.43216, 0.394666, 0.37645], np.float32)
DEVIAZIONE_KINETICS = np.array([0.22803, 0.22145, 0.216989], np.float32)


def riduci_per_pixel(frame: np.ndarray) -> np.ndarray:
    """Frame BGR del lettore → RGB 171×128 (come `scale=171:128` di ffmpeg nei kernel, che è bicubico)."""
    return cv2.cvtColor(cv2.resize(frame, LATO_PIXEL, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)


class RamoPixel:
    """percorso_rete: pesi della rete fine-tunata sulla violenza (D53, `rete_violenza.pt` del kernel finetune-rgb,
    con la sua `testa_violenza.npz`); senza, la rete Kinetics congelata di D48."""
    def __init__(self, percorso_testa: str, dispositivo: str = "cpu", percorso_rete: str | None = None):
        import torchvision
        pesi = None if percorso_rete else torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1
        self.rete = torchvision.models.video.r2plus1d_18(weights=pesi)
        self.rete.fc = torch.nn.Identity()
        if percorso_rete:
            stato = torch.load(percorso_rete, map_location="cpu")
            self.rete.load_state_dict({chiave: valore.float() for chiave, valore in stato.items()})
        self.rete = self.rete.to(dispositivo).eval()
        self.dispositivo = dispositivo
        with np.load(percorso_testa) as testa:
            self.testa = {chiave: testa[chiave] for chiave in ("media", "scala", "pesi", "bias")}

    @torch.no_grad()
    def vettore(self, frame_ridotti: list[np.ndarray]) -> np.ndarray:
        scelti = np.linspace(0, len(frame_ridotti) - 1, FRAME_PIXEL).round().astype(int)
        ritagli = np.stack([frame_ridotti[indice] for indice in scelti])[:, 8:120, 29:141]
        ingresso = (ritagli.astype(np.float32) / 255 - MEDIA_KINETICS) / DEVIAZIONE_KINETICS   # [16, 112, 112, 3]
        return self.rete(torch.from_numpy(ingresso).permute(3, 0, 1, 2)[None].to(self.dispositivo))[0].cpu().numpy()

    def probabilita_violenza(self, frame_ridotti: list[np.ndarray]) -> float:
        testa = self.testa
        logit = ((self.vettore(frame_ridotti) - testa["media"]) / testa["scala"]) @ testa["pesi"] + testa["bias"]
        return float(1 / (1 + np.exp(-logit)))


def fondi_con_pixel(probabilita: np.ndarray, probabilita_pixel: float) -> np.ndarray:
    """4 classi con la violenza = media tra pixel e scheletri; dentro i due gruppi (classi violente / non violente)
    restano le proporzioni degli scheletri."""
    violente = np.isin(CLASSI, CLASSI_VIOLENTE)
    totale_violenza = (probabilita[violente].sum() + probabilita_pixel) / 2
    fuse = probabilita.astype(np.float64)
    for gruppo, totale in ((violente, totale_violenza), (~violente, 1 - totale_violenza)):
        somma = fuse[gruppo].sum()
        fuse[gruppo] = fuse[gruppo] / somma * totale if somma > 0 else totale / gruppo.sum()
    return fuse.astype(np.float32)


def classe_finale_con_pixel(probabilita_fuse: np.ndarray, arma_confermata: bool) -> str:
    """Prima violenza sì/no (gruppo violento > 0,5, come nella misura della fusione), poi la classe più probabile del
    gruppo; infine la regola di D6 come in `classe_finale`."""
    violente = np.isin(CLASSI, CLASSI_VIOLENTE)
    gruppo = np.flatnonzero(violente if probabilita_fuse[violente].sum() > 0.5 else ~violente)
    classe = CLASSI[int(gruppo[np.argmax(probabilita_fuse[gruppo])])]
    if classe == "violenza" and arma_confermata:
        return "violenza_con_arma"
    return classe


# ALLARME VIOLENZA (D52): la stessa regola con cui si misura il sistema sui pezzi di video (`valuta-allarme`,
# `runs/finetune_rgb/fusione_finetune.json`): probabilità mediate sulle ultime MEDIA_MOBILE_ALLARME finestre, allarme
# quando FINESTRE_DI_FILA_ALLARME finestre di fila sono violente. 7 e 1 = scelti sul validation per la rete video
# fine-tunata (D53) + scheletri a pezzi da 30 s (0,90 sul test). Con la rete congelata di D48 erano 5 e 5 (0,82). Con i pixel "violenta" = gruppo violento > 0,5 (come nella misura della fusione);
# senza, la classe più probabile (come `allarmi_per_video`).
FINESTRE_DI_FILA_ALLARME = 7
MEDIA_MOBILE_ALLARME = 1


class AllarmeViolenza:
    def __init__(self, finestre_di_fila: int = FINESTRE_DI_FILA_ALLARME, media: int = MEDIA_MOBILE_ALLARME,
                 con_pixel: bool = True):
        self.finestre_di_fila, self.con_pixel = finestre_di_fila, con_pixel
        self.ultime = deque(maxlen=media)
        self.di_fila = 0

    def aggiorna(self, probabilita: np.ndarray) -> bool:
        self.ultime.append(probabilita)
        media = np.mean(self.ultime, axis=0)
        if self.con_pixel:
            violenta = classe_finale_con_pixel(media, False) in CLASSI_VIOLENTE
        else:
            violenta = CLASSI[int(np.argmax(media))] in CLASSI_VIOLENTE
        self.di_fila = self.di_fila + 1 if violenta else 0
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
    (video pubblici dei dataset, D58). pannello_nel_frame: False nel pannello web, che mostra le classi per conto suo."""

    def __init__(self, percorso_classificatore: str | list[str] | None = None, percorso_modello_armi: str | None = None,
                 percorso_modello_pose: str = MODELLO_POSE_PREDEFINITO, dispositivo: str = "cpu",
                 taratura: dict | None = None, percorso_modello_mani: str | None = None,
                 tracker: str = TRACKER_PREDEFINITO, percorso_testa_pixel: str | None = None,
                 finestre_di_fila: int = FINESTRE_DI_FILA_ALLARME, media_mobile_allarme: int = MEDIA_MOBILE_ALLARME,
                 percorso_rete_pixel: str | None = None, solo_scheletri: bool = False, pannello_nel_frame: bool = True):
        self.regola = {"soglia_confidenza": SOGLIA_CONFIDENZA_ARMA, "frame_minimi": FRAME_MINIMI_CON_ARMA,
                       "vicinanza": VICINANZA_PREDEFINITA}
        self.regola.update({chiave: (taratura or {})[chiave] for chiave in self.regola if chiave in (taratura or {})})
        self.dispositivo = dispositivo
        self.estrattore = Estrattore(percorso_modello_pose, percorso_modello_armi, dispositivo, tracker=tracker,
                                     percorso_modello_mani=percorso_modello_mani)   # D44: anche le mani, se c'è il modello
        if percorso_classificatore is None:
            percorsi = []
        else:
            percorsi = [percorso_classificatore] if isinstance(percorso_classificatore, str) else percorso_classificatore
        self.classificatori = [carica_classificatore(percorso, dispositivo) for percorso in percorsi]
        self.ramo_pixel = (RamoPixel(percorso_testa_pixel, dispositivo, percorso_rete_pixel)
                           if percorso_testa_pixel and self.classificatori else None)
        self.allarme_violenza = AllarmeViolenza(finestre_di_fila, media_mobile_allarme, con_pixel=self.ramo_pixel is not None)
        self.solo_scheletri, self.pannello_nel_frame = solo_scheletri, pannello_nel_frame
        self.ultimi_frame = deque(maxlen=FRAME_PER_FINESTRA)   # (persone, armi) degli ultimi 40 frame
        self.ultimi_ridotti = deque(maxlen=FRAME_PER_FINESTRA)  # gli stessi frame a 171×128 RGB, solo per il ramo pixel
        self.indice_frame = 0
        self.decisione = {"probabilita": None, "classe": None, "warning": False, "allarme": False}

    @torch.no_grad()
    def analizza(self, frame: np.ndarray) -> tuple[np.ndarray, dict, dict, dict | None]:
        """→ (frame annotato, persone, armi, evento della finestra appena conclusa oppure None)."""
        persone, armi = self.estrattore.analizza_frame(frame, primo_frame_del_video=self.indice_frame == 0)
        self.ultimi_frame.append((persone, armi))
        if self.ramo_pixel:
            self.ultimi_ridotti.append(riduci_per_pixel(frame))
        inizio_finestra = self.indice_frame - FRAME_PER_FINESTRA + 1
        evento = None
        if inizio_finestra >= 0 and inizio_finestra % PASSO_FINESTRE == 0:
            evento = self.decidi(inizio_finestra, frame.shape[1], frame.shape[0])
        annotato = np.zeros_like(frame) if self.solo_scheletri else frame.copy()
        disegna_persone(annotato, persone)
        disegna_armi(annotato, armi, self.regola["soglia_confidenza"])
        if self.pannello_nel_frame:
            decisione = self.decisione
            disegna_pannello(annotato, decisione["probabilita"], decisione["classe"], decisione["warning"],
                             decisione["allarme"],
                             "in attesa di 40 frame..." if self.classificatori else "solo scheletri: pesi non trovati")
        self.indice_frame += 1
        return annotato, persone, armi, evento

    def decidi(self, inizio_finestra: int, larghezza: int, altezza: int) -> dict:
        righe_persone = [(numero, dati_persone) for numero, (dati_persone, _) in enumerate(self.ultimi_frame)]
        righe_armi = [(numero, dati_armi) for numero, (_, dati_armi) in enumerate(self.ultimi_frame)]
        timeline = timeline_da_dati(impacchetta_estrazione(righe_persone, righe_armi, FRAME_PER_FINESTRA,
                                                           larghezza, altezza, 0.0))
        evento = {"inizio_secondi": inizio_finestra / FPS_ANALISI, "fine_secondi": (self.indice_frame + 1) / FPS_ANALISI}
        if not self.classificatori:
            warning = warning_arma(timeline, 0, **self.regola)
            self.decisione = {"probabilita": None, "classe": None, "warning": warning, "allarme": False}
            return {**evento, "classe": None, "warning_arma": warning, "allarme_violenza": False, "probabilita": None}
        nodi, maschera, classe_arma = costruisci_finestra(timeline, 0)
        probabilita = probabilita_media(self.classificatori, nodi, maschera, classe_arma, self.dispositivo)
        warning = warning_arma(timeline, 0, **self.regola)
        if self.ramo_pixel:
            probabilita_pixel = self.ramo_pixel.probabilita_violenza(list(self.ultimi_ridotti))
            probabilita = fondi_con_pixel(probabilita, probabilita_pixel)
            classe = classe_finale_con_pixel(probabilita, warning)
        else:
            classe = classe_finale(probabilita, warning)
        allarme = self.allarme_violenza.aggiorna(probabilita)
        self.decisione = {"probabilita": probabilita, "classe": classe, "warning": warning, "allarme": allarme}
        evento.update({"classe": classe, "warning_arma": warning, "allarme_violenza": allarme,
                       "probabilita": dict(zip(CLASSI, np.round(probabilita, 4).tolist()))})
        if self.ramo_pixel:
            evento["probabilita_violenza_pixel"] = round(probabilita_pixel, 4)
        return evento


def riga_keypoint(indice_frame: int, persone: dict, armi: dict) -> dict:
    """Una riga di keypoint.jsonl: chi c'è nel frame, dove, e le armi viste (in pixel del frame)."""
    return {"frame": indice_frame, "secondi": round(indice_frame / FPS_ANALISI, 2),
            "persone": [{"id": int(id_traccia), "box": np.round(np.asarray(box, float), 1).tolist(),
                         "keypoint": np.round(np.asarray(keypoint, float), 3).tolist()}
                        for id_traccia, box, keypoint in zip(persone["id_traccia"], persone["box"], persone["keypoint"])],
            "armi": [{"classe": NOMI_ARMI[int(classe)], "confidenza": round(float(confidenza), 3),
                      "box": np.round(np.asarray(box, float), 1).tolist()}
                     for box, classe, confidenza in zip(armi["box"], armi["classe"], armi["confidenza"])]}


def salva_eventi(cartella_uscita: Path, eventi: list[dict]) -> None:
    (cartella_uscita / "eventi.json").write_text(json.dumps(eventi, indent=2, ensure_ascii=False))
    with open(cartella_uscita / "probabilita.csv", "w", newline="") as file_csv:
        scrittore_csv = csv.writer(file_csv)
        scrittore_csv.writerow(["inizio_secondi", "fine_secondi", *CLASSI, "classe", "warning_arma", "allarme_violenza"])
        for evento in eventi:
            probabilita = [evento["probabilita"][classe] if evento["probabilita"] else "" for classe in CLASSI]
            scrittore_csv.writerow([evento["inizio_secondi"], evento["fine_secondi"], *probabilita,
                                    evento["classe"], evento["warning_arma"], evento["allarme_violenza"]])


def analizza_video(percorso_video: str, percorso_classificatore: str | list[str] | None, cartella_uscita: str | Path,
                   percorso_modello_armi: str | None = None, percorso_modello_pose: str = MODELLO_POSE_PREDEFINITO,
                   dispositivo: str = "cpu", taratura: dict | None = None,
                   percorso_modello_mani: str | None = None, tracker: str = TRACKER_PREDEFINITO,
                   percorso_testa_pixel: str | None = None, finestre_di_fila: int = FINESTRE_DI_FILA_ALLARME,
                   media_mobile_allarme: int = MEDIA_MOBILE_ALLARME, percorso_rete_pixel: str | None = None,
                   solo_scheletri: bool = False) -> list[dict]:
    """Un video → video_annotato.mp4, eventi.json, probabilita.csv e keypoint.jsonl (opzioni come `AnalisiFlusso`)."""
    cartella_uscita = Path(cartella_uscita)
    cartella_uscita.mkdir(parents=True, exist_ok=True)
    info = leggi_info_video(percorso_video)
    analisi = AnalisiFlusso(percorso_classificatore, percorso_modello_armi, percorso_modello_pose, dispositivo, taratura,
                            percorso_modello_mani, tracker, percorso_testa_pixel, finestre_di_fila, media_mobile_allarme,
                            percorso_rete_pixel, solo_scheletri)
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


def comando_analizza_video(argomenti: list[str] | None = None) -> None:
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
                        help="pesi/testa_rgb.npz: fusione pixel + scheletri sulla violenza (D48)")
    parser.add_argument("--rete-pixel", default=None,
                        help="rete_violenza.pt fine-tunata (D53), da usare con la sua --testa-pixel testa_violenza.npz")
    parser.add_argument("--finestre-di-fila", type=int, default=FINESTRE_DI_FILA_ALLARME,
                        help="ALLARME VIOLENZA dopo tante finestre violente di fila (D52)")
    parser.add_argument("--media-mobile", type=int, default=MEDIA_MOBILE_ALLARME,
                        help="finestre su cui mediare le probabilità per l'allarme (D52)")
    parser.add_argument("--solo-scheletri", action="store_true",
                        help="scheletri e armi su fondo nero, senza i pixel del video (video pubblici dei dataset, D58)")
    argomenti = parser.parse_args(argomenti)
    taratura = json.loads(Path(argomenti.taratura).read_text())["migliore"] if argomenti.taratura else None
    eventi = analizza_video(argomenti.video, argomenti.classificatore, argomenti.uscita, argomenti.modello_armi,
                            argomenti.modello_pose, argomenti.dispositivo, taratura, argomenti.modello_mani,
                            argomenti.tracker, argomenti.testa_pixel, argomenti.finestre_di_fila,
                            argomenti.media_mobile, argomenti.rete_pixel, argomenti.solo_scheletri)
    print(f"{len(eventi)} finestre analizzate → {argomenti.uscita}")

# ==============================================================================================================
# taratura del WARNING ARMA (D35: massimo F1 sul validation set)
# Riferimento: una finestra "ha un'arma" se la sua etichetta è violenza_con_arma (le altre classi non hanno armi
# annotate). Per ogni finestra si calcola, per frame, la confidenza massima di un'arma vicina a una persona; poi si
# provano tutte le combinazioni di soglia di confidenza, frame minimi e criterio di vicinanza.
# ==============================================================================================================
SOGLIE_DA_PROVARE = tuple(round(0.25 + 0.05 * passo, 2) for passo in range(12))      # 0,25 … 0,80
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
    criteri_detector = [criterio for criterio in criteri if criterio != "mani"]
    if not criteri_detector:
        return confidenze
    for indice in np.flatnonzero((timeline.armi_frame >= inizio_finestra) & (timeline.armi_frame < fine_finestra)):
        frame = int(timeline.armi_frame[indice])
        persone = persone_nel_frame(timeline, frame)
        if any(arma_vicina(timeline.armi_box[indice], *persone, criterio) for criterio in criteri_detector):
            posizione = frame - inizio_finestra
            confidenze[posizione] = max(confidenze[posizione], float(timeline.armi_confidenza[indice]))
    return confidenze


def tara_warning_arma(confidenze_per_vicinanza: dict[str, np.ndarray], arma_vera: np.ndarray) -> dict:
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
    migliore = max(tabella, key=lambda riga: (riga["f1"], riga["precisione"]))
    return {"migliore": migliore, "tabella": tabella}


def comando_tara_warning(argomenti: list[str] | None = None) -> None:
    from .modelli import carica_timeline
    from .dati import INDICE_CLASSE, nome_file_estratto

    parser = argparse.ArgumentParser(description="Taratura del WARNING ARMA (D35) sulle finestre di validazione")
    parser.add_argument("--finestre", required=True, help="finestre.npz (id_clip, inizio_finestra, etichetta)")
    parser.add_argument("--split", required=True, help="split.npz prodotto dall'addestramento (indici 'val')")
    parser.add_argument("--estratti", nargs="+", required=True)
    parser.add_argument("--uscita", required=True, help="file .json con migliore combinazione e tabella")
    argomenti = parser.parse_args(argomenti)

    with np.load(argomenti.finestre) as finestre:
        id_clip, inizi, etichette = finestre["id_clip"], finestre["inizio_finestra"], finestre["etichetta"]
    indici_val = np.load(argomenti.split)["val"]
    percorso_per_nome = {file.name: file for cartella in argomenti.estratti for file in Path(cartella).glob("*.npz")}
    confidenze = {vicinanza: [] for vicinanza in VICINANZE_DA_PROVARE}
    timeline_in_memoria = {}
    for indice in indici_val:
        nome = nome_file_estratto(str(id_clip[indice]))
        if nome not in timeline_in_memoria:
            timeline_in_memoria = {nome: carica_timeline(percorso_per_nome[nome])}  # una clip alla volta in memoria
        for vicinanza in VICINANZE_DA_PROVARE:
            confidenze[vicinanza].append(confidenza_arma_per_frame(timeline_in_memoria[nome], int(inizi[indice]),
                                                                   vicinanza))
    arma_vera = etichette[indici_val] == INDICE_CLASSE["violenza_con_arma"]
    risultato = tara_warning_arma({m: np.stack(c) for m, c in confidenze.items()}, arma_vera)
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
    from .dati import INDICE_CLASSE
    from .modelli import metriche
    predette_sistema = np.array([INDICE_CLASSE[classe_finale(p, w)] for p, w in zip(probabilita, warning)])
    con_arma = etichette == INDICE_CLASSE["violenza_con_arma"]
    return {"solo_modello": metriche(etichette, probabilita.argmax(1)),
            "modello_piu_regola": metriche(etichette, predette_sistema),
            "finestre_con_warning": int(warning.sum()), "finestre": len(etichette),
            "warning_su_finestre_con_arma": round(float(warning[con_arma].mean()), 4) if con_arma.any() else None,
            "warning_su_finestre_senza_arma": round(float(warning[~con_arma].mean()), 4)}


def comando_valuta_sistema(argomenti: list[str] | None = None) -> None:
    from .dati import nome_file_estratto
    from .modelli import carica_timeline

    parser = argparse.ArgumentParser()
    parser.add_argument("--finestre", required=True)
    parser.add_argument("--split", required=True, help="split.npz dell'addestramento (indici 'test')")
    parser.add_argument("--probabilita", required=True, help="probabilita_test.npy dello stesso addestramento")
    parser.add_argument("--taratura", required=True, help="taratura_warning.json (combinazione 'migliore')")
    parser.add_argument("--estratti", nargs="+", required=True)
    parser.add_argument("--uscita", required=True)
    argomenti = parser.parse_args(argomenti)

    regola = json.loads(Path(argomenti.taratura).read_text())["migliore"]
    opzioni = {chiave: regola[chiave] for chiave in ("soglia_confidenza", "vicinanza") if chiave in regola}
    with np.load(argomenti.finestre) as finestre:
        id_clip, inizi, etichette = finestre["id_clip"], finestre["inizio_finestra"], finestre["etichetta"]
    indici_test = np.load(argomenti.split)["test"]
    percorso_per_nome = {file.name: file for cartella in argomenti.estratti for file in Path(cartella).glob("*.npz")}
    warning, timeline_in_memoria = [], {}
    for indice in indici_test:
        nome = nome_file_estratto(str(id_clip[indice]))
        if nome not in timeline_in_memoria:
            timeline_in_memoria = {nome: carica_timeline(percorso_per_nome[nome])}  # una clip alla volta in memoria
        warning.append(warning_arma(timeline_in_memoria[nome], int(inizi[indice]), regola["frame_minimi"], **opzioni))
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


def pezzi_di_video(id_clip: np.ndarray, inizi: np.ndarray, secondi: float | None) -> np.ndarray:
    """I video lunghi contano come tanti pezzi da `secondi`: un video lungo con una rissa da qualche parte farebbe
    scattare l'allarme quasi sempre, e "per video" gonfierebbe l'accuratezza (CCTV-Fights: tutti i video hanno una rissa)."""
    if not secondi:
        return np.asarray(id_clip)
    pezzo = (np.asarray(inizi) // int(round(secondi * FPS_ANALISI))).astype(str)
    return np.char.add(np.char.add(np.asarray(id_clip).astype(str), "#"), pezzo)


def finestre_di_fila_massime(valori: np.ndarray) -> int:
    massimo = corrente = 0
    for valore in valori:
        corrente = corrente + 1 if valore else 0
        massimo = max(massimo, corrente)
    return massimo


def allarmi_per_video(probabilita: np.ndarray, etichette: np.ndarray, id_clip: np.ndarray, inizi: np.ndarray,
                      finestre_di_fila: int, media: int = 1) -> dict:
    indici_violenti = [CLASSI.index(classe) for classe in CLASSI_VIOLENTE]
    predette = np.isin(media_mobile(probabilita, id_clip, inizi, media).argmax(1), indici_violenti)
    vere = np.isin(etichette, indici_violenti)
    video_veri, video_allarme = [], []
    for clip in np.unique(id_clip):
        righe = np.where(id_clip == clip)[0]
        righe = righe[np.argsort(inizi[righe], kind="stable")]
        video_veri.append(vere[righe].any())
        video_allarme.append(finestre_di_fila_massime(predette[righe]) >= finestre_di_fila)
    video_veri, video_allarme = np.array(video_veri), np.array(video_allarme)
    quota = lambda maschera, valori: round(float(valori[maschera].mean()), 4) if maschera.any() else None
    return {"finestre_di_fila": finestre_di_fila, "media_mobile": media, "video": len(video_veri),
            "video_violenti": int(video_veri.sum()),
            "accuratezza": round(float((video_veri == video_allarme).mean()), 4),
            "recall": quota(video_veri, video_allarme), "falsi_allarmi": quota(~video_veri, video_allarme),
            "precisione": quota(video_allarme, video_veri)}


def tara_allarme(probabilita: np.ndarray, etichette: np.ndarray, id_clip: np.ndarray, inizi: np.ndarray) -> dict:
    tabella = [allarmi_per_video(probabilita, etichette, id_clip, inizi, di_fila, media)
               for media in MEDIE_MOBILI_DA_PROVARE for di_fila in FINESTRE_DI_FILA_DA_PROVARE]
    migliore = max(tabella, key=lambda riga: (riga["accuratezza"], -(riga["falsi_allarmi"] or 0)))
    return {"migliore": migliore, "tabella": tabella}


def comando_valuta_allarme(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finestre-val", required=True, help="finestre_val.npz di `addestra` (etichetta, id_clip, ...)")
    parser.add_argument("--probabilita-val", required=True, nargs="+", help="uno o più .npy: si fa la media")
    parser.add_argument("--finestre-test", required=True)
    parser.add_argument("--probabilita-test", required=True, nargs="+")
    parser.add_argument("--secondi-pezzo", type=float, default=None,
                        help="misura per pezzi di video di questa durata invece che per video interi (es. 10)")
    parser.add_argument("--uscita", required=True)
    argomenti = parser.parse_args(argomenti)
    insiemi = {}
    for nome, percorso_finestre, percorsi_probabilita in (("val", argomenti.finestre_val, argomenti.probabilita_val),
                                                          ("test", argomenti.finestre_test, argomenti.probabilita_test)):
        with np.load(percorso_finestre) as finestre:
            insiemi[nome] = (np.mean([np.load(percorso) for percorso in percorsi_probabilita], axis=0),
                             finestre["etichetta"],
                             pezzi_di_video(finestre["id_clip"], finestre["inizio_finestra"], argomenti.secondi_pezzo),
                             finestre["inizio_finestra"])
    taratura = tara_allarme(*insiemi["val"])
    scelta = taratura["migliore"]
    risultato = {"secondi_pezzo": argomenti.secondi_pezzo, "scelta_sul_val": scelta, "tabella_val": taratura["tabella"],
                 "test": allarmi_per_video(*insiemi["test"], scelta["finestre_di_fila"], scelta["media_mobile"]),
                 "test_allarme_alla_prima_finestra": allarmi_per_video(*insiemi["test"], 1, 1)}
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
CARTELLA_PAGINA = Path(__file__).parent / "pannello"
FILE_PAGINA = {"/": ("index.html", "text/html; charset=utf-8"), "/pannello.css": ("pannello.css", "text/css"),
               "/pannello.js": ("pannello.js", "text/javascript"), "/logo.svg": ("logo.svg", "image/svg+xml")}
FILE_ARCHIVIO = {"video_annotato.mp4": "video/mp4", "eventi.json": "application/json", "probabilita.csv": "text/csv"}
SECONDI_SEGMENTO = 10
# I pesi del sistema consegnato (D60): allegati alla Release v1.0 del repo, si scaricano con `argo scarica-pesi`.
FILE_PESI = {"classificatore": "stgcn.pt", "modello_armi": "armi_v3.pt", "rete_pixel": "rete_violenza.pt",
             "testa_pixel": "testa_violenza.npz", "taratura": "taratura_warning.json"}
URL_PESI = "https://github.com/marcozambrella/argo/releases/download/v1.0/"


def ingresso_ffmpeg(sorgente: str, come_telecamera: bool = False) -> list[str]:
    """Opzioni d'ingresso di ffmpeg per una telecamera: RTSP, http(s) (MJPEG/HLS, anche un telefono usato come
    telecamera IP), webcam (/dev/videoN) o un file (con come_telecamera letto a velocità reale)."""
    if sorgente.startswith("rtsp://"):
        return ["-rtsp_transport", "tcp", "-i", sorgente]
    if sorgente.startswith("/dev/video"):
        return ["-f", "v4l2", "-i", sorgente]
    if re.match(r"^[a-z][a-z0-9+.-]*://", sorgente):
        return ["-i", sorgente]
    return (["-re"] if come_telecamera else []) + ["-i", sorgente]


def sorgente_senza_password(sorgente: str) -> str:
    """rtsp://utente:password@host/… → rtsp://utente:***@host/… (nel pannello e nei log non deve comparire)."""
    return re.sub(r"(://[^:/@]+):[^@/]*@", r"\1:***@", sorgente)


def opzioni_dai_pesi(cartella: Path) -> dict:
    """Le opzioni di `AnalisiFlusso` per i pesi presenti nella cartella. Senza classificatore: modalità scheletri."""
    presenti = {chiave: cartella / nome for chiave, nome in FILE_PESI.items() if (cartella / nome).exists()}
    opzioni = {}
    if "classificatore" in presenti:
        opzioni["percorso_classificatore"] = str(presenti["classificatore"])
        if "rete_pixel" in presenti and "testa_pixel" in presenti:
            opzioni["percorso_rete_pixel"] = str(presenti["rete_pixel"])
            opzioni["percorso_testa_pixel"] = str(presenti["testa_pixel"])
    if "modello_armi" in presenti:
        opzioni["percorso_modello_armi"] = str(presenti["modello_armi"])
    if "taratura" in presenti:
        opzioni["taratura"] = json.loads(presenti["taratura"].read_text())["migliore"]
    return opzioni


def descrivi_modalita(opzioni: dict) -> str:
    if "percorso_classificatore" not in opzioni:
        return "solo scheletri"
    return "sistema completo" if "percorso_rete_pixel" in opzioni else "scheletri + ST-GCN"


class Videoregistratore:
    """ffmpeg che registra una telecamera a segmenti chiusi da `secondi_segmento` (D57)."""
    def __init__(self, sorgente: str, cartella: Path, come_telecamera: bool = False,
                 secondi_segmento: int = SECONDI_SEGMENTO):
        (cartella / "segmenti").mkdir(parents=True, exist_ok=True)
        self.elenco = cartella / "segmenti.csv"
        self.cartella_segmenti = cartella / "segmenti"
        self.inizio = time.time()
        with open(cartella / "ffmpeg.log", "w") as log:
            self.processo = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-nostdin", "-y", *ingresso_ffmpeg(sorgente, come_telecamera), "-an",
                 "-vf", f"fps={FPS_ANALISI},scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "ultrafast",
                 "-crf", "20", "-pix_fmt", "yuv420p", "-force_key_frames", "expr:gte(t,n_forced*1)",
                 "-f", "segment", "-segment_time", str(secondi_segmento), "-reset_timestamps", "1",
                 "-segment_list", str(self.elenco), "-segment_list_type", "csv",
                 str(self.cartella_segmenti / "%05d.mp4")], stderr=log)

    def segmenti_chiusi(self) -> list[Path]:
        """I segmenti finiti, in ordine: ffmpeg aggiunge una riga all'elenco solo quando chiude un segmento."""
        if not self.elenco.exists():
            return []
        righe = [riga for riga in self.elenco.read_text().splitlines() if riga.strip()]
        return [self.cartella_segmenti / Path(riga.split(",")[0]).name for riga in righe]

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
        self.al_evento = al_evento or (lambda nome, evento: None)
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
            self.stato.update(fase="errore", errore=f"{type(errore).__name__}: {errore}")
        with self.nuovo_frame:
            self.nuovo_frame.notify_all()

    def segmento_successivo(self, fatti: int) -> Path | None:
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
                    raise RuntimeError(f"nessun video dalla sorgente ({log.splitlines()[-1] if log else 'ffmpeg uscito'})")
                return None
            time.sleep(0.3)
        return None

    def analizza_segmenti(self) -> None:
        analisi = AnalisiFlusso(**self.opzioni_analisi, pannello_nel_frame=False)
        self.stato["fase"] = "in attesa del primo segmento"
        scrittore, eventi, fatti, indice_frame, velocita = None, [], 0, 0, None
        with open(self.cartella / "keypoint.jsonl", "w") as file_keypoint, \
                open(self.cartella / "eventi.jsonl", "w") as file_eventi:
            while (segmento := self.segmento_successivo(fatti)) is not None:
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
                        self.al_evento(self.nome, evento)
                    indice_frame += 1
                    durata = time.perf_counter() - inizio
                    velocita = durata if velocita is None else 0.9 * velocita + 0.1 * durata
                    self.stato.update(frame_al_secondo=round(1 / max(velocita, 1e-6), 1),
                                      secondi_analizzati=round(indice_frame / FPS_ANALISI, 1))
                    if self.dal_vivo:
                        self.stato["ritardo_secondi"] = round(
                            max(0.0, time.time() - (self.registratore.inizio + indice_frame / FPS_ANALISI)), 1)
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


def sessioni_in_archivio(cartelle: list[Path]) -> list[Path]:
    """Le cartelle con un'analisi finita (video_annotato.mp4 + eventi.json): sessioni del pannello e `analizza-video`."""
    trovate = {file.parent for cartella in cartelle if Path(cartella).exists()
               for file in Path(cartella).rglob("eventi.json") if (file.parent / "video_annotato.mp4").exists()}
    return sorted(trovate, key=lambda percorso: (percorso.stat().st_mtime, str(percorso)), reverse=True)


def riassunto_sessione(numero: int, cartella: Path, radici: list[Path]) -> dict:
    eventi = json.loads((cartella / "eventi.json").read_text())
    radice = next((Path(r) for r in radici if Path(cartella).is_relative_to(Path(r))), cartella.parent)
    return {"numero": numero, "nome": str(cartella.relative_to(radice)) if cartella != radice else cartella.name,
            "durata_secondi": eventi[-1]["fine_secondi"] if eventi else 0.0,
            "finestre_con_allarme": sum(bool(evento["allarme_violenza"]) for evento in eventi),
            "finestre_con_warning": sum(bool(evento["warning_arma"]) for evento in eventi)}


class Pannello:
    def __init__(self, telecamere: list[tuple[str, str]], cartella_sessioni: Path, cartelle_archivio: list[Path],
                 opzioni_analisi: dict, come_telecamera: bool = False):
        self.cartella_sessioni = Path(cartella_sessioni)
        self.cartelle_archivio = [self.cartella_sessioni, *map(Path, cartelle_archivio)]
        self.modalita = descrivi_modalita(opzioni_analisi)
        self.eventi, self.blocco = [], threading.Lock()
        self.telecamere = {nome: Telecamera(nome, sorgente, self.cartella_sessioni, opzioni_analisi, come_telecamera,
                                            self.aggiungi_evento)
                           for nome, sorgente in telecamere}

    def aggiungi_evento(self, nome: str, evento: dict) -> None:
        with self.blocco:
            self.eventi.append({"numero": len(self.eventi), "telecamera": nome,
                                "ora": time.strftime("%H:%M:%S"), **evento})

    def eventi_dopo(self, numero: int) -> list[dict]:
        with self.blocco:
            return self.eventi[max(numero, 0):]

    def stato(self) -> dict:
        return {"modalita": self.modalita, "telecamere": [dict(t.stato) for t in self.telecamere.values()]}

    def archivio(self) -> list[Path]:
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
                radici = self.pannello.cartelle_archivio
                self.invia_json([riassunto_sessione(numero, cartella, radici)
                                 for numero, cartella in enumerate(self.pannello.archivio())])
            elif (trovato := re.fullmatch(r"/flusso/(.+)\.mjpg", indirizzo.path)) \
                    and urllib.parse.unquote(trovato[1]) in self.pannello.telecamere:
                self.invia_mjpeg(self.pannello.telecamere[urllib.parse.unquote(trovato[1])])
            elif (trovato := re.fullmatch(r"/archivio/(\d+)/([a-z0-9_.]+)", indirizzo.path)) and trovato[2] in FILE_ARCHIVIO:
                sessioni = self.pannello.archivio()   # solo per numero: nessun percorso arriva dal browser
                if int(trovato[1]) >= len(sessioni):
                    self.send_error(404)
                else:
                    self.invia_file(sessioni[int(trovato[1])] / trovato[2], FILE_ARCHIVIO[trovato[2]])
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass

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
        inizio, fine = 0, dimensione - 1
        intervallo = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
        if intervallo and (intervallo[1] or intervallo[2]):
            if intervallo[1]:
                inizio, fine = int(intervallo[1]), min(int(intervallo[2]) if intervallo[2] else fine, fine)
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
            while restanti > 0 and (pezzo := file.read(min(restanti, 1 << 16))):
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
                telecamera.nuovo_frame.wait_for(lambda: telecamera.jpeg is not ultimo
                                                or telecamera.stato["fase"] in ("finita", "errore"), timeout=5)
                jpeg = telecamera.jpeg
            if jpeg is not None and jpeg is not ultimo:
                self.wfile.write(b"--argo\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                ultimo = jpeg
            elif telecamera.stato["fase"] in ("finita", "errore"):
                return


def crea_server(pannello: Pannello, host: str = "127.0.0.1", porta: int = 8765) -> ThreadingHTTPServer:
    gestore = type("GestoreDiQuestoPannello", (GestorePannello,), {"pannello": pannello})
    server = ThreadingHTTPServer((host, porta), gestore)
    server.daemon_threads = True
    return server


def leggi_telecamera(testo: str, numero: int) -> tuple[str, str]:
    """"ingresso=rtsp://…" → ("ingresso", "rtsp://…"); senza nome: "cam1", "cam2", …"""
    nome, uguale, sorgente = testo.partition("=")
    if uguale and re.fullmatch(r"[\w-]+", nome):
        return nome, sorgente
    return f"cam{numero}", testo


def comando_pannello(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Argo: pannello web con le telecamere (D56)")
    parser.add_argument("--telecamera", action="append", default=[], metavar="NOME=SORGENTE",
                        help="rtsp://…, http(s)://…, /dev/video0 o un file; si ripete per più telecamere")
    parser.add_argument("--come-telecamera", action="store_true",
                        help="i file si leggono a velocità reale, come una telecamera (per le prove)")
    parser.add_argument("--pesi", default="pesi", help="cartella dei pesi (`argo scarica-pesi`)")
    parser.add_argument("--sessioni", default="sessioni", help="dove registrare video e analisi delle telecamere")
    parser.add_argument("--archivio", nargs="*", default=[],
                        help="altre cartelle con analisi da rivedere (es. runs/demo_finale)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--porta", type=int, default=8765)
    parser.add_argument("--dispositivo", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--modello-pose", default=MODELLO_POSE_PREDEFINITO)
    parser.add_argument("--tracker", default=TRACKER_PREDEFINITO)
    argomenti = parser.parse_args(argomenti)
    opzioni = {**opzioni_dai_pesi(Path(argomenti.pesi)), "percorso_modello_pose": argomenti.modello_pose,
               "tracker": argomenti.tracker, "dispositivo": argomenti.dispositivo}
    if "percorso_classificatore" not in opzioni:
        print(f"Pesi non trovati in {argomenti.pesi}/: modalità solo scheletri. Per il sistema completo: uv run argo scarica-pesi")
    if argomenti.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"ATTENZIONE: il pannello è raggiungibile da altri computer su {argomenti.host} (immagini delle telecamere, senza password)")
    telecamere = [leggi_telecamera(testo, numero) for numero, testo in enumerate(argomenti.telecamera, start=1)]
    pannello = Pannello(telecamere, Path(argomenti.sessioni), [Path(c) for c in argomenti.archivio], opzioni,
                        argomenti.come_telecamera)
    server = crea_server(pannello, argomenti.host, argomenti.porta)
    print(f"Argo ({pannello.modalita}, {argomenti.dispositivo}): http://{argomenti.host}:{argomenti.porta}  — Ctrl+C per fermare")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFermo le telecamere e salvo le sessioni…")
    finally:
        server.server_close()
        pannello.ferma()


def comando_scarica_pesi(argomenti: list[str] | None = None) -> None:
    from .strumenti import scarica_con_riprova
    parser = argparse.ArgumentParser(description="Scarica i pesi del sistema consegnato dalla Release del repo (D60)")
    parser.add_argument("--cartella", default="pesi")
    parser.add_argument("--url", default=URL_PESI)
    argomenti = parser.parse_args(argomenti)
    cartella = Path(argomenti.cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    for nome in FILE_PESI.values():
        if (cartella / nome).exists():
            print(f"{nome}: c'è già")
            continue
        print(f"{nome}: scarico…", flush=True)
        parziale = cartella / (nome + ".parziale")
        scarica_con_riprova(argomenti.url + nome, parziale)
        parziale.rename(cartella / nome)
    print(f"Pesi in {cartella}/ (solo ricerca, uso non commerciale: vedi il README)")


# ==============================================================================================================
# riga di comando: python -m sicurezza.demo <comando> [opzioni]
# ==============================================================================================================
COMANDI = {"pannello": comando_pannello, "scarica-pesi": comando_scarica_pesi,
           "analizza-video": comando_analizza_video, "tara-warning": comando_tara_warning,
           "valuta-sistema": comando_valuta_sistema, "valuta-allarme": comando_valuta_allarme}


def main(argomenti: list[str] | None = None) -> None:
    argomenti = sys.argv[1:] if argomenti is None else argomenti
    if not argomenti or argomenti[0] not in COMANDI:
        print("uso: argo {pannello,scarica-pesi,analizza-video} [opzioni]  (--help dopo il comando)")
        raise SystemExit(2)
    COMANDI[argomenti[0]](argomenti[1:])


if __name__ == "__main__":
    main()
