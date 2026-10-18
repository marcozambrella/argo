"""Sicurezza NN — demo. File unico (richiesta di Marco: massimo 5 file Python)."""

from collections import deque
from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys

import cv2
import numpy as np
import torch

from .dati import CLASSI, FPS_ANALISI, FRAME_PER_FINESTRA, OSSA, PASSO_FINESTRE, leggi_frame, leggi_info_video
from .modelli import TimelineVideo, costruisci_finestra, crea_modello, timeline_da_dati
from .visione import Estrattore, MODELLO_POSE_PREDEFINITO, impacchetta_estrazione


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


def frame_con_arma(timeline: TimelineVideo, inizio_finestra: int, frame_per_finestra: int = FRAME_PER_FINESTRA,
                   soglia_confidenza: float = SOGLIA_CONFIDENZA_ARMA, margine: float = MARGINE_VICINANZA) -> int:
    """Quanti frame della finestra hanno almeno un'arma sicura vicino a una persona tracciata."""
    frame_validi = set()
    fine_finestra = inizio_finestra + frame_per_finestra
    candidati = np.flatnonzero((timeline.armi_frame >= inizio_finestra) & (timeline.armi_frame < fine_finestra)
                               & (timeline.armi_confidenza >= soglia_confidenza))
    for indice in candidati:
        frame = int(timeline.armi_frame[indice])
        if frame in frame_validi:
            continue
        box_persone = np.array([box[frame] for box in timeline.box_per_traccia.values() if not np.isnan(box[frame, 0])])
        if arma_vicina_a_persona(timeline.armi_box[indice], box_persone.reshape(-1, 4), margine):
            frame_validi.add(frame)
    return len(frame_validi)


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


def disegna_pannello(frame: np.ndarray, probabilita: np.ndarray | None, classe: str | None, warning: bool) -> None:
    """Pannello in alto a sinistra: classe decisa, barre delle probabilità, WARNING ARMA."""
    stile = STILE_OVERLAY
    larghezza_pannello, altezza_riga, inizio_barre = 330, 24, 160
    altezza_pannello = altezza_riga * (len(CLASSI) + 2) + 10
    sovrapposizione = frame.copy()
    cv2.rectangle(sovrapposizione, (0, 0), (larghezza_pannello, altezza_pannello), stile["sfondo_pannello"], -1)
    cv2.addWeighted(sovrapposizione, stile["opacita_pannello"], frame, 1 - stile["opacita_pannello"], 0, frame)
    if classe is None:
        cv2.putText(frame, "in attesa di 40 frame...", (10, altezza_riga), stile["font"], stile["scala_font"],
                    stile["colore_testo"], 1, cv2.LINE_AA)
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


# ==============================================================================================================
# inferenza
# Demo: un video → video annotato + eventi.json + probabilita.csv (D12).
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


@torch.no_grad()
def analizza_video(percorso_video: str, percorso_classificatore: str, cartella_uscita: str | Path,
                   percorso_modello_armi: str | None = None, percorso_modello_pose: str = MODELLO_POSE_PREDEFINITO,
                   dispositivo: str = "cpu") -> list[dict]:
    cartella_uscita = Path(cartella_uscita)
    cartella_uscita.mkdir(parents=True, exist_ok=True)
    info = leggi_info_video(percorso_video)
    estrattore = Estrattore(percorso_modello_pose, percorso_modello_armi, dispositivo)
    classificatore = carica_classificatore(percorso_classificatore, dispositivo)
    scrittore = ScrittoreVideo(cartella_uscita / "video_annotato.mp4", info.larghezza, info.altezza)

    ultimi_frame = deque(maxlen=FRAME_PER_FINESTRA)   # (persone, armi) degli ultimi 40 frame
    eventi, decisione = [], {"probabilita": None, "classe": None, "warning": False}
    for indice_frame, frame in enumerate(leggi_frame(percorso_video)):
        persone, armi = estrattore.analizza_frame(frame, primo_frame_del_video=indice_frame == 0)
        ultimi_frame.append((persone, armi))
        inizio_finestra = indice_frame - FRAME_PER_FINESTRA + 1
        if inizio_finestra >= 0 and inizio_finestra % PASSO_FINESTRE == 0:
            righe_persone = [(numero, dati_persone) for numero, (dati_persone, _) in enumerate(ultimi_frame)]
            righe_armi = [(numero, dati_armi) for numero, (_, dati_armi) in enumerate(ultimi_frame)]
            timeline = timeline_da_dati(impacchetta_estrazione(righe_persone, righe_armi, FRAME_PER_FINESTRA,
                                                               info.larghezza, info.altezza, 0.0))
            nodi, maschera, classe_arma = costruisci_finestra(timeline, 0)
            logit = classificatore(torch.from_numpy(nodi)[None].to(dispositivo),
                                   torch.from_numpy(maschera)[None].to(dispositivo),
                                   torch.from_numpy(classe_arma.astype(np.int64))[None].to(dispositivo))
            probabilita = torch.softmax(logit, -1)[0].cpu().numpy()
            warning = warning_arma(timeline, 0)
            decisione = {"probabilita": probabilita, "classe": classe_finale(probabilita, warning), "warning": warning}
            eventi.append({"inizio_secondi": inizio_finestra / FPS_ANALISI,
                           "fine_secondi": (indice_frame + 1) / FPS_ANALISI, "classe": decisione["classe"],
                           "warning_arma": warning,
                           "probabilita": dict(zip(CLASSI, np.round(probabilita, 4).tolist()))})
        annotato = frame.copy()
        disegna_persone(annotato, persone)
        disegna_armi(annotato, armi, SOGLIA_CONFIDENZA_ARMA)
        disegna_pannello(annotato, decisione["probabilita"], decisione["classe"], decisione["warning"])
        scrittore.scrivi(annotato)
    scrittore.chiudi()

    (cartella_uscita / "eventi.json").write_text(json.dumps(eventi, indent=2, ensure_ascii=False))
    with open(cartella_uscita / "probabilita.csv", "w", newline="") as file_csv:
        scrittore_csv = csv.writer(file_csv)
        scrittore_csv.writerow(["inizio_secondi", "fine_secondi", *CLASSI, "classe", "warning_arma"])
        for evento in eventi:
            scrittore_csv.writerow([evento["inizio_secondi"], evento["fine_secondi"],
                                    *evento["probabilita"].values(), evento["classe"], evento["warning_arma"]])
    return eventi


def comando_analizza_video(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--classificatore", required=True, help="migliore.pt prodotto da `python -m sicurezza.modelli addestra`")
    parser.add_argument("--modello-armi", default=None)
    parser.add_argument("--modello-pose", default=MODELLO_POSE_PREDEFINITO)
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--dispositivo", default="cpu")
    argomenti = parser.parse_args(argomenti)
    eventi = analizza_video(argomenti.video, argomenti.classificatore, argomenti.uscita, argomenti.modello_armi,
                            argomenti.modello_pose, argomenti.dispositivo)
    print(f"{len(eventi)} finestre analizzate → {argomenti.uscita}")

# ==============================================================================================================
# taratura del WARNING ARMA (D35: massimo F1 sul validation set)
# Riferimento: una finestra "ha un'arma" se la sua etichetta è violenza_con_arma (le altre classi non hanno armi
# annotate). Per ogni finestra si calcola, per frame, la confidenza massima di un'arma vicina a una persona; poi si
# provano tutte le combinazioni di soglia di confidenza, frame minimi e margine di vicinanza.
# ==============================================================================================================
SOGLIE_DA_PROVARE = tuple(round(0.25 + 0.05 * passo, 2) for passo in range(12))      # 0,25 … 0,80
FRAME_MINIMI_DA_PROVARE = tuple(range(1, 16))
MARGINI_DA_PROVARE = (0.25, 0.5, 1.0)


def confidenza_arma_per_frame(timeline: TimelineVideo, inizio_finestra: int, margine: float,
                              frame_per_finestra: int = FRAME_PER_FINESTRA) -> np.ndarray:
    """[frame_per_finestra]: confidenza massima di un'arma vicina a una persona in ogni frame (0 se nessuna)."""
    confidenze = np.zeros(frame_per_finestra, np.float32)
    fine_finestra = inizio_finestra + frame_per_finestra
    for indice in np.flatnonzero((timeline.armi_frame >= inizio_finestra) & (timeline.armi_frame < fine_finestra)):
        frame = int(timeline.armi_frame[indice])
        box_persone = np.array([box[frame] for box in timeline.box_per_traccia.values() if not np.isnan(box[frame, 0])])
        if arma_vicina_a_persona(timeline.armi_box[indice], box_persone.reshape(-1, 4), margine):
            posizione = frame - inizio_finestra
            confidenze[posizione] = max(confidenze[posizione], float(timeline.armi_confidenza[indice]))
    return confidenze


def tara_warning_arma(confidenze_per_margine: dict[float, np.ndarray], arma_vera: np.ndarray) -> dict:
    """confidenze_per_margine: margine → [finestre, frame]; arma_vera: [finestre] bool. Ritorna la combinazione con
    F1 massimo e la tabella completa."""
    tabella = []
    for margine, confidenze in confidenze_per_margine.items():
        for soglia in SOGLIE_DA_PROVARE:
            frame_sopra_soglia = (confidenze >= soglia).sum(axis=1)
            for frame_minimi in FRAME_MINIMI_DA_PROVARE:
                previsto = frame_sopra_soglia >= frame_minimi
                veri_positivi = int((previsto & arma_vera).sum())
                precisione = veri_positivi / max(int(previsto.sum()), 1)
                recall = veri_positivi / max(int(arma_vera.sum()), 1)
                f1 = 2 * precisione * recall / max(precisione + recall, 1e-12)
                tabella.append({"margine": margine, "soglia_confidenza": soglia, "frame_minimi": frame_minimi,
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
    confidenze = {margine: [] for margine in MARGINI_DA_PROVARE}
    timeline_in_memoria = {}
    for indice in indici_val:
        nome = nome_file_estratto(str(id_clip[indice]))
        if nome not in timeline_in_memoria:
            timeline_in_memoria = {nome: carica_timeline(percorso_per_nome[nome])}  # una clip alla volta in memoria
        for margine in MARGINI_DA_PROVARE:
            confidenze[margine].append(confidenza_arma_per_frame(timeline_in_memoria[nome], int(inizi[indice]), margine))
    arma_vera = etichette[indici_val] == INDICE_CLASSE["violenza_con_arma"]
    risultato = tara_warning_arma({m: np.stack(c) for m, c in confidenze.items()}, arma_vera)
    Path(argomenti.uscita).write_text(json.dumps(risultato, indent=1, ensure_ascii=False))
    print("migliore:", risultato["migliore"])


# ==============================================================================================================
# riga di comando: python -m sicurezza.demo <comando> [opzioni]
# ==============================================================================================================
COMANDI = {"analizza-video": comando_analizza_video, "tara-warning": comando_tara_warning}


def main(argomenti: list[str] | None = None) -> None:
    argomenti = sys.argv[1:] if argomenti is None else argomenti
    if not argomenti or argomenti[0] not in COMANDI:
        print("uso: python -m sicurezza.demo {analizza-video} [opzioni]  (--help dopo il comando)")
        raise SystemExit(2)
    COMANDI[argomenti[0]](argomenti[1:])


if __name__ == "__main__":
    main()
