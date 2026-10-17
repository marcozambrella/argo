"""Demo: un video → video annotato + eventi.json + probabilita.csv (D12).

Ogni 10 frame, appena ci sono 40 frame, la finestra appena conclusa passa al classificatore. Il pannello mostra
l'ultima decisione fino alla successiva. Sul portatile va più lento del tempo reale (~2,7 frame/s), ma il video
prodotto è a 25 fps e fluido.

Uso:
    python -m sicurezza.inferenza --video ingresso.mp4 --classificatore runs/gt/migliore.pt \
        --modello-armi runs/armi/yolo26s_armi/weights/best.pt --uscita demo/
"""

import argparse
import csv
import json
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .allarme_arma import SOGLIA_CONFIDENZA_ARMA, classe_finale, warning_arma
from .caratteristiche import costruisci_finestra, timeline_da_dati
from .config import CLASSI, FPS_ANALISI, FRAME_PER_FINESTRA, PASSO_FINESTRE
from .disegno import disegna_armi, disegna_pannello, disegna_persone
from .estrazione import MODELLO_POSE_PREDEFINITO, Estrattore, impacchetta_estrazione
from .modelli import crea_modello
from .video import leggi_frame, leggi_info_video


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--classificatore", required=True, help="migliore.pt prodotto da sicurezza.addestramento")
    parser.add_argument("--modello-armi", default=None)
    parser.add_argument("--modello-pose", default=MODELLO_POSE_PREDEFINITO)
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--dispositivo", default="cpu")
    argomenti = parser.parse_args()
    eventi = analizza_video(argomenti.video, argomenti.classificatore, argomenti.uscita, argomenti.modello_armi,
                            argomenti.modello_pose, argomenti.dispositivo)
    print(f"{len(eventi)} finestre analizzate → {argomenti.uscita}")


if __name__ == "__main__":
    main()
