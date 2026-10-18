"""Estrazione delle timeline: video a 25 fps → pose tracciate (YOLO26s-pose + BoT-SORT) + armi → un .npz per video.

L'estrazione è la parte costosa e non dipende dalle etichette: si fa una volta sola e i file si riusano per ogni
esperimento. Riprende da dove si era fermata (salta i .npz già presenti) e si divide in parti, una per GPU.

Uso (una parte per GPU):
    python -m sicurezza.estrazione --elenco elenco.jsonl --uscita estratti/ --parte 0 --parti 2 --dispositivo 0
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import FPS_ANALISI
from .video import leggi_frame, leggi_info_video

MODELLO_POSE_PREDEFINITO = "yolo26s-pose.pt"   # D3
TRACKER_PREDEFINITO = "botsort.yaml"           # tracker di default di Ultralytics (D3)


@dataclass
class VideoDaEstrarre:
    """Un video (o un suo segmento) da estrarre. `id_clip` diventa il nome del file .npz."""
    id_clip: str
    percorso_video: str
    inizio_secondi: float | None = None
    fine_secondi: float | None = None


def leggi_elenco(percorso_elenco: str | Path) -> list[VideoDaEstrarre]:
    with open(percorso_elenco) as file_elenco:
        return [VideoDaEstrarre(**json.loads(riga)) for riga in file_elenco if riga.strip()]


def scrivi_elenco(video_da_estrarre: list[VideoDaEstrarre], percorso_elenco: str | Path) -> None:
    with open(percorso_elenco, "w") as file_elenco:
        for video in video_da_estrarre:
            file_elenco.write(json.dumps(asdict(video)) + "\n")


def nome_file_estratto(id_clip: str) -> str:
    return id_clip.replace("/", "__") + ".npz"


class Estrattore:
    """Tiene caricati i modelli e produce i dati grezzi di un video, frame per frame."""

    def __init__(self, percorso_modello_pose: str = MODELLO_POSE_PREDEFINITO,
                 percorso_modello_armi: str | None = None, dispositivo: str | int | None = None,
                 tracker: str = TRACKER_PREDEFINITO, confidenza_minima_persone: float = 0.1,
                 confidenza_minima_armi: float = 0.25, mezza_precisione: bool = False):
        from ultralytics import YOLO  # import qui: i moduli che non estraggono non devono caricare torch

        self.modello_pose = YOLO(percorso_modello_pose)
        self.modello_armi = YOLO(percorso_modello_armi) if percorso_modello_armi else None
        self.dispositivo = dispositivo
        self.tracker = tracker
        # 0,1 è il default di Ultralytics per .track (D3): il tracker usa anche le detection poco sicure nel
        # secondo passaggio. Con 0,25 le tracce in una scena affollata duravano la metà (prova del 2026-10-17).
        self.confidenza_minima_persone = confidenza_minima_persone
        # soglia bassa in estrazione: il filtro vero (≥ 0,5 per il WARNING ARMA) si applica dopo, così si può tarare
        self.confidenza_minima_armi = confidenza_minima_armi
        self.mezza_precisione = mezza_precisione  # FP16 su GPU (D18): ~stessi risultati, più veloce

    def analizza_frame(self, frame: np.ndarray, primo_frame_del_video: bool) -> tuple[dict, dict]:
        """Persone tracciate e armi di un frame. Al primo frame di ogni video il tracker si azzera."""
        risultato_pose = self.modello_pose.track(
            frame, persist=not primo_frame_del_video, tracker=self.tracker, device=self.dispositivo,
            conf=self.confidenza_minima_persone, half=self.mezza_precisione, verbose=False)[0]
        box_persone = risultato_pose.boxes
        if box_persone.id is not None and len(box_persone):
            persone = {
                "id_traccia": box_persone.id.int().cpu().numpy(),
                "box": box_persone.xyxy.cpu().numpy(),
                "confidenza": box_persone.conf.cpu().numpy(),
                "keypoint": risultato_pose.keypoints.data.cpu().numpy(),  # (persone, 17, 3): x, y, confidenza
            }
        else:  # nessuna persona tracciata in questo frame
            persone = {"id_traccia": np.empty(0, np.int32), "box": np.empty((0, 4), np.float32),
                       "confidenza": np.empty(0, np.float32), "keypoint": np.empty((0, 17, 3), np.float32)}

        armi = {"box": np.empty((0, 4), np.float32), "classe": np.empty(0, np.int32),
                "confidenza": np.empty(0, np.float32)}
        if self.modello_armi is not None:
            box_armi = self.modello_armi.predict(frame, device=self.dispositivo, conf=self.confidenza_minima_armi,
                                                 half=self.mezza_precisione, verbose=False)[0].boxes
            if len(box_armi):
                armi = {"box": box_armi.xyxy.cpu().numpy(), "classe": box_armi.cls.int().cpu().numpy(),
                        "confidenza": box_armi.conf.cpu().numpy()}
        return persone, armi

    def estrai_video(self, video: VideoDaEstrarre) -> dict[str, np.ndarray]:
        """Tutti i frame del video a 25 fps → array "lunghi" (una riga per persona o arma per frame)."""
        info = leggi_info_video(video.percorso_video)
        righe_persone, righe_armi = [], []
        numero_frame = 0
        for indice_frame, frame in enumerate(leggi_frame(video.percorso_video, FPS_ANALISI,
                                                         video.inizio_secondi, video.fine_secondi)):
            persone, armi = self.analizza_frame(frame, primo_frame_del_video=indice_frame == 0)
            righe_persone.append((indice_frame, persone))
            righe_armi.append((indice_frame, armi))
            numero_frame = indice_frame + 1
        return impacchetta_estrazione(righe_persone, righe_armi, numero_frame, info.larghezza, info.altezza,
                                      video.inizio_secondi or 0.0)


def impacchetta_estrazione(righe_persone, righe_armi, numero_frame: int, larghezza: int, altezza: int,
                           inizio_secondi: float) -> dict[str, np.ndarray]:
    """Converte le liste per frame in array compatti per np.savez_compressed."""
    def concatena(righe, chiave, forma_vuota, tipo):
        blocchi = [dati[chiave] for _, dati in righe if len(dati[chiave])]
        return np.concatenate(blocchi).astype(tipo) if blocchi else np.empty(forma_vuota, tipo)

    def indici_frame(righe, chiave):
        blocchi = [np.full(len(dati[chiave]), indice_frame, np.int32) for indice_frame, dati in righe]
        return np.concatenate(blocchi) if blocchi else np.empty(0, np.int32)

    return {
        "numero_frame": np.int32(numero_frame),
        "larghezza": np.int32(larghezza),
        "altezza": np.int32(altezza),
        "fps": np.int32(FPS_ANALISI),
        "inizio_secondi": np.float32(inizio_secondi),
        "persone_frame": indici_frame(righe_persone, "id_traccia"),
        "persone_id_traccia": concatena(righe_persone, "id_traccia", (0,), np.int32),
        "persone_box": concatena(righe_persone, "box", (0, 4), np.float32),
        "persone_confidenza": concatena(righe_persone, "confidenza", (0,), np.float16),
        "persone_keypoint": concatena(righe_persone, "keypoint", (0, 17, 3), np.float32),
        "armi_frame": indici_frame(righe_armi, "classe"),
        "armi_box": concatena(righe_armi, "box", (0, 4), np.float32),
        "armi_classe": concatena(righe_armi, "classe", (0,), np.int32),
        "armi_confidenza": concatena(righe_armi, "confidenza", (0,), np.float16),
    }


def estrai_elenco(video_da_estrarre: list[VideoDaEstrarre], cartella_uscita: str | Path, estrattore: Estrattore,
                  parte: int = 0, parti: int = 1, cartelle_gia_estratte: list[str | Path] = (),
                  secondi_massimi: float | None = None) -> None:
    """Estrae la propria parte dell'elenco. Con `secondi_massimi` si ferma (pulito) prima di sforare il tempo:
    le sessioni Kaggle durano al massimo 12 h e quello che manca si riprende nella sessione dopo."""
    cartella_uscita = Path(cartella_uscita)
    cartella_uscita.mkdir(parents=True, exist_ok=True)
    gia_fatti = {file.name for cartella in [cartella_uscita, *map(Path, cartelle_gia_estratte)]
                 if cartella.exists() for file in cartella.glob("*.npz")}
    mia_parte = video_da_estrarre[parte::parti]
    da_fare = [video for video in mia_parte if nome_file_estratto(video.id_clip) not in gia_fatti]
    print(f"[parte {parte}/{parti}] {len(mia_parte)} video, {len(mia_parte) - len(da_fare)} già estratti, "
          f"{len(da_fare)} da fare", flush=True)

    frame_totali, inizio = 0, time.perf_counter()
    for numero, video in enumerate(da_fare, 1):
        if secondi_massimi is not None and time.perf_counter() - inizio > secondi_massimi:
            print(f"[parte {parte}] tempo massimo raggiunto: mi fermo, {len(da_fare) - numero + 1} video restano "
                  f"per la prossima sessione", flush=True)
            break
        file_finale = cartella_uscita / nome_file_estratto(video.id_clip)
        try:
            dati = estrattore.estrai_video(video)
        except Exception as errore:  # un video rotto non deve fermare ore di estrazione
            print(f"  ERRORE {video.id_clip}: {errore}", flush=True)
            with open(cartella_uscita / f"errori_parte{parte}.txt", "a") as file_errori:
                file_errori.write(f"{video.id_clip}\t{errore}\n")
            continue
        file_temporaneo = file_finale.with_suffix(".tmp.npz")
        np.savez_compressed(file_temporaneo, **dati)
        os.replace(file_temporaneo, file_finale)  # mai un .npz a metà, anche se la sessione Kaggle si chiude
        frame_totali += int(dati["numero_frame"])
        secondi = time.perf_counter() - inizio
        print(f"  [{numero}/{len(da_fare)}] {video.id_clip}: {int(dati['numero_frame'])} frame, "
              f"media {frame_totali / secondi:.1f} frame/s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--elenco", required=True, help="file .jsonl con un VideoDaEstrarre per riga")
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--parte", type=int, default=0)
    parser.add_argument("--parti", type=int, default=1)
    parser.add_argument("--dispositivo", default=None, help="es. 0 per la prima GPU, cpu per la CPU")
    parser.add_argument("--modello-pose", default=MODELLO_POSE_PREDEFINITO)
    parser.add_argument("--modello-armi", default=None, help="pesi del detector armi fine-tunato (D4)")
    parser.add_argument("--gia-estratti", nargs="*", default=[], help="cartelle con .npz di sessioni precedenti")
    parser.add_argument("--ore-massime", type=float, default=None, help="fermati dopo queste ore (sessioni Kaggle)")
    parser.add_argument("--mezza-precisione", action="store_true", help="FP16 su GPU (D18)")
    argomenti = parser.parse_args()

    estrattore = Estrattore(argomenti.modello_pose, argomenti.modello_armi, argomenti.dispositivo,
                            mezza_precisione=argomenti.mezza_precisione)
    estrai_elenco(leggi_elenco(argomenti.elenco), argomenti.uscita, estrattore, argomenti.parte, argomenti.parti,
                  argomenti.gia_estratti, argomenti.ore_massime * 3600 if argomenti.ore_massime else None)


if __name__ == "__main__":
    main()
