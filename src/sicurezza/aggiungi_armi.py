"""Secondo passaggio OPZIONALE (DA DECIDERE, punto 5 del diario): aggiunge le armi a .npz estratti con la sola pose.

Rilegge ogni video a 25 fps (stesso segmento dell'estrazione), passa solo il detector armi e sostituisce i campi
armi_* nel .npz. Le persone non vengono toccate.

Uso:
    python -m sicurezza.aggiungi_armi --elenco elenco.jsonl --estratti estratti/ --uscita estratti_con_armi/ \
        --modello-armi best.pt --parte 0 --parti 2 --dispositivo 0
"""

import argparse
import os
from pathlib import Path

import numpy as np

from .config import FPS_ANALISI
from .estrazione import leggi_elenco, nome_file_estratto
from .video import leggi_frame


def armi_di_un_video(modello_armi, video, dispositivo, confidenza_minima: float = 0.25) -> dict[str, np.ndarray]:
    frame_armi, box, classi, confidenze = [], [], [], []
    for indice_frame, frame in enumerate(leggi_frame(video.percorso_video, FPS_ANALISI, video.inizio_secondi,
                                                     video.fine_secondi)):
        risultato = modello_armi.predict(frame, device=dispositivo, conf=confidenza_minima, verbose=False)[0].boxes
        if len(risultato):
            frame_armi.append(np.full(len(risultato), indice_frame, np.int32))
            box.append(risultato.xyxy.cpu().numpy())
            classi.append(risultato.cls.int().cpu().numpy())
            confidenze.append(risultato.conf.cpu().numpy())
    unisci = lambda blocchi, forma, tipo: np.concatenate(blocchi).astype(tipo) if blocchi else np.empty(forma, tipo)
    return {"armi_frame": unisci(frame_armi, (0,), np.int32), "armi_box": unisci(box, (0, 4), np.float32),
            "armi_classe": unisci(classi, (0,), np.int32), "armi_confidenza": unisci(confidenze, (0,), np.float16)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--elenco", required=True)
    parser.add_argument("--estratti", required=True, help="cartella con i .npz della sola pose")
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--modello-armi", required=True)
    parser.add_argument("--parte", type=int, default=0)
    parser.add_argument("--parti", type=int, default=1)
    parser.add_argument("--dispositivo", default=None)
    argomenti = parser.parse_args()

    from ultralytics import YOLO
    modello_armi = YOLO(argomenti.modello_armi)
    cartella_uscita = Path(argomenti.uscita)
    cartella_uscita.mkdir(parents=True, exist_ok=True)
    for video in leggi_elenco(argomenti.elenco)[argomenti.parte::argomenti.parti]:
        nome = nome_file_estratto(video.id_clip)
        sorgente, destinazione = Path(argomenti.estratti) / nome, cartella_uscita / nome
        if not sorgente.exists() or destinazione.exists():
            continue
        dati = dict(np.load(sorgente))
        armi = armi_di_un_video(modello_armi, video, argomenti.dispositivo)
        if len(armi["armi_frame"]) and armi["armi_frame"].max() >= int(dati["numero_frame"]):
            print(f"ATTENZIONE {video.id_clip}: numero di frame diverso dall'estrazione, salto")
            continue
        dati.update(armi)
        temporaneo = destinazione.with_suffix(".tmp.npz")
        np.savez_compressed(temporaneo, **dati)
        os.replace(temporaneo, destinazione)
        print(f"{video.id_clip}: {len(armi['armi_frame'])} armi", flush=True)


if __name__ == "__main__":
    main()
