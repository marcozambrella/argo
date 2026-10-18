"""Rilevazioni VERE per il video hero: persone (box, 17 keypoint, ID del tracker) e armi, frame per frame.

Usa l'Estrattore della pipeline (src/sicurezza/visione.py) senza modificarlo: stesso modello pose, stesso tracker
TrackTrack, stesse soglie. Il risultato è un JSON che la composizione Remotion disegna sopra il video.
Coordinate normalizzate 0–1 rispetto a larghezza e altezza del video, così la grafica non dipende dalla risoluzione.

Uso (dalla radice del repo):
    uv run python site/montaggio/strumenti/estrai_rilevazioni.py --video site/montaggio/sorgenti/clip.mp4 \
        --uscita site/montaggio/public/rilevazioni/clip.json [--modello-armi runs/armi/.../best.pt]
"""

from pathlib import Path
import argparse
import json
import time

from sicurezza.dati import FPS_ANALISI, leggi_frame, leggi_info_video
from sicurezza.visione import Estrattore, MODELLO_POSE_PREDEFINITO

NOMI_ARMI = ("pistola", "coltello", "fucile")


def arrotonda(valori, cifre=4):
    return [round(float(valore), cifre) for valore in valori]


def estrai(percorso_video: str, percorso_uscita: str, percorso_modello_pose: str, percorso_modello_armi: str | None):
    info = leggi_info_video(percorso_video)
    estrattore = Estrattore(percorso_modello_pose, percorso_modello_armi, "cpu")
    estrattore.nuovo_video(telecamera_mobile=False)
    larghezza, altezza = info.larghezza, info.altezza
    frame_analizzati, lotto, inizio = [], [], time.time()

    def elabora_lotto():
        for persone, armi in estrattore.analizza_lotto(lotto):
            frame_analizzati.append({
                "persone": [
                    {"id": int(id_traccia),
                     "box": arrotonda([box[0] / larghezza, box[1] / altezza, box[2] / larghezza, box[3] / altezza]),
                     "confidenza": round(float(confidenza), 3),
                     "keypoint": [arrotonda([punto[0] / larghezza, punto[1] / altezza, punto[2]], 3)
                                  for punto in keypoint]}
                    for id_traccia, box, confidenza, keypoint in
                    zip(persone["id_traccia"], persone["box"], persone["confidenza"], persone["keypoint"])],
                "armi": [
                    {"classe": NOMI_ARMI[int(classe)],
                     "box": arrotonda([box[0] / larghezza, box[1] / altezza, box[2] / larghezza, box[3] / altezza]),
                     "confidenza": round(float(confidenza), 3)}
                    for box, classe, confidenza in zip(armi["box"], armi["classe"], armi["confidenza"])],
            })
        lotto.clear()
        print(f"\r{len(frame_analizzati)} frame ({len(frame_analizzati) / (time.time() - inizio):.1f} frame/s)",
              end="", flush=True)

    for frame in leggi_frame(percorso_video, FPS_ANALISI):
        lotto.append(frame)
        if len(lotto) == estrattore.dimensione_lotto:
            elabora_lotto()
    if lotto:
        elabora_lotto()
    print()

    risultato = {
        "_nota": "Rilevazioni reali prodotte dalla pipeline (pose + TrackTrack"
                 + (" + detector armi" if percorso_modello_armi else ", detector armi non ancora disponibile") + ").",
        "video": Path(percorso_video).name, "fps": FPS_ANALISI, "larghezza": larghezza, "altezza": altezza,
        "modello_pose": Path(percorso_modello_pose).name,
        "modello_armi": Path(percorso_modello_armi).name if percorso_modello_armi else None,
        "frame": frame_analizzati,
    }
    Path(percorso_uscita).parent.mkdir(parents=True, exist_ok=True)
    Path(percorso_uscita).write_text(json.dumps(risultato, separators=(",", ":")))
    identita = {persona["id"] for frame in frame_analizzati for persona in frame["persone"]}
    print(f"{percorso_uscita}: {len(frame_analizzati)} frame, {len(identita)} ID di traccia")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--modello-pose", default=MODELLO_POSE_PREDEFINITO)
    parser.add_argument("--modello-armi", default=None)
    argomenti = parser.parse_args()
    estrai(argomenti.video, argomenti.uscita, argomenti.modello_pose, argomenti.modello_armi)
