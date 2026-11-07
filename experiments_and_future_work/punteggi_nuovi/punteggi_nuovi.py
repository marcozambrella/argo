# ===== avvio comune: uguale in tutti i kernel di training/ =====
# Kaggle carica un solo file per kernel, quindi questo blocco è copiato in testa a ognuno.
import os
import shutil
import subprocess
import sys
URL_REPO = "https://github.com/marcozambrella/argo"
# D68: gli esperimenti usano il branch esperimenti-armi-scippi, non main
subprocess.run(["git", "clone", "--depth", "1", "--branch", "esperimenti-armi-scippi", URL_REPO, "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4", "lap"], check=True)
sys.path.insert(0, "/tmp/repo/src")
commit = subprocess.run(["git", "-C", "/tmp/repo", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
print("commit:", commit.stdout.strip(), flush=True)
if shutil.which("ffprobe") is None:  # il lettore video usa ffmpeg/ffprobe: build statica se l'immagine non li ha
    subprocess.run("curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | "
                   "tar -xJ -C /tmp && cp /tmp/ffmpeg-*-static/ff* /usr/local/bin/", shell=True, check=True)
# i comandi `python -m sicurezza...` lanciati dai kernel trovano il pacchetto in /tmp/repo/src
ambiente = dict(os.environ)
ambiente["PYTHONPATH"] = "/tmp/repo/src"
# =====
sys.stdout.reconfigure(line_buffering=True)   # anche le stampe senza flush arrivano subito nel log

"""Goal 80% (07/11, Marco: "per i test di accuracy integra anche i test sui nuovi dataset"): punteggi delle reti video
già addestrate (b1, b2, b4, b5, b1a: i loro migliore.pt fra gli input) sulle finestre di val e test di argo-anomalie
(NWPU, CHAD, ATMA-V), che b1, b2, b4 e b5 non hanno mai visto. Stesse finestre di b1a e clip_video (inizio multiplo di
20 frame), stesso ritaglio e stessa normalizzazione di rete_video.py; il nome della variante si legge dal percorso."""
import json
import re
import time
from pathlib import Path
import numpy as np
import torch
import torchvision
from sicurezza.dati import INDICE_CLASSE
from sicurezza.demo import salva_punteggi

LATI = {"b1": 112, "b2": 224, "b4": 160, "b5": 192, "b1a": 112}
ETICHETTA_DEL_TIPO = {"normale": INDICE_CLASSE["normale"], "violenza": INDICE_CLASSE["violenza"],
                      "arma": INDICE_CLASSE["violenza_con_arma"], "scippo": INDICE_CLASSE["scippo"],
                      "furto": -1, "altra": -1}
MEDIA = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
DEVIAZIONE = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)
LOTTO = 16
avvio = time.time()


def trascorse() -> float:
    return (time.time() - avvio) / 3600


def misure(variante: str) -> tuple:
    """(lato, lato_x, altezza, larghezza) come in rete_video.py."""
    lato = LATI[variante]
    lato_x = lato
    if variante == "b5":
        lato_x = lato * 4 // 3
    altezza = lato * 8 // 7
    larghezza = altezza * 4 // 3 + (altezza * 4 // 3) % 2
    return lato, lato_x, altezza, larghezza


def sedici(frame: np.ndarray) -> np.ndarray:
    return frame[np.linspace(0, len(frame) - 1, 16).round().astype(int)]


def frame_del_video(percorso: str, altezza: int, larghezza: int) -> np.ndarray:
    """Tutti i frame del video a 25 fps (i video di argo-anomalie sono corti, 480p)."""
    comando = ["ffmpeg", "-v", "error", "-i", percorso, "-vf", f"fps=25,scale={larghezza}:{altezza}",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    grezzi = np.frombuffer(subprocess.run(comando, capture_output=True).stdout, np.uint8)
    byte_per_frame = altezza * larghezza * 3
    numero = grezzi.size // byte_per_frame
    return grezzi[: numero * byte_per_frame].reshape(numero, altezza, larghezza, 3)
# --- esecuzione ---


if __name__ == "__main__":
    ingresso = Path("/kaggle/input")
    uscita = Path("/kaggle/working")
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    percorso_anomalie = next(ingresso.glob("**/anomalie.json"))
    finestre_per_split = {"val": [], "test": []}
    for voce in json.loads(percorso_anomalie.read_text()):
        if voce["split"] == "train":
            continue
        for inizio_secondi, tipo in voce["finestre"]:
            inizio = int(round(inizio_secondi * 25))
            if inizio % 20 != 0:
                continue
            finestre_per_split[voce["split"]].append((voce["chiave"], voce["video"], inizio, tipo))
    print({split: len(f) for split, f in finestre_per_split.items()}, flush=True)

    pesi_trovati = []
    for percorso in sorted(ingresso.glob("**/*migliore.pt")):
        trovata = re.search(r"(b1a|b1|b2|b4|b5)", str(percorso))
        if trovata is not None:
            pesi_trovati.append((trovata.group(1), percorso))
    print("pesi:", pesi_trovati, flush=True)

    for variante, percorso_pesi in pesi_trovati:
        lato, lato_x, altezza, larghezza = misure(variante)
        stato = torch.load(percorso_pesi, map_location="cpu")
        numero_uscite = stato["fc.weight"].shape[0]
        rete = torchvision.models.video.r2plus1d_18()
        rete.fc = torch.nn.Linear(512, numero_uscite)
        stato_float = {}
        for chiave, valore in stato.items():
            stato_float[chiave] = valore.float()
        rete.load_state_dict(stato_float)
        rete = rete.to(dispositivo).eval()
        uscite_nomi = ["violenza", "arma", "fuoco", "coltello", "scippo"]
        if numero_uscite == 6:
            uscite_nomi.append("anomalia")
        cartella = uscita / variante
        cartella.mkdir(exist_ok=True)
        for split, finestre in finestre_per_split.items():
            per_video = {}
            for chiave, video, inizio, tipo in finestre:
                if video not in per_video:
                    per_video[video] = []
                per_video[video].append((chiave, inizio, tipo))
            id_clip = []
            inizi = []
            etichette = []
            sorgenti = []
            probabilita = []
            for video, righe in per_video.items():
                frame = frame_del_video(str(percorso_anomalie.parent / video), altezza, larghezza)
                lotto = []
                for chiave, inizio, tipo in righe:
                    pezzo = frame[inizio: inizio + 40]
                    if len(pezzo) < 16:
                        continue
                    pezzo = sedici(pezzo)
                    y = (altezza - lato) // 2
                    x = (larghezza - lato_x) // 2
                    lotto.append(pezzo[:, y:y + lato, x:x + lato_x])
                    id_clip.append("anomalie/" + chiave)
                    inizi.append(inizio)
                    etichette.append(ETICHETTA_DEL_TIPO[tipo])
                    sorgenti.append("anomalie_" + chiave.split("__")[0])
                for primo in range(0, len(lotto), LOTTO):
                    tensore = torch.from_numpy(np.ascontiguousarray(np.stack(lotto[primo: primo + LOTTO])))
                    tensore = tensore.permute(0, 4, 1, 2, 3).float() / 255
                    tensore = ((tensore - MEDIA) / DEVIAZIONE).to(dispositivo)
                    with torch.no_grad():
                        probabilita.append(torch.sigmoid(rete(tensore)).cpu().numpy())
            probabilita = np.concatenate(probabilita)
            finestre_scelte = {"id_clip": np.array(id_clip), "inizio_finestra": np.array(inizi),
                               "etichetta": np.array(etichette), "sorgente": np.array(sorgenti)}
            punteggi = {}
            for numero, nome in enumerate(uscite_nomi):
                if nome != "violenza":
                    punteggi[nome] = probabilita[:, numero]
            salva_punteggi(cartella / f"punteggi_{split}.npz", finestre_scelte, punteggi)
            np.save(cartella / f"violenza_{split}.npy", probabilita[:, 0])
            print(f"{variante} {split}: {len(id_clip)} finestre, {trascorse():.2f} h", flush=True)
    print("fatto", flush=True)
