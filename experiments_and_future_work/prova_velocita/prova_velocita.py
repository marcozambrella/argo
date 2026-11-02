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

sys.stdout.reconfigure(line_buffering=True)

"""03/11, CPU (Marco: diminuire i tempi di pose e detector su un PC senza GPU, "fai un test rispetto alla seconda").
Sulle finestre v3 di val e test, con lo ST-GCN e il Graph-Transformer v3 già addestrati (addestramento-v3, migliore
sul val), probabilità in 4 casi: come oggi; senza il detector delle armi (nodi arma vuoti); pose a 12,5 frame/s
(un frame su due, i dispari interpolati); tutte e due. Uscita: probabilita_<caso>_<modello>_<split>.npy e metriche
per finestra (macro-F1, recall per classe) in metriche.json."""
import json
from pathlib import Path
import numpy as np
import torch
from sicurezza.modelli import crea_modello, lotti, metriche, pose_a_meta_frame, senza_detector

CASI = ("oggi", "senza_detector", "pose_12fps", "senza_detector_pose_12fps")
ingresso = Path("/kaggle/input")
uscita = Path("/kaggle/working")
torch.set_num_threads(os.cpu_count())

percorso_finestre = next(ingresso.glob("**/finestre.npz"))
with np.load(percorso_finestre) as file_finestre:
    nodi_tutti = file_finestre["nodi"]
    maschera_tutta = file_finestre["maschera"]
    classe_arma_tutta = file_finestre["classe_arma"]
    etichette = file_finestre["etichetta"]
print("finestre:", len(etichette), flush=True)

risultati = {}
for nome_modello in ("stgcn", "graph_transformer"):
    cartella = next(ingresso.glob(f"**/{nome_modello}/migliore.pt")).parent
    salvato = torch.load(cartella / "migliore.pt", map_location="cpu")
    modello = crea_modello(salvato["nome_modello"], **salvato["iperparametri_modello"])
    modello.load_state_dict(salvato["stato"])
    modello.eval()
    split = np.load(cartella / "split.npz")
    for nome_split in ("val", "test"):
        indici = split[nome_split]
        for caso in CASI:
            probabilita = []
            for indici_lotto in lotti(indici, 512, False, None):
                nodi = nodi_tutti[indici_lotto].astype(np.float32)
                maschera = maschera_tutta[indici_lotto]
                classe_arma = classe_arma_tutta[indici_lotto]
                if caso in ("senza_detector", "senza_detector_pose_12fps"):
                    nodi, maschera, classe_arma = senza_detector(nodi, maschera, classe_arma)
                if caso in ("pose_12fps", "senza_detector_pose_12fps"):
                    nodi, maschera, classe_arma = pose_a_meta_frame(nodi, maschera, classe_arma)
                with torch.no_grad():
                    logit = modello(torch.from_numpy(nodi), torch.from_numpy(maschera),
                                    torch.from_numpy(classe_arma.astype(np.int64)))
                probabilita.append(torch.softmax(logit, -1).numpy())
            probabilita = np.concatenate(probabilita)
            np.save(uscita / f"probabilita_{caso}_{nome_modello}_{nome_split}.npy", probabilita)
            misura = metriche(etichette[indici], probabilita.argmax(1))
            risultati[f"{nome_modello}/{nome_split}/{caso}"] = misura
            print(nome_modello, nome_split, caso, json.dumps(misura)[:300], flush=True)
    (uscita / "metriche.json").write_text(json.dumps(risultati, indent=1))
print("fatto", flush=True)
