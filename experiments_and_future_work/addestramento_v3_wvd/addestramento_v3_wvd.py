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
"""D68, GPU: ST-GCN e Graph-Transformer sulle finestre v3 (dati nuovi, SCVD senza le armi non chiare), uno per GPU
in parallelo. Marco: niente early stop → pazienza = epoche, si fanno tutte; si salvano il migliore sul val e le
probabilità dell'ultima epoca. Variante CON i video sintetici WVD nel train (addestramento_v3: senza)."""
from pathlib import Path

SORGENTI_TOLTE = []
ORE_MASSIME = 10.0   # solo sicurezza (sessione Kaggle da 12 h): con queste epoche finiscono in ~6 h
OPZIONI = {"stgcn": ["--epoche", "30", "--pazienza", "30"],
           "graph_transformer": ["--epoche", "15", "--pazienza", "15", "--dropout", "0.3", "--decadimento-pesi", "0.1"]}

percorso_finestre = str(next(Path("/kaggle/input").glob("**/finestre.npz")))
print("finestre:", percorso_finestre, "| tolte dal train:", SORGENTI_TOLTE, flush=True)
processi = []
for gpu, modello in enumerate(OPZIONI):
    comando = [sys.executable, "-m", "sicurezza.modelli", "addestra", "--finestre", percorso_finestre,
               "--modello", modello, "--uscita", f"/kaggle/working/{modello}", "--ore-massime", str(ORE_MASSIME)]
    comando += OPZIONI[modello] + ["--senza-sorgenti-train"] + SORGENTI_TOLTE
    ambiente_gpu = dict(ambiente)
    ambiente_gpu["CUDA_VISIBLE_DEVICES"] = str(gpu)
    processi.append(subprocess.Popen(comando, env=ambiente_gpu))
for processo in processi:
    processo.wait()
print("fatto", flush=True)
