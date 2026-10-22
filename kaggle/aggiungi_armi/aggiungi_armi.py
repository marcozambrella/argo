# ===== avvio comune (da kaggle/avvio_comune.py) =====
# Blocco iniziale comune ai kernel (copiato in testa a ogni script: Kaggle carica un solo file per kernel).
import os, shutil, subprocess, sys
URL_REPO = "https://github.com/marcozambrella/argo"
subprocess.run(["git", "clone", "--depth", "1", URL_REPO, "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4", "lap"], check=True)
sys.path.insert(0, "/tmp/repo/src")
print("commit:", subprocess.run(["git", "-C", "/tmp/repo", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip(), flush=True)
if shutil.which("ffprobe") is None:  # il lettore video usa ffmpeg/ffprobe: build statica se l'immagine non li ha
    subprocess.run("curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | "
                   "tar -xJ -C /tmp && cp /tmp/ffmpeg-*-static/ff* /usr/local/bin/", shell=True, check=True)
# =====

"""D28: dopo un nuovo addestramento del detector armi, ricalcola SOLO le armi sui .npz dell'estrazione (le pose restano).
Sorgenti: output del kernel di estrazione (elenco.jsonl + estratti/) e del kernel del detector (best.pt nuovo).
D44: se c'è anche il classificatore delle mani (mani_best.pt), calcola pure le probabilità di arma in mano per persona."""
import subprocess, sys
from pathlib import Path

ambiente = {**os.environ, "PYTHONPATH": "/tmp/repo/src"}
elenco = next(Path("/kaggle/input").glob("**/elenco.jsonl"))
estratti = next(c for c in Path("/kaggle/input").glob("**/estratti") if c.is_dir())
pesi_armi = next(Path("/kaggle/input").glob("**/yolo26s_armi/weights/best.pt"))
pesi_mani = sorted(Path("/kaggle/input").glob("**/mani_best.pt"))
opzione_mani = ["--modello-mani", str(pesi_mani[0])] if pesi_mani else []
print("elenco:", elenco, "| estratti:", estratti, "| detector:", pesi_armi, "| mani:", opzione_mani, flush=True)
PROCESSI_PER_GPU, NUMERO_GPU = 2, 2
parti = PROCESSI_PER_GPU * NUMERO_GPU
processi = [subprocess.Popen([sys.executable, "-m", "sicurezza.visione", "aggiungi-armi", "--elenco", str(elenco),
                              "--estratti", str(estratti), "--uscita", "/kaggle/working/estratti",
                              "--modello-armi", str(pesi_armi), "--parte", str(parte), "--parti", str(parti),
                              "--dispositivo", str(parte % NUMERO_GPU), *opzione_mani], env=ambiente)
            for parte in range(parti)]
for processo in processi:
    processo.wait()
# copia anche annotazioni ed elenco, così il kernel di addestramento trova tutto in un'unica sorgente
import shutil
for nome in ("annotazioni.jsonl", "elenco.jsonl"):
    shutil.copy2(elenco.parent / nome, "/kaggle/working" / Path(nome))
print("fatti:", len(list(Path("/kaggle/working/estratti").glob("*.npz"))), "file .npz", flush=True)
