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

"""Analisi senza GPU (non consuma quota): ricostruisce le finestre dalle estrazioni, rifà lo split (identico a quello
dell'addestramento) e confronta con le probabilità salvate → metriche per sorgente e classe prevalente per video
del test (per capire dove sbagliano i modelli e per scegliere i video della demo tra quelli mai visti)."""
import subprocess, sys
from pathlib import Path

ambiente = {**os.environ, "PYTHONPATH": "/tmp/repo/src"}
annotazioni = next(Path("/kaggle/input").glob("**/annotazioni.jsonl"))
cartelle_estratti = [str(c) for c in Path("/kaggle/input").glob("**/estratti") if c.is_dir()]
subprocess.run([sys.executable, "-m", "sicurezza.modelli", "dataset-finestre", "--annotazioni", str(annotazioni),
                "--estratti", *cartelle_estratti, "--uscita", "/tmp/finestre.npz"], check=True, env=ambiente)
probabilita = [f"{p.parent.name}={p}" for p in sorted(Path("/kaggle/input").glob("**/probabilita_test.npy"))]
print("probabilità:", probabilita, flush=True)
subprocess.run([sys.executable, "-m", "sicurezza.modelli", "analizza-predizioni", "--finestre", "/tmp/finestre.npz",
                "--probabilita", *probabilita, "--uscita", "/kaggle/working/analisi_test.json"], check=True, env=ambiente)
