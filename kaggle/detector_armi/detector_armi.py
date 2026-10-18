# ===== avvio comune (da kaggle/avvio_comune.py) =====
# Blocco iniziale comune ai kernel (copiato in testa a ogni script: Kaggle carica un solo file per kernel).
import os, shutil, subprocess, sys
URL_REPO = "https://github.com/marcozambrella/roma-safe-cctv"
subprocess.run(["git", "clone", "--depth", "1", URL_REPO, "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4", "lap"], check=True)
sys.path.insert(0, "/tmp/repo/src")
print("commit:", subprocess.run(["git", "-C", "/tmp/repo", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip(), flush=True)
if shutil.which("ffprobe") is None:  # il lettore video usa ffmpeg/ffprobe: build statica se l'immagine non li ha
    subprocess.run("curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | "
                   "tar -xJ -C /tmp && cp /tmp/ffmpeg-*-static/ff* /usr/local/bin/", shell=True, check=True)
# =====

"""Passo 3: fine-tuning YOLO26s sul dataset armi unito (dataset Kaggle privato creato dagli export Roboflow)."""
import subprocess, sys
from pathlib import Path

export = sorted({p.parent for p in Path("/kaggle/input").glob("**/data.yaml")})
print("export trovati:", export, flush=True)
ambiente = {**os.environ, "PYTHONPATH": "/tmp/repo/src"}
subprocess.run([sys.executable, "-m", "sicurezza.visione", "unisci-armi", "--sorgenti", *map(str, export),
                "--uscita", "/tmp/armi_unite"], check=True, env=ambiente)
subprocess.run([sys.executable, "-m", "sicurezza.visione", "addestra-detector", "--dati", "/tmp/armi_unite/data.yaml",
                "--uscita", "/kaggle/working", "--dispositivo", "0"], check=True, env=ambiente)
