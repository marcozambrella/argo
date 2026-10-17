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

"""Passi 5-6: finestre etichettate dalle estrazioni, poi Graph-Transformer (GPU 0) e ST-GCN (GPU 1) in parallelo."""
import os, subprocess, sys
from pathlib import Path

ingresso = next(Path("/kaggle/input").glob("**/annotazioni.jsonl"))
cartelle_estratti = [str(c) for c in Path("/kaggle/input").glob("**/estratti") if c.is_dir()]
ambiente = {**os.environ, "PYTHONPATH": "/tmp/repo/src"}
subprocess.run([sys.executable, "-m", "sicurezza.dataset_finestre", "--annotazioni", str(ingresso),
                "--estratti", *cartelle_estratti, "--uscita", "/tmp/finestre.npz"], check=True, env=ambiente)
processi = [subprocess.Popen([sys.executable, "-m", "sicurezza.addestramento", "--finestre", "/tmp/finestre.npz",
                              "--modello", modello, "--uscita", f"/kaggle/working/{modello}"],
                             env={**ambiente, "CUDA_VISIBLE_DEVICES": str(gpu)})
            for gpu, modello in enumerate(["graph_transformer", "stgcn"])]
for processo in processi:
    processo.wait()
