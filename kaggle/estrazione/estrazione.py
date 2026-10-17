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

"""Passo 4: estrazione delle timeline da tutte le sorgenti, una parte per GPU (2x T4), max ~11 h.
Se la sessione finisce prima, rilanciare: l'output della sessione precedente va aggiunto come kernel_sources
e i video già fatti vengono saltati (--gia-estratti)."""
import subprocess, sys
from pathlib import Path

uscita = Path("/kaggle/working")
subprocess.run([sys.executable, "-m", "sicurezza.prepara_elenchi", "--input", "/kaggle/input", "--uscita",
                str(uscita)], check=True, env={**os.environ, "PYTHONPATH": "/tmp/repo/src"})

# detector armi fine-tunato, se il kernel del detector è tra le sorgenti (DA DECIDERE: aspettarlo o no)
pesi_armi = sorted(Path("/kaggle/input").glob("**/yolo26s_armi/weights/best.pt"))
opzione_armi = ["--modello-armi", str(pesi_armi[0])] if pesi_armi else []
print("detector armi:", pesi_armi[0] if pesi_armi else "NESSUNO (solo pose)", flush=True)
gia_estratti = [str(cartella) for cartella in Path("/kaggle/input").glob("**/estratti") if cartella.is_dir()]

processi = [subprocess.Popen([sys.executable, "-m", "sicurezza.estrazione", "--elenco", str(uscita / "elenco.jsonl"),
                              "--uscita", str(uscita / "estratti"), "--parte", str(gpu), "--parti", "2",
                              "--dispositivo", str(gpu), "--ore-massime", "11", *opzione_armi,
                              "--gia-estratti", *gia_estratti],
                             env={**os.environ, "PYTHONPATH": "/tmp/repo/src"})
            for gpu in (0, 1)]
for processo in processi:
    processo.wait()
print("fatti:", len(list((uscita / "estratti").glob("*.npz"))), "file .npz")
