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

"""Solo CPU (non consuma quota GPU): finestre etichettate da aggiungi-armi + estrazione-normali, split per video e
taratura del WARNING ARMA (D35) sul validation, che usa solo armi e mani e non il classificatore. L'output
(finestre.npz, split.npz, taratura_warning.json) serve all'addestramento su qualsiasi GPU: niente più ~1 h di GPU
spesa a costruire le finestre (20/10, addestramento 2)."""
import json, subprocess, sys
from pathlib import Path

ambiente = {**os.environ, "PYTHONPATH": "/tmp/repo/src"}
righe_annotazioni = {}   # più sorgenti: un id_clip una volta sola (come nel kernel di addestramento)
for file_annotazioni in sorted(Path("/kaggle/input").glob("**/annotazioni.jsonl")):
    for riga in file_annotazioni.read_text().splitlines():
        righe_annotazioni.setdefault(json.loads(riga)["id_clip"], riga)
ingresso = Path("/tmp/annotazioni.jsonl")
ingresso.write_text("\n".join(righe_annotazioni.values()) + "\n")
cartelle_estratti = [str(c) for c in Path("/kaggle/input").glob("**/estratti") if c.is_dir()]
print("clip annotate:", len(righe_annotazioni), "| estratti:", cartelle_estratti, flush=True)
uscita = Path("/kaggle/working")
modulo = lambda nome, *argomenti: subprocess.run([sys.executable, "-m", nome, *argomenti], check=True, env=ambiente)
modulo("sicurezza.modelli", "dataset-finestre", "--annotazioni", str(ingresso), "--estratti", *cartelle_estratti,
       "--uscita", str(uscita / "finestre.npz"))
modulo("sicurezza.modelli", "salva-split", "--finestre", str(uscita / "finestre.npz"), "--uscita", str(uscita / "split.npz"))
modulo("sicurezza.demo", "tara-warning", "--finestre", str(uscita / "finestre.npz"), "--split", str(uscita / "split.npz"),
       "--estratti", *cartelle_estratti, "--uscita", str(uscita / "taratura_warning.json"))
print(json.loads((uscita / "taratura_warning.json").read_text())["migliore"], flush=True)
