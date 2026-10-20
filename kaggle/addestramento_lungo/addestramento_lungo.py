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

"""DA DECIDERE 22 c (riaddestrare più a lungo), PRONTO MA NON LANCIATO: per un account con quota GPU (es. di un
compagno di squadra a cui Marco ha condiviso `marcozambrella/roma-safe-cctv-finestre`). Stesse finestre e stesso
split dell'addestramento 2 (finestre.npz del kernel solo CPU), Graph-Transformer (GPU 0) e ST-GCN (GPU 1) in
parallelo, più epoche e più pazienza; niente taratura del warning (già fatta, taratura v2: serve solo il
classificatore). ORE_MASSIME = ore di quota − ~0,7 h (avvio, ultima epoca oltre il tetto, predizioni, margine)."""
import subprocess, sys
from pathlib import Path

ORE_MASSIME = 5.0   # per ciascun modello (in parallelo): da regolare sulla quota GPU dell'account che lo lancia
OPZIONI = {"graph_transformer": ["--epoche", "30", "--pazienza", "8", "--dropout", "0.3", "--decadimento-pesi", "0.1"],
           "stgcn": ["--epoche", "40", "--pazienza", "8"]}

ambiente = {**os.environ, "PYTHONPATH": "/tmp/repo/src"}
percorso_finestre = str(next(Path("/kaggle/input").glob("**/finestre.npz")))
print("finestre:", percorso_finestre, flush=True)
processi = [subprocess.Popen([sys.executable, "-m", "sicurezza.modelli", "addestra", "--finestre", percorso_finestre,
                              "--modello", modello, "--uscita", f"/kaggle/working/{modello}",
                              "--ore-massime", str(ORE_MASSIME), *OPZIONI[modello]],
                             env={**ambiente, "CUDA_VISIBLE_DEVICES": str(gpu)})
            for gpu, modello in enumerate(["graph_transformer", "stgcn"])]
for processo in processi:
    processo.wait()
subprocess.run([sys.executable, "-m", "sicurezza.modelli", "analizza-predizioni", "--finestre", percorso_finestre,
                "--probabilita", *[f"{m}=/kaggle/working/{m}/probabilita_test.npy" for m in OPZIONI],
                "--uscita", "/kaggle/working/analisi_test.json"], check=True, env=ambiente)
