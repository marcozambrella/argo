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

"""D68, GPU: seconda parte dei dati nuovi, scippi Kaggle annotati e WVD. La prima (estrazione_nuovi) li aveva
persi: prepara-elenchi cercava in /tmp/dati, dove /kaggle/input era collegato, e con Python ≥ 3.13 il glob "**"
non entra nelle cartelle collegate. Qui si cerca direttamente in /kaggle/input. Stesse impostazioni."""
from pathlib import Path

uscita = Path("/kaggle/working")
subprocess.run([sys.executable, "-m", "sicurezza.dati", "prepara-elenchi", "--input", "/kaggle/input", "--uscita",
                str(uscita), "--sorgenti", "scippi_kaggle", "wvd"], check=True, env=ambiente)
pesi_armi = sorted(Path("/kaggle/input").glob("**/yolo26s_armi/weights/best.pt"))
pesi_mani = sorted(Path("/kaggle/input").glob("**/mani_best.pt"))
opzioni = ["--modello-armi", str(pesi_armi[0]), "--modello-mani", str(pesi_mani[0])]
opzioni += ["--tracker", "tracktrack.yaml"]   # lo stesso dei dati dell'addestramento (D47 mai fatta)
print("modelli:", opzioni, flush=True)
PROCESSI_PER_GPU = 2
NUMERO_GPU = 2
parti = PROCESSI_PER_GPU * NUMERO_GPU
processi = []
for parte in range(parti):
    comando = [sys.executable, "-m", "sicurezza.visione", "estrai", "--elenco", str(uscita / "elenco.jsonl"),
               "--uscita", str(uscita / "estratti"), "--parte", str(parte), "--parti", str(parti),
               "--dispositivo", str(parte % NUMERO_GPU), "--ore-massime", "8"]
    processi.append(subprocess.Popen(comando + opzioni, env=ambiente))
for processo in processi:
    processo.wait()
print("fatti:", len(list((uscita / "estratti").glob("*.npz"))), "file .npz", flush=True)
