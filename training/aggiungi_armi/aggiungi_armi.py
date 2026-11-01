# ===== avvio comune: uguale in tutti i kernel di training/ =====
# Kaggle carica un solo file per kernel, quindi questo blocco è copiato in testa a ognuno.
import os
import shutil
import subprocess
import sys
URL_REPO = "https://github.com/marcozambrella/argo"
subprocess.run(["git", "clone", "--depth", "1", URL_REPO, "/tmp/repo"], check=True)
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

"""D28: dopo un nuovo addestramento del detector armi, ricalcola SOLO le armi sui .npz dell'estrazione (le pose restano).
Sorgenti: output del kernel di estrazione (elenco.jsonl + estratti/) e del kernel del detector (best.pt nuovo).
D44: se c'è anche il classificatore delle mani (mani_best.pt), calcola pure le probabilità di arma in mano per persona."""
from pathlib import Path

# next(...) = il primo file trovato (il kernel si ferma con un errore se non c'è)
elenco = next(Path("/kaggle/input").glob("**/elenco.jsonl"))
estratti = None
for cartella in Path("/kaggle/input").glob("**/estratti"):
    if cartella.is_dir():
        estratti = cartella
        break
if estratti is None:
    raise SystemExit("nessuna cartella estratti/ tra le sorgenti del kernel")
pesi_armi = next(Path("/kaggle/input").glob("**/yolo26s_armi/weights/best.pt"))
pesi_mani = sorted(Path("/kaggle/input").glob("**/mani_best.pt"))
opzione_mani = []
if pesi_mani:
    opzione_mani = ["--modello-mani", str(pesi_mani[0])]
print("elenco:", elenco, "| estratti:", estratti, "| detector:", pesi_armi, "| mani:", opzione_mani, flush=True)

# 2 processi per GPU, 2 GPU: ognuno fa una parte dell'elenco
PROCESSI_PER_GPU = 2
NUMERO_GPU = 2
parti = PROCESSI_PER_GPU * NUMERO_GPU
processi = []
for parte in range(parti):
    comando = [sys.executable, "-m", "sicurezza.visione", "aggiungi-armi", "--elenco", str(elenco),
               "--estratti", str(estratti), "--uscita", "/kaggle/working/estratti",
               "--modello-armi", str(pesi_armi), "--parte", str(parte), "--parti", str(parti),
               "--dispositivo", str(parte % NUMERO_GPU)]
    processi.append(subprocess.Popen(comando + opzione_mani, env=ambiente))
for processo in processi:
    processo.wait()
# copia anche annotazioni ed elenco, così il kernel di addestramento trova tutto in un'unica sorgente
for nome in ("annotazioni.jsonl", "elenco.jsonl"):
    shutil.copy2(elenco.parent / nome, "/kaggle/working" / Path(nome))
print("fatti:", len(list(Path("/kaggle/working/estratti").glob("*.npz"))), "file .npz", flush=True)
