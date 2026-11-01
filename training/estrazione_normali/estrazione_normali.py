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

"""D41/D46: scene NORMALI in luoghi affollati (stazioni, metro, incroci, campus, strade), ~3 h di video.
Prepara i video (sequenze jpg → mp4, HiEve senza le scene violente), poi pose + tracking + armi (+ mani, D44, se il
modello c'è) come l'estrazione principale: 2 processi per GPU (D18/D20), telecamere fisse (niente GMC)."""
from pathlib import Path

uscita = Path("/kaggle/working")
subprocess.run([sys.executable, "-m", "sicurezza.dati", "prepara-normali", "--input", "/kaggle/input", "--uscita",
                "/tmp", "--hieve"], check=True, env=ambiente)
subprocess.run([sys.executable, "-m", "sicurezza.dati", "prepara-elenchi", "--input", "/tmp", "--uscita", str(uscita),
                "--sorgenti", "normali_folla"], check=True, env=ambiente)

pesi_armi = sorted(Path("/kaggle/input").glob("**/yolo26s_armi/weights/best.pt"))
pesi_mani = sorted(Path("/kaggle/input").glob("**/mani_best.pt"))
opzioni = []
if pesi_armi:
    opzioni += ["--modello-armi", str(pesi_armi[0])]
if pesi_mani:
    opzioni += ["--modello-mani", str(pesi_mani[0])]
# stesso tracker dell'estrazione principale usata nell'addestramento (TrackTrack di default): la riestrazione con
# BoT-SORT (D47) non si fa prima della consegna di venerdì 23/10; quando si farà, qui va "botsort.yaml"
opzioni += ["--tracker", "tracktrack.yaml"]
print("modelli:", opzioni, flush=True)
# 2 processi per GPU, 2 GPU: ognuno fa una parte dell'elenco
PROCESSI_PER_GPU = 2
NUMERO_GPU = 2
parti = PROCESSI_PER_GPU * NUMERO_GPU
processi = []
for parte in range(parti):
    comando = [sys.executable, "-m", "sicurezza.visione", "estrai", "--elenco", str(uscita / "elenco.jsonl"),
               "--uscita", str(uscita / "estratti"), "--parte", str(parte), "--parti", str(parti),
               "--dispositivo", str(parte % NUMERO_GPU), "--ore-massime", "6"]
    processi.append(subprocess.Popen(comando + opzioni, env=ambiente))
for processo in processi:
    processo.wait()
print("fatti:", len(list((uscita / "estratti").glob("*.npz"))), "file .npz", flush=True)
