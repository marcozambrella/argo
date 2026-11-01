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

"""Passo 4: estrazione delle timeline da tutte le sorgenti, 2 processi per GPU (2x T4, D18/D20), max ~11 h.
D47 (riestrazione dopo il rinnovo della quota): BoT-SORT + armi + mani in un passaggio, id nuovo `-botsort` così
l'output della prima estrazione (TrackTrack) resta per aggiungi-armi e per confronto.
Benchmark v5: ~84 frame/s totali (66 con GMC per i video da cellulare) → ~9,6 h per tutto.
Se la sessione finisce prima, rilanciare: l'output della sessione precedente va aggiunto come kernel_sources
e i video già fatti vengono saltati (--gia-estratti)."""
from pathlib import Path

uscita = Path("/kaggle/working")
subprocess.run([sys.executable, "-m", "sicurezza.dati", "prepara-elenchi", "--input", "/kaggle/input", "--uscita",
                str(uscita)], check=True, env=ambiente)

# detector armi fine-tunato dal kernel del detector (D16: prima il detector, poi l estrazione)
pesi_armi = sorted(Path("/kaggle/input").glob("**/yolo26s_armi/weights/best.pt"))
opzione_armi = []
if pesi_armi:
    opzione_armi = ["--modello-armi", str(pesi_armi[0])]
    print("detector armi:", pesi_armi[0], flush=True)
else:
    print("detector armi: NESSUNO (solo pose)", flush=True)
# D44: classificatore dell'ingrandimento sulle mani (dataset roma-safe-cctv-mani-modello), se è tra le sorgenti
pesi_mani = sorted(Path("/kaggle/input").glob("**/mani_best.pt"))
opzione_mani = []
if pesi_mani:
    opzione_mani = ["--modello-mani", str(pesi_mani[0])]
    print("modello mani:", pesi_mani[0], flush=True)
else:
    print("modello mani: NESSUNO", flush=True)
TRACKER = "botsort.yaml"   # D47: TrackTrack con le soglie di default perdeva molte persone nei video CCTV
gia_estratti = []   # output di sessioni precedenti: quei video si saltano
for cartella in Path("/kaggle/input").glob("**/estratti"):
    if cartella.is_dir():
        gia_estratti.append(str(cartella))

# 2 processi per GPU, 2 GPU: ognuno fa una parte dell'elenco
PROCESSI_PER_GPU = 2
NUMERO_GPU = 2
parti = PROCESSI_PER_GPU * NUMERO_GPU
processi = []
for parte in range(parti):
    comando = [sys.executable, "-m", "sicurezza.visione", "estrai", "--elenco", str(uscita / "elenco.jsonl"),
               "--uscita", str(uscita / "estratti"), "--parte", str(parte), "--parti", str(parti),
               "--dispositivo", str(parte % NUMERO_GPU), "--ore-massime", "11", "--tracker", TRACKER]
    comando += opzione_armi + opzione_mani + ["--gia-estratti"] + gia_estratti
    processi.append(subprocess.Popen(comando, env=ambiente))
for processo in processi:
    processo.wait()
print("fatti:", len(list((uscita / "estratti").glob("*.npz"))), "file .npz")
