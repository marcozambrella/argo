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

"""Passo 3, detector v3 (D37-D39): fine-tuning YOLO26s su 2 GPU con
- export Roboflow (dataset Kaggle privato) + US Mock Attack (D30, CCTV vera, da Hugging Face);
- Sohas di DaSCI (D39: coltelli e pistole impugnati + oggetti innocui impugnati come negativi difficili);
- negativi difficili COCO (telefoni, coltelli da cucina, telecomandi, ...) diversi da quelli della prova;
- via da train/val i primi piani: armi con riquadro > 10% dell'immagine (D37). Il test resta intero.
Tetto D37: l'addestramento si ferma a ORE_MASSIME_ADDESTRAMENTO (best.pt si salva a ogni epoca)."""
import signal
import socket
import time
import zipfile
from pathlib import Path

from sicurezza.strumenti import scarica_con_riprova
from sicurezza.visione import CARTELLA_SOHAS_NEL_REPO, URL_DASCI, URL_MOCK_ATTACK

ORE_MASSIME_ADDESTRAMENTO = 4.75     # D37: "se ci mette meno di 5h"
AREA_MASSIMA_TRAIN = 0.10            # D37: primi piani fuori (in CCTV le armi stanno sotto l'1% del frame)


def visione(argomenti: list) -> None:
    """Lancia `python -m sicurezza.visione <argomenti>` e si ferma se fallisce."""
    subprocess.run([sys.executable, "-m", "sicurezza.visione"] + argomenti, check=True, env=ambiente)


# US Mock Attack: ~1,5 GB da Hugging Face, estratto e convertito in /tmp (non finisce nell'output del kernel)
socket.setdefaulttimeout(300)   # un download bloccato fa fallire il kernel invece di appenderlo
scarica_con_riprova(URL_MOCK_ATTACK, Path("/tmp/mock_attack.zip"))
with zipfile.ZipFile("/tmp/mock_attack.zip") as archivio:
    archivio.extractall("/tmp/mock_attack_voc")
Path("/tmp/mock_attack.zip").unlink()
visione(["converti-mock-attack", "--sorgente", "/tmp/mock_attack_voc", "--uscita", "/tmp/mock_attack"])

# Sohas (DaSCI): solo la cartella che serve del repo (~1,8 GB)
subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", URL_DASCI, "/tmp/dasci"],
               check=True)
subprocess.run(["git", "-C", "/tmp/dasci", "sparse-checkout", "set", CARTELLA_SOHAS_NEL_REPO], check=True)
visione(["prepara-sohas", "--sorgente", f"/tmp/dasci/{CARTELLA_SOHAS_NEL_REPO}", "--uscita", "/tmp/sohas"])

# negativi difficili COCO per il train (le immagini della prova dei falsi allarmi sono escluse)
subprocess.run([sys.executable, "-m", "sicurezza.strumenti", "scarica-negativi-coco", "--cartella",
                "/tmp/negativi_coco", "--per-categoria", "100", "--per-addestramento"], check=True, env=ambiente)

# gli export Roboflow sono le cartelle con un data.yaml (una volta sola ciascuna)
export = set()
for data_yaml in Path("/kaggle/input").glob("**/data.yaml"):
    export.add(data_yaml.parent)
export = sorted(export)
export += [Path("/tmp/mock_attack"), Path("/tmp/sohas"), Path("/tmp/negativi_coco")]
print("export:", export, flush=True)
sorgenti = []
for cartella in export:
    sorgenti.append(str(cartella))
visione(["unisci-armi", "--sorgenti"] + sorgenti + ["--uscita", "/tmp/armi_unite",
                                                   "--area-massima-train", str(AREA_MASSIMA_TRAIN)])

inizio = time.time()
addestramento = subprocess.Popen([sys.executable, "-m", "sicurezza.visione", "addestra-detector", "--dati",
                                  "/tmp/armi_unite/data.yaml", "--uscita", "/kaggle/working", "--dispositivo", "0,1",
                                  "--lotto", "32"], env=ambiente, start_new_session=True)
try:
    codice = addestramento.wait(timeout=ORE_MASSIME_ADDESTRAMENTO * 3600)
    if codice != 0:
        raise SystemExit(f"addestramento fallito (codice {addestramento.returncode})")
    print(f"addestramento finito in {(time.time() - inizio) / 3600:.2f} h", flush=True)
except subprocess.TimeoutExpired:   # ferma anche i processi DDP figli; best.pt resta quello dell'epoca migliore
    os.killpg(addestramento.pid, signal.SIGTERM)
    addestramento.wait()
    print(f"FERMATO dopo {ORE_MASSIME_ADDESTRAMENTO} h (tetto D37): si valuta il best.pt salvato", flush=True)
    visione(["addestra-detector", "--dati", "/tmp/armi_unite/data.yaml", "--uscita", "/kaggle/working",
             "--dispositivo", "0", "--solo-valutazione"])
