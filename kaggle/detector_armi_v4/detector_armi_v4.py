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

"""D53 (2), detector v4: come il v3 (stessi dati, stessi filtri) ma YOLO26m a 1280 px invece di YOLO26s a 640, per
le armi piccole delle CCTV (v3 su Mock Attack: mAP50 0,20, coltelli ~0). Tetto 7 h (~20 epoche da ~20 min).
Il resto della descrizione è quella del v3 (D37-D39): fine-tuning su 2 GPU con
- export Roboflow (dataset Kaggle privato) + US Mock Attack (D30, CCTV vera, da Hugging Face);
- Sohas di DaSCI (D39: coltelli e pistole impugnati + oggetti innocui impugnati come negativi difficili);
- negativi difficili COCO (telefoni, coltelli da cucina, telecomandi, ...) diversi da quelli della prova;
- via da train/val i primi piani: armi con riquadro > 10% dell'immagine (D37). Il test resta intero.
Tetto D37: l'addestramento si ferma a ORE_MASSIME_ADDESTRAMENTO (best.pt si salva a ogni epoca)."""
import signal, subprocess, sys, time, zipfile
from pathlib import Path

ORE_MASSIME_ADDESTRAMENTO = 7.0      # D53: sessione da ~8 h con la preparazione dei dati
GRANDE = ["--modello-base", "yolo26m.pt", "--dimensione-immagine", "1280", "--pazienza", "10"]   # D53
AREA_MASSIMA_TRAIN = 0.10            # D37: primi piani fuori (in CCTV le armi stanno sotto l'1% del frame)
ambiente = {**os.environ, "PYTHONPATH": "/tmp/repo/src"}
visione = lambda *argomenti: subprocess.run([sys.executable, "-m", "sicurezza.visione", *argomenti], check=True,
                                            env=ambiente)

# US Mock Attack: ~1,5 GB da Hugging Face, estratto e convertito in /tmp (non finisce nell'output del kernel)
from sicurezza.strumenti import scarica_con_riprova
from sicurezza.visione import CARTELLA_SOHAS_NEL_REPO, URL_DASCI, URL_MOCK_ATTACK
import socket
socket.setdefaulttimeout(300)   # un download bloccato fa fallire il kernel invece di appenderlo
scarica_con_riprova(URL_MOCK_ATTACK, Path("/tmp/mock_attack.zip"))
zipfile.ZipFile("/tmp/mock_attack.zip").extractall("/tmp/mock_attack_voc")
Path("/tmp/mock_attack.zip").unlink()
visione("converti-mock-attack", "--sorgente", "/tmp/mock_attack_voc", "--uscita", "/tmp/mock_attack")

# Sohas (DaSCI): solo la cartella che serve del repo (~1,8 GB)
subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", URL_DASCI, "/tmp/dasci"],
               check=True)
subprocess.run(["git", "-C", "/tmp/dasci", "sparse-checkout", "set", CARTELLA_SOHAS_NEL_REPO], check=True)
visione("prepara-sohas", "--sorgente", f"/tmp/dasci/{CARTELLA_SOHAS_NEL_REPO}", "--uscita", "/tmp/sohas")

# negativi difficili COCO per il train (le immagini della prova dei falsi allarmi sono escluse)
subprocess.run([sys.executable, "-m", "sicurezza.strumenti", "scarica-negativi-coco", "--cartella",
                "/tmp/negativi_coco", "--per-categoria", "100", "--per-addestramento"], check=True, env=ambiente)

export = sorted({p.parent for p in Path("/kaggle/input").glob("**/data.yaml")})
export += [Path("/tmp/mock_attack"), Path("/tmp/sohas"), Path("/tmp/negativi_coco")]
print("export:", export, flush=True)
visione("unisci-armi", "--sorgenti", *map(str, export), "--uscita", "/tmp/armi_unite",
        "--area-massima-train", str(AREA_MASSIMA_TRAIN))

inizio = time.time()
addestramento = subprocess.Popen([sys.executable, "-m", "sicurezza.visione", "addestra-detector", "--dati",
                                  "/tmp/armi_unite/data.yaml", "--uscita", "/kaggle/working", "--dispositivo", "0,1",
                                  "--lotto", "8", *GRANDE], env=ambiente, start_new_session=True)
try:
    if addestramento.wait(timeout=ORE_MASSIME_ADDESTRAMENTO * 3600):
        raise SystemExit(f"addestramento fallito (codice {addestramento.returncode})")
    print(f"addestramento finito in {(time.time() - inizio) / 3600:.2f} h", flush=True)
except subprocess.TimeoutExpired:   # ferma anche i processi DDP figli; best.pt resta quello dell'epoca migliore
    os.killpg(addestramento.pid, signal.SIGTERM)
    addestramento.wait()
    print(f"FERMATO dopo {ORE_MASSIME_ADDESTRAMENTO} h (tetto D37): si valuta il best.pt salvato", flush=True)
    visione("addestra-detector", "--dati", "/tmp/armi_unite/data.yaml", "--uscita", "/kaggle/working",
            "--dispositivo", "0", "--solo-valutazione", *GRANDE)
