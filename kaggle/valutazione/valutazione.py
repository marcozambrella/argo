# ===== avvio comune (da kaggle/avvio_comune.py) =====
# Blocco iniziale comune ai kernel (copiato in testa a ogni script: Kaggle carica un solo file per kernel).
import os, shutil, subprocess, sys
URL_REPO = "https://github.com/marcozambrella/argo"
subprocess.run(["git", "clone", "--depth", "1", URL_REPO, "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4", "lap"], check=True)
sys.path.insert(0, "/tmp/repo/src")
print("commit:", subprocess.run(["git", "-C", "/tmp/repo", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip(), flush=True)
if shutil.which("ffprobe") is None:  # il lettore video usa ffmpeg/ffprobe: build statica se l'immagine non li ha
    subprocess.run("curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | "
                   "tar -xJ -C /tmp && cp /tmp/ffmpeg-*-static/ff* /usr/local/bin/", shell=True, check=True)
# =====

"""Solo CPU: dopo un addestramento fatto fuori da Kaggle (gpu_esterna/addestra.sh → dataset
`roma-safe-cctv-addestramento-2`), il sistema intero sul test (modello + WARNING ARMA tarato + regola D6/D42) e
l'analisi per sorgente e per video. Le timeline stanno qui (aggiungi-armi + estrazione-normali), le finestre, lo split
e la taratura vengono dal kernel `roma-safe-cctv-finestre`."""
import json, shutil, subprocess, sys, zipfile
from pathlib import Path

ambiente = {**os.environ, "PYTHONPATH": "/tmp/repo/src"}
ingresso, uscita = Path("/kaggle/input"), Path("/kaggle/working")
finestre = next(ingresso.glob("**/finestre.npz"))
split = next(ingresso.glob("**/split.npz"))
taratura = next(ingresso.glob("**/taratura_warning.json"))
cartelle_estratti = [str(c) for c in ingresso.glob("**/estratti") if c.is_dir()]
modelli = {}
for modello in ("stgcn", "graph_transformer"):
    cartella = next((p.parent for p in ingresso.glob(f"**/{modello}/probabilita_test.npy")), None)
    archivio = next(ingresso.glob(f"**/{modello}.zip"), None)
    if cartella is None and archivio is not None:   # dataset caricato con --dir-mode zip e non decompresso
        zipfile.ZipFile(archivio).extractall(Path("/tmp") / modello)
        cartella = next((p.parent for p in (Path("/tmp") / modello).glob("**/probabilita_test.npy")), None)
    if cartella is not None:
        modelli[modello] = cartella
        shutil.copytree(cartella, uscita / modello, dirs_exist_ok=True)
print("finestre:", finestre, "| modelli:", modelli, flush=True)
shutil.copy2(taratura, uscita / "taratura_warning.json")
for modello, cartella in modelli.items():
    subprocess.run([sys.executable, "-m", "sicurezza.demo", "valuta-sistema", "--finestre", str(finestre),
                    "--split", str(split), "--probabilita", str(cartella / "probabilita_test.npy"),
                    "--taratura", str(taratura), "--estratti", *cartelle_estratti,
                    "--uscita", str(uscita / modello / "sistema_test.json")], check=True, env=ambiente)
subprocess.run([sys.executable, "-m", "sicurezza.modelli", "analizza-predizioni", "--finestre", str(finestre),
                "--probabilita", *[f"{m}={c / 'probabilita_test.npy'}" for m, c in modelli.items()],
                "--uscita", str(uscita / "analisi_test.json")], check=True, env=ambiente)
for modello in modelli:
    print(modello, json.loads((uscita / modello / "sistema_test.json").read_text()), flush=True)
