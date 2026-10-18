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

"""Passo 2 del piano, v5: pipeline nuova (D20 lotti + D21 TrackTrack, GMC spenta per telecamere fisse)."""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from sicurezza.estrazione import VideoDaEstrarre, scrivi_elenco

video = sorted(Path("/kaggle/input").glob("**/RWF-2000/val/*/*.avi"))[:48]
ambiente = {**os.environ, "PYTHONPATH": "/tmp/repo/src"}
risultati = {"video": len(video)}

def misura(nome, processi_per_gpu, telecamera_mobile):
    elenco = Path(f"/tmp/elenco_{nome}.jsonl")
    scrivi_elenco([VideoDaEstrarre(f"{nome}/{p.stem}", str(p), telecamera_mobile=telecamera_mobile) for p in video], elenco)
    uscita, parti = Path(f"/tmp/{nome}"), 2 * processi_per_gpu
    inizio = time.perf_counter()
    lanciati = [subprocess.Popen([sys.executable, "-m", "sicurezza.estrazione", "--elenco", str(elenco), "--uscita",
                                  str(uscita), "--parte", str(parte), "--parti", str(parti), "--dispositivo",
                                  str(parte % 2), "--modello-armi", "yolo26s.pt"], env=ambiente,
                                 stdout=subprocess.DEVNULL) for parte in range(parti)]
    for processo in lanciati:
        processo.wait()
    secondi = time.perf_counter() - inizio
    frame = sum(int(np.load(f)["numero_frame"]) for f in uscita.glob("*.npz"))
    risultati[nome] = round(frame / secondi, 1)
    print(nome, risultati[nome], "fps totali", flush=True)

subprocess.run([sys.executable, "-c", "from ultralytics import YOLO; YOLO('yolo26s-pose.pt'); YOLO('yolo26s.pt')"], env=ambiente)
misura("lotti_tracktrack_senza_gmc_1_per_gpu", 1, False)
misura("lotti_tracktrack_senza_gmc_2_per_gpu", 2, False)
misura("lotti_tracktrack_con_gmc_2_per_gpu", 2, True)
print(json.dumps(risultati, indent=2))
Path("/kaggle/working/benchmark.json").write_text(json.dumps(risultati, indent=2))
