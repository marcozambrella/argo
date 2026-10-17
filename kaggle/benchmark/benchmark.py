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

"""Passo 2 del piano: velocità reale su T4 per aggiornare le stime nel diario."""
import json, time
from pathlib import Path
from sicurezza.estrazione import Estrattore, VideoDaEstrarre
from sicurezza.video import leggi_frame

video = sorted(Path("/kaggle/input").glob("**/RWF-2000/val/*/*.avi"))[:12]
risultati = {}
inizio, frame = time.perf_counter(), 0
for percorso in video:
    frame += sum(1 for _ in leggi_frame(str(percorso)))
risultati["solo_decodifica_fps"] = frame / (time.perf_counter() - inizio)

for nome, armi in [("pose_s_track", None), ("pose_s_track_piu_detector_s", "yolo26s.pt")]:
    estrattore = Estrattore("yolo26s-pose.pt", armi, dispositivo=0)
    estrattore.estrai_video(VideoDaEstrarre("riscaldamento", str(video[0])))
    inizio, frame = time.perf_counter(), 0
    for percorso in video:
        frame += int(estrattore.estrai_video(VideoDaEstrarre(percorso.stem, str(percorso)))["numero_frame"])
    risultati[f"{nome}_fps_una_gpu"] = frame / (time.perf_counter() - inizio)
print(json.dumps(risultati, indent=2))
Path("/kaggle/working/benchmark.json").write_text(json.dumps(risultati, indent=2))
