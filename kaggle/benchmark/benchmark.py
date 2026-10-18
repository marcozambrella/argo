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

"""Passo 2 del piano, v4: profilo delle fasi su T4 (la v3 ha mostrato che il limite è la CPU, non la GPU)."""
import json, os, subprocess, sys, time
from pathlib import Path
import cv2, numpy as np, ultralytics
from ultralytics import YOLO
from sicurezza.video import leggi_frame

video = sorted(Path("/kaggle/input").glob("**/RWF-2000/val/Fight/*.avi"))[:6]
frame_pieni = [f for percorso in video for f in leggi_frame(str(percorso))][:600]
frame_640 = [cv2.resize(f, (640, int(f.shape[0] * 640 / f.shape[1]))) for f in frame_pieni]
cartella_tracker = Path(ultralytics.__file__).parent / "cfg" / "trackers"
senza_gmc = Path("/tmp/botsort_senza_gmc.yaml")
senza_gmc.write_text((cartella_tracker / "botsort.yaml").read_text().replace("gmc_method: sparseOptFlow", "gmc_method: none"))
risultati = {"cpu": os.cpu_count(), "frame": len(frame_pieni), "risoluzione": list(frame_pieni[0].shape[:2])}

def ms_per_frame(funzione, frame):
    funzione(frame[:5])
    inizio = time.perf_counter(); funzione(frame); return (time.perf_counter() - inizio) / len(frame) * 1000

inizio = time.perf_counter()
for percorso in video:
    for _ in leggi_frame(str(percorso)): pass
risultati["decodifica_ms"] = (time.perf_counter() - inizio) / len(frame_pieni) * 1000

pose = YOLO("yolo26s-pose.pt"); armi = YOLO("yolo26s.pt")
def predici(modello, lotto=1):
    def esegui(frame):
        for indice in range(0, len(frame), lotto):
            modello.predict(frame[indice:indice + lotto], device=0, verbose=False)
    return esegui
def traccia(tracker):
    def esegui(frame):
        for indice, f in enumerate(frame):
            pose.track(f, persist=indice > 0, tracker=tracker, device=0, conf=0.1, verbose=False)
    return esegui

risultati["pose_predict_pieni_ms"] = ms_per_frame(predici(pose), frame_pieni)
risultati["pose_predict_640_ms"] = ms_per_frame(predici(pose), frame_640)
risultati["pose_predict_lotto8_pieni_ms"] = ms_per_frame(predici(pose, 8), frame_pieni)
risultati["pose_track_gmc_pieni_ms"] = ms_per_frame(traccia("botsort.yaml"), frame_pieni)
risultati["pose_track_senza_gmc_pieni_ms"] = ms_per_frame(traccia(str(senza_gmc)), frame_pieni)
risultati["pose_track_gmc_640_ms"] = ms_per_frame(traccia("botsort.yaml"), frame_640)
risultati["pose_track_senza_gmc_640_ms"] = ms_per_frame(traccia(str(senza_gmc)), frame_640)
risultati["armi_predict_pieni_ms"] = ms_per_frame(predici(armi), frame_pieni)
risultati["armi_predict_lotto8_pieni_ms"] = ms_per_frame(predici(armi, 8), frame_pieni)
print(json.dumps(risultati, indent=2))
Path("/kaggle/working/benchmark.json").write_text(json.dumps(risultati, indent=2))
