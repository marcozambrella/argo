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

"""Passo 2 del piano: velocità reale su T4. v3 (D18): confronta FP32, FP16 e 2 processi per GPU."""
import json, os, subprocess, sys, time
from pathlib import Path
from sicurezza.estrazione import Estrattore, VideoDaEstrarre, scrivi_elenco
from sicurezza.video import leggi_frame

video = sorted(Path("/kaggle/input").glob("**/RWF-2000/val/*/*.avi"))[:48]
risultati = {}
inizio, frame = time.perf_counter(), 0
for percorso in video[:12]:
    frame += sum(1 for _ in leggi_frame(str(percorso)))
risultati["solo_decodifica_fps"] = frame / (time.perf_counter() - inizio)

# 1. un processo su una GPU, pose + detector (yolo26s generico al posto di quello armi), FP32 e FP16
for nome, mezza in [("un_processo_fp32", False), ("un_processo_fp16", True)]:
    estrattore = Estrattore("yolo26s-pose.pt", "yolo26s.pt", dispositivo=0, mezza_precisione=mezza)
    estrattore.estrai_video(VideoDaEstrarre("riscaldamento", str(video[0])))
    inizio, frame = time.perf_counter(), 0
    for percorso in video[:12]:
        frame += int(estrattore.estrai_video(VideoDaEstrarre(percorso.stem, str(percorso)))["numero_frame"])
    risultati[f"{nome}_fps"] = frame / (time.perf_counter() - inizio)
    print(nome, risultati[f"{nome}_fps"], flush=True)

# 2. throughput totale con N processi FP16 in parallelo sulle 2 GPU (come farà il kernel di estrazione)
elenco = Path("/tmp/elenco_bench.jsonl")
scrivi_elenco([VideoDaEstrarre(f"bench/{p.stem}", str(p)) for p in video], elenco)
ambiente = {**os.environ, "PYTHONPATH": "/tmp/repo/src"}
for processi_per_gpu in (1, 2, 3):
    uscita = Path(f"/tmp/bench_{processi_per_gpu}")
    parti = 2 * processi_per_gpu
    inizio = time.perf_counter()
    lanciati = [subprocess.Popen([sys.executable, "-m", "sicurezza.estrazione", "--elenco", str(elenco), "--uscita",
                                  str(uscita), "--parte", str(parte), "--parti", str(parti), "--dispositivo",
                                  str(parte % 2), "--modello-armi", "yolo26s.pt", "--mezza-precisione"], env=ambiente,
                                 stdout=subprocess.DEVNULL) for parte in range(parti)]
    for processo in lanciati:
        processo.wait()
    secondi = time.perf_counter() - inizio
    import numpy as np
    frame = sum(int(np.load(f)["numero_frame"]) for f in uscita.glob("*.npz"))
    risultati[f"fp16_{processi_per_gpu}_processi_per_gpu_fps_totali"] = frame / secondi
    print(processi_per_gpu, "processi per GPU:", risultati[f"fp16_{processi_per_gpu}_processi_per_gpu_fps_totali"], flush=True)
print(json.dumps(risultati, indent=2))
Path("/kaggle/working/benchmark.json").write_text(json.dumps(risultati, indent=2))
