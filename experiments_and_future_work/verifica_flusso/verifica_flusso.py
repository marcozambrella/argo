# ===== avvio comune: uguale in tutti i kernel di training/ =====
# Kaggle carica un solo file per kernel, quindi questo blocco è copiato in testa a ognuno.
import os
import shutil
import subprocess
import sys
URL_REPO = "https://github.com/marcozambrella/argo"
# D71: il branch flusso-veloce (il flusso nuovo da verificare prima di main)
subprocess.run(["git", "clone", "--depth", "1", "--branch", "flusso-veloce", URL_REPO, "/tmp/repo"], check=True)
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

sys.stdout.reconfigure(line_buffering=True)

"""D71, CPU: verifica del flusso veloce (pose a 480 px un frame su due, niente detector) con la vera AnalisiFlusso
su clip di val e test v3 mai viste (Marco: "un po' più di 60, così diversifichi"): CLIP_PER_CLASSE per classe
(la più grave tra le sue finestre) e per split, alternando le sorgenti, al più 30 s ciascuna. Il flusso gira con
il solo ST-GCN consegnato (pesi/stgcn.pt della Release v1.1): b1 e b2 non cambiano rispetto a D70 e i loro punteggi
ci sono già, la fusione con b1 si fa dopo in locale. Le clip sono divise in PARTI kernel (stesso seme in tutti)."""
import json
import time
from pathlib import Path
import numpy as np
from sicurezza.dati import FPS_ANALISI, INDICE_CLASSE, leggi_frame, prepara
from sicurezza.demo import AnalisiFlusso

PARTE = 0
PARTI = 4
CLIP_PER_CLASSE = 16
FRAME_MASSIMI = 750   # 30 s a 25 fps
SORGENTI = ["rwf2000", "cctv_fights", "scvd", "ucf_uca", "chain_snatching"]   # scippi_kaggle: solo negli esperimenti
URL_STGCN = "https://github.com/marcozambrella/argo/releases/download/v1.1/stgcn.pt"
ingresso = Path("/kaggle/input")
uscita = Path("/kaggle/working")

subprocess.run(["curl", "-sL", "-o", "/tmp/stgcn.pt", URL_STGCN], check=True)
clip_annotate = {}
for clip in prepara(ingresso, Path("/tmp/elenchi"), SORGENTI):
    clip_annotate[clip.id_clip] = clip

percorso_finestre = next(ingresso.glob("**/finestre.npz"))
with np.load(percorso_finestre) as finestre:
    id_clip = finestre["id_clip"]
    etichette = finestre["etichetta"]
    sorgenti = finestre["sorgente"]
split = np.load(percorso_finestre.parent / "split.npz")
gravita = {INDICE_CLASSE["normale"]: 0, INDICE_CLASSE["violenza"]: 1, INDICE_CLASSE["scippo"]: 2,
           INDICE_CLASSE["violenza_con_arma"]: 3}
generatore = np.random.default_rng(0)
scelte = []
for nome_split in ("val", "test"):
    classe_della_clip = {}
    sorgente_della_clip = {}
    for riga in split[nome_split]:
        clip = str(id_clip[riga])
        if clip not in clip_annotate:
            continue
        valore = gravita[int(etichette[riga])]
        if clip not in classe_della_clip or valore > classe_della_clip[clip]:
            classe_della_clip[clip] = valore
        sorgente_della_clip[clip] = str(sorgenti[riga])
    for valore in range(4):
        # per ogni sorgente le sue clip in ordine casuale, poi una per sorgente a turno
        per_sorgente = {}
        for clip in sorted(classe_della_clip):
            if classe_della_clip[clip] != valore:
                continue
            sorgente = sorgente_della_clip[clip]
            if sorgente not in per_sorgente:
                per_sorgente[sorgente] = []
            per_sorgente[sorgente].append(clip)
        code = []
        for sorgente in sorted(per_sorgente):
            code.append(list(generatore.permutation(per_sorgente[sorgente])))
        rimaste = 0
        for coda in code:
            rimaste += len(coda)
        prese = []
        while len(prese) < CLIP_PER_CLASSE and rimaste > 0:
            for coda in code:
                if len(coda) > 0 and len(prese) < CLIP_PER_CLASSE:
                    prese.append(str(coda.pop(0)))
                    rimaste -= 1
        for clip in prese:
            scelte.append((nome_split, clip))
mie = scelte[PARTE::PARTI]
print(f"parte {PARTE}/{PARTI}: {len(mie)} clip delle {len(scelte)} scelte", flush=True)

campi = {"split": [], "id_clip": [], "inizio_finestra": [], "probabilita": []}
secondi_calcolo = 0.0
secondi_video = 0.0
for numero, (nome_split, clip) in enumerate(mie):
    video = clip_annotate[clip]
    analisi = AnalisiFlusso("/tmp/stgcn.pt", "yolo26s-pose.pt", "cpu")
    inizio = time.perf_counter()
    frame_letti = 0
    for frame in leggi_frame(video.percorso_video, FPS_ANALISI, video.inizio_secondi, video.fine_secondi):
        if frame_letti >= FRAME_MASSIMI:
            break
        _, _, _, evento = analisi.analizza(frame)
        frame_letti += 1
        if evento is not None:
            campi["split"].append(nome_split)
            campi["id_clip"].append(clip)
            campi["inizio_finestra"].append(int(round(evento["inizio_secondi"] * FPS_ANALISI)))
            campi["probabilita"].append(analisi.decisione["probabilita"])   # ST-GCN senza arrotondare
    secondi_calcolo += time.perf_counter() - inizio
    secondi_video += frame_letti / FPS_ANALISI
    print(f"{numero + 1}/{len(mie)} {nome_split} {clip}: {frame_letti} frame", flush=True)

np.savez_compressed(uscita / f"verifica_{PARTE}.npz", split=np.array(campi["split"]),
                    id_clip=np.array(campi["id_clip"]), inizio_finestra=np.array(campi["inizio_finestra"]),
                    probabilita=np.stack(campi["probabilita"]).astype(np.float32))
tempi = {"parte": PARTE, "clip": len(mie), "secondi_video": secondi_video, "secondi_calcolo": secondi_calcolo,
         "secondi_per_secondo_di_video": secondi_calcolo / max(1.0, secondi_video)}
(uscita / f"tempi_{PARTE}.json").write_text(json.dumps(tempi, indent=1))
print(tempi, flush=True)
print("fatto", flush=True)
