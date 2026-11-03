# ===== avvio comune: uguale in tutti i kernel di training/ =====
# Kaggle carica un solo file per kernel, quindi questo blocco è copiato in testa a ognuno.
import os
import shutil
import subprocess
import sys
URL_REPO = "https://github.com/marcozambrella/argo"
# notte 03→04/11: il branch notte-tempi (esperimenti D68/D69 + D70)
subprocess.run(["git", "clone", "--depth", "1", "--branch", "notte-tempi", URL_REPO, "/tmp/repo"], check=True)
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

"""Notte 03→04/11, E3, CPU (Marco: diminuire i tempi di pose e detector su un PC senza GPU). Su 60 clip del test v3
(15 per classe: la classe più grave tra le sue finestre), al più 20 s ciascuna, si rifanno pose + tracking +
detector in una VARIANTE (modello della pose, lato dell'immagine, un frame su uno o su due) e si salvano le finestre
come l'estrazione vera (costruisci_finestra), più i millisecondi per frame di pose e detector. Le probabilità degli
scheletri v3 si calcolano dopo, in locale, sulle stesse finestre di tutte le varianti. Con un frame su due il tracker
vede solo i frame pari e i dispari sono copie (in locale `pose_a_meta_frame` li rifà per interpolazione)."""
import json
import time
from pathlib import Path
import numpy as np
from sicurezza.dati import INDICE_CLASSE, prepara, leggi_frame, leggi_info_video
from sicurezza.modelli import costruisci_finestra, timeline_da_dati
from sicurezza.visione import Estrattore, impacchetta_estrazione

VARIANTE = "rif"
VARIANTI = {"rif": ("yolo26s-pose.pt", 640, 1), "s640_12": ("yolo26s-pose.pt", 640, 2),
            "s480": ("yolo26s-pose.pt", 480, 1), "s320": ("yolo26s-pose.pt", 320, 1),
            "n640": ("yolo26n-pose.pt", 640, 1), "s480_12": ("yolo26s-pose.pt", 480, 2)}
CLIP_PER_CLASSE = 15
FRAME_MASSIMI = 500   # 20 s a 25 fps
SORGENTI = ["rwf2000", "cctv_fights", "scvd", "ucf_uca", "chain_snatching", "scippi_kaggle"]
URL_DETECTOR = "https://github.com/marcozambrella/argo/releases/download/v1.1/armi_v3.pt"
ingresso = Path("/kaggle/input")
uscita = Path("/kaggle/working")

modello_pose, lato, passo = VARIANTI[VARIANTE]
subprocess.run(["curl", "-sL", "-o", "/tmp/armi_v3.pt", URL_DETECTOR], check=True)
clip_annotate = {}
for clip in prepara(ingresso, Path("/tmp/elenchi"), SORGENTI):
    clip_annotate[clip.id_clip] = clip

# le clip del test v3, 15 per classe (seme fisso: tutte le varianti hanno le stesse)
percorso_finestre = next(ingresso.glob("**/finestre.npz"))
with np.load(percorso_finestre) as finestre:
    id_clip = finestre["id_clip"]
    etichette = finestre["etichetta"]
test = np.load(percorso_finestre.parent / "split.npz")["test"]
gravita = {INDICE_CLASSE["normale"]: 0, INDICE_CLASSE["violenza"]: 1, INDICE_CLASSE["scippo"]: 2,
           INDICE_CLASSE["violenza_con_arma"]: 3}
classe_della_clip = {}
for riga in test:
    clip = str(id_clip[riga])
    if clip not in clip_annotate:
        continue
    valore = gravita[int(etichette[riga])]
    if clip not in classe_della_clip or valore > classe_della_clip[clip]:
        classe_della_clip[clip] = valore
generatore = np.random.default_rng(0)
scelte = []
for valore in range(4):
    candidate = []
    for clip, classe in classe_della_clip.items():
        if classe == valore:
            candidate.append(clip)
    candidate.sort()
    quante = min(CLIP_PER_CLASSE, len(candidate))
    scelte += list(generatore.choice(candidate, quante, replace=False))
print(f"variante {VARIANTE}: {len(scelte)} clip", flush=True)

estrattore = Estrattore(modello_pose, "/tmp/armi_v3.pt", "cpu")
estrattore.modello_pose.overrides["imgsz"] = lato
campi = {"id_clip": [], "inizio_finestra": [], "nodi": [], "maschera": [], "classe_arma": []}
secondi_pose = 0.0
secondi_armi = 0.0
frame_analizzati = 0
for numero, clip in enumerate(scelte):
    video = clip_annotate[clip]
    info = leggi_info_video(video.percorso_video)
    estrattore.nuovo_video()
    righe_persone = []
    righe_armi = []
    precedente = None
    for indice, frame in enumerate(leggi_frame(video.percorso_video, 25, video.inizio_secondi, video.fine_secondi)):
        if indice >= FRAME_MASSIMI:
            break
        if indice % passo == 0 and numero < 3:
            # sulle prime 3 clip pose e detector cronometrati a parte (rifà il calcolo: solo per misurare)
            inizio = time.perf_counter()
            estrattore.modello_pose.predict([frame], device="cpu", conf=0.1, verbose=False)
            secondi_pose += time.perf_counter() - inizio
            inizio = time.perf_counter()
            estrattore.modello_armi.predict([frame], device="cpu", conf=0.25, verbose=False)
            secondi_armi += time.perf_counter() - inizio
            frame_analizzati += 1
        if indice % passo == 0:
            precedente = estrattore.analizza_lotto([frame])[0]   # questo dà il risultato vero, con il tracker
        righe_persone.append((indice, precedente[0]))
        righe_armi.append((indice, precedente[1]))
    dati = impacchetta_estrazione(righe_persone, righe_armi, len(righe_persone), info.larghezza, info.altezza, 0.0)
    timeline = timeline_da_dati(dati)
    for inizio_finestra in range(0, len(righe_persone) - 40 + 1, 10):
        nodi, maschera, classe_arma = costruisci_finestra(timeline, inizio_finestra)
        campi["id_clip"].append(clip)
        campi["inizio_finestra"].append(inizio_finestra)
        campi["nodi"].append(nodi.astype(np.float16))
        campi["maschera"].append(maschera)
        campi["classe_arma"].append(classe_arma)
    print(f"{numero + 1}/{len(scelte)} {clip}: {len(righe_persone)} frame", flush=True)

np.savez_compressed(uscita / f"finestre_{VARIANTE}.npz", id_clip=np.array(campi["id_clip"]),
                    inizio_finestra=np.array(campi["inizio_finestra"]), nodi=np.stack(campi["nodi"]),
                    maschera=np.stack(campi["maschera"]), classe_arma=np.stack(campi["classe_arma"]))
tempi = {"variante": VARIANTE,
         "frame_cronometrati": frame_analizzati, "ms_pose_per_frame_analizzato": 1000 * secondi_pose / max(1, frame_analizzati),
         "ms_armi_per_frame_analizzato": 1000 * secondi_armi / max(1, frame_analizzati), "passo": passo}
(uscita / f"tempi_{VARIANTE}.json").write_text(json.dumps(tempi, indent=1))
print(tempi, flush=True)
print("fatto", flush=True)
