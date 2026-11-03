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

"""Notte 03→04/11, E2, CPU: la rete violenza del sistema consegnato (rete_violenza.pt + testa_violenza.npz della
Release) sulle finestre v3 di uno SPLIT, una ogni quattro (inizio multiplo di 40 frame, la griglia dei confronti).
Serve a misurare se la violenza di b1 (che con D70 gira già ogni 0,8 s) può prendere il posto della rete violenza
(ogni 0,4 s, ~2,3 s di calcolo per secondo di video). Uscita: violenza_pixel_<split>.npz per `salva_punteggi`."""
from pathlib import Path
import numpy as np
from sicurezza.dati import prepara_video_esperimenti
from sicurezza.demo import RamoPixel, salva_punteggi

SPLIT = "val"
URL_PESI = "https://github.com/marcozambrella/argo/releases/download/v1.1/"
FRAME_PER_BLOCCO = 600
ingresso = Path("/kaggle/input")
uscita = Path("/kaggle/working")
for nome in ("rete_violenza.pt", "testa_violenza.npz"):
    subprocess.run(["curl", "-sL", "-o", f"/tmp/{nome}", URL_PESI + nome], check=True)
ramo = RamoPixel("/tmp/testa_violenza.npz", "cpu", "/tmp/rete_violenza.pt")

clip_annotate = prepara_video_esperimenti(ingresso, Path("/tmp/dati"))
percorso_finestre = next(ingresso.glob("**/finestre.npz"))
with np.load(percorso_finestre) as finestre:
    id_clip = finestre["id_clip"]
    inizi = finestre["inizio_finestra"]
    etichette = finestre["etichetta"]
    sorgenti = finestre["sorgente"]
indici_split = np.load(percorso_finestre.parent / "split.npz")[SPLIT]
indici = []
for riga in indici_split:
    if inizi[riga] % 40 == 0 and str(id_clip[riga]) in clip_annotate:
        indici.append(riga)
indici = np.array(indici)
print(f"{SPLIT}: {len(indici)} finestre", flush=True)


def frame_del_blocco(clip: str, primo_frame: int) -> np.ndarray:
    """FRAME_PER_BLOCCO frame della clip a 25 fps, 171×128 RGB come `riduci_per_pixel` (bicubico)."""
    annotata = clip_annotate[clip]
    secondi = primo_frame / 25
    if annotata.inizio_secondi is not None:
        secondi += annotata.inizio_secondi
    comando = ["ffmpeg", "-v", "error", "-ss", f"{secondi:.3f}", "-i", annotata.percorso_video,
               "-t", f"{FRAME_PER_BLOCCO / 25 + 0.2:.3f}", "-vf", "fps=25,scale=171:128",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    grezzi = np.frombuffer(subprocess.run(comando, capture_output=True).stdout, np.uint8)
    byte_per_frame = 128 * 171 * 3
    frame_completi = grezzi.size // byte_per_frame
    return grezzi[: frame_completi * byte_per_frame].reshape(-1, 128, 171, 3)


righe_fatte = []
probabilita = []
clip_fatte = 0
for clip in np.unique(id_clip[indici]):
    righe = indici[id_clip[indici] == clip]
    righe = righe[np.argsort(inizi[righe])]
    numero_blocco = inizi[righe] // (FRAME_PER_BLOCCO - 40)
    for valore in np.unique(numero_blocco):
        primo_frame = int(valore) * (FRAME_PER_BLOCCO - 40)
        frame = frame_del_blocco(str(clip), primo_frame)
        for riga in righe[numero_blocco == valore]:
            inizio = int(inizi[riga]) - primo_frame
            pezzo = frame[inizio: inizio + 40]
            if len(pezzo) >= 16:
                righe_fatte.append(riga)
                probabilita.append(ramo.probabilita_violenza(list(pezzo)))
    clip_fatte += 1
    if clip_fatte % 20 == 0:
        print(f"{clip_fatte} clip, {len(righe_fatte)} finestre", flush=True)
righe_fatte = np.array(righe_fatte)
finestre_scelte = {"id_clip": id_clip[righe_fatte], "inizio_finestra": inizi[righe_fatte],
                   "etichetta": etichette[righe_fatte], "sorgente": sorgenti[righe_fatte]}
salva_punteggi(uscita / f"violenza_pixel_{SPLIT}.npz", finestre_scelte, {"violenza": np.array(probabilita)})
print("fatto", flush=True)
