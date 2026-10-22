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

"""Solo CPU, ESPLORATIVO (DA DECIDERE 23 c, seguito di `prova-rgb`): vettori R(2+1)D-18 congelato per TUTTE le
finestre di clip intere di validation e test (split dell'addestramento 2), così in locale si può misurare la fusione
pixel + scheletri anche con la media mobile e la regola d'allarme sui pezzi da 10/30 s (servono finestre consecutive).
Il test usa solo clip mai viste dall'addestramento 1 (elenco di `roma-safe-cctv-analisi`). Ogni clip si decodifica
una volta sola; le finestre (40 frame, passo 10) sono quelle di finestre.npz.
Uscite: rgb_clip_test.npz, rgb_clip_val.npz (indici delle finestre, vettori, id_clip, inizio, etichetta, sorgente)."""
import json, time
from pathlib import Path
import numpy as np
import torch, torchvision
from sicurezza.dati import prepara

torch.set_num_threads(os.cpu_count())
ingresso, uscita = Path("/kaggle/input"), Path("/kaggle/working")
FINESTRE_PER_SORGENTE = {"cctv_fights": 3000, "rwf2000": 800, "ucf_uca": 700, "scvd": 300}
SECONDI_MASSIMI = 10.5 * 3600
generatore = np.random.default_rng(1)

clip_annotate = {clip.id_clip: clip for clip in prepara(ingresso, Path("/tmp/elenchi"), tuple(FINESTRE_PER_SORGENTE))}
percorso_finestre = next(ingresso.glob("**/finestre.npz"))
with np.load(percorso_finestre) as finestre:
    id_clip, inizi, etichette, sorgenti = (finestre[k] for k in ("id_clip", "inizio_finestra", "etichetta", "sorgente"))
split = np.load(percorso_finestre.parent / "split.npz")
analisi_v1 = json.loads(next(ingresso.glob("**/analisi_test.json")).read_text())
mai_viste_dal_modello_1 = {clip["id_clip"] for clip in analisi_v1["modelli"]["stgcn"]["per_clip"]}


def scegli_clip(indici_split: np.ndarray, ammesse=None) -> list[str]:
    """Clip intere a caso, per sorgente, finché non si arriva al numero di finestre previsto."""
    scelte = []
    for sorgente, budget in FINESTRE_PER_SORGENTE.items():
        righe = indici_split[sorgenti[indici_split] == sorgente]
        candidate = [c for c in np.unique(id_clip[righe]) if c in clip_annotate and (ammesse is None or c in ammesse)]
        totale = 0
        for clip in generatore.permutation(candidate):
            if totale >= budget:
                break
            scelte.append(str(clip))
            totale += int((id_clip[righe] == clip).sum())
    return scelte


modello = torchvision.models.video.r2plus1d_18(weights=torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1)
modello.fc = torch.nn.Identity()
modello.eval()
MEDIA = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
DEVIAZIONE = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)
ALTEZZA, LARGHEZZA, LOTTO = 128, 171, 8


def frame_clip(clip, frame_necessari: int) -> np.ndarray:
    """Tutta la clip a 25 fps (come il lettore del progetto), ridotta a 171×128, RGB."""
    comando = ["ffmpeg", "-v", "error", "-ss", f"{clip.inizio_secondi or 0:.3f}", "-i", clip.percorso_video,
               "-t", f"{frame_necessari / 25 + 0.2:.3f}", "-vf", f"fps=25,scale={LARGHEZZA}:{ALTEZZA}",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    grezzi = np.frombuffer(subprocess.run(comando, capture_output=True).stdout, np.uint8)
    numero = grezzi.size // (ALTEZZA * LARGHEZZA * 3)
    return grezzi[: numero * ALTEZZA * LARGHEZZA * 3].reshape(numero, ALTEZZA, LARGHEZZA, 3)


@torch.no_grad()
def vettori_clip(frame: np.ndarray, inizi_finestre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tenute, lotti = [], []
    for inizio in inizi_finestre:
        finestra = frame[inizio: inizio + 40]
        if len(finestra) < 16:
            continue
        scelti = finestra[np.linspace(0, len(finestra) - 1, 16).round().astype(int), 8:120, 29:141]
        lotti.append(torch.from_numpy(scelti.copy()).permute(3, 0, 1, 2))
        tenute.append(inizio)
    if not lotti:
        return np.zeros(0, bool), np.zeros((0, 512), np.float32)
    vettori = [modello((torch.stack(lotti[i: i + LOTTO]).float() / 255 - MEDIA) / DEVIAZIONE).numpy()
               for i in range(0, len(lotti), LOTTO)]
    return np.isin(inizi_finestre, tenute), np.concatenate(vettori)


avvio = time.time()
for nome, indici_split, ammesse in (("test", split["test"], mai_viste_dal_modello_1), ("val", split["val"], None)):
    clip_scelte = scegli_clip(indici_split, ammesse)
    print(f"{nome}: {len(clip_scelte)} clip", flush=True)
    tutte_le_righe, tutti_i_vettori = [], []
    for numero, clip in enumerate(clip_scelte):
        if time.time() - avvio > SECONDI_MASSIMI:
            print("tempo finito", flush=True)
            break
        righe = indici_split[id_clip[indici_split] == clip]
        righe = righe[np.argsort(inizi[righe])]
        frame = frame_clip(clip_annotate[clip], int(inizi[righe].max()) + 40)
        tenute, vettori = vettori_clip(frame, inizi[righe])
        tutte_le_righe.append(righe[tenute])
        tutti_i_vettori.append(vettori)
        if numero % 20 == 0:
            print(f"{nome} clip {numero}/{len(clip_scelte)}, finestre {sum(map(len, tutte_le_righe))}, "
                  f"{time.time() - avvio:.0f} s", flush=True)
    righe = np.concatenate(tutte_le_righe) if tutte_le_righe else np.zeros(0, np.int64)
    np.savez(uscita / f"rgb_clip_{nome}.npz", indici=righe,
             vettori=np.concatenate(tutti_i_vettori) if tutti_i_vettori else np.zeros((0, 512), np.float32),
             id_clip=id_clip[righe], inizio_finestra=inizi[righe], etichetta=etichette[righe], sorgente=sorgenti[righe])
    print(f"{nome}: {len(righe)} finestre salvate, {time.time() - avvio:.0f} s", flush=True)
