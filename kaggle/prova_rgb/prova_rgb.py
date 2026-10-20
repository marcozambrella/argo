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

"""Solo CPU, ESPLORATIVO (DA DECIDERE 23 c, 20/10): quanto vale un modello video preaddestrato che guarda i PIXEL
invece degli scheletri? R(2+1)D-18 di torchvision (Kinetics-400) congelato → vettore da 512 per finestra di 40 frame
(16 frame campionati, ritaglio centrale 112×112) → regressione logistica violenza sì/no.
Stesse finestre e stesso split dell'addestramento 2 (RWF-2000, CCTV-Fights, SCVD, UCF; niente scippi e folle).
Non tocca il sistema consegnato: serve solo a capire se un ramo RGB porterebbe la violenza sopra l'80%.
Uscite: rgb_train.npz / rgb_test.npz (indici delle finestre, vettori, ...), rgb_probabilita_test.npy, prova_rgb.json."""
import json, time
from pathlib import Path
import numpy as np
import torch, torchvision
from sklearn.linear_model import LogisticRegression
from sicurezza.dati import INDICE_CLASSE, prepara

torch.set_num_threads(os.cpu_count())
ingresso, uscita = Path("/kaggle/input"), Path("/kaggle/working")
FINESTRE_TRAIN_PER_CLASSE, FINESTRE_TEST, SECONDI_MASSIMI = 1500, 4000, 10.5 * 3600
SORGENTI = ("rwf2000", "cctv_fights", "scvd", "ucf_uca")
generatore = np.random.default_rng(0)

clip_annotate = {clip.id_clip: clip for clip in prepara(ingresso, Path("/tmp/elenchi"), SORGENTI)}
percorso_finestre = next(ingresso.glob("**/finestre.npz"))
with np.load(percorso_finestre) as finestre:
    id_clip, inizi, etichette, sorgenti = (finestre[k] for k in ("id_clip", "inizio_finestra", "etichetta", "sorgente"))
split = np.load(percorso_finestre.parent / "split.npz")
violenta = np.isin(etichette, [INDICE_CLASSE["violenza"], INDICE_CLASSE["violenza_con_arma"]])
utilizzabile = np.where(np.array([clip in clip_annotate for clip in id_clip]))[0]
print(len(clip_annotate), "clip nei dataset,", len(utilizzabile), "finestre utilizzabili", flush=True)
train = np.intersect1d(split["train"], utilizzabile)
scelti_train = np.concatenate([generatore.choice(train[violenta[train] == valore], FINESTRE_TRAIN_PER_CLASSE,
                                                 replace=False) for valore in (True, False)])
test = np.intersect1d(split["test"], utilizzabile)
scelti_test = np.sort(generatore.choice(test, min(FINESTRE_TEST, len(test)), replace=False))

modello = torchvision.models.video.r2plus1d_18(weights=torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1)
modello.fc = torch.nn.Identity()
modello.eval()
MEDIA = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
DEVIAZIONE = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
ALTEZZA, LARGHEZZA = 128, 171


def frame_finestra(indice: int) -> np.ndarray | None:
    """I 40 frame della finestra a 25 fps (come il lettore del progetto), ridotti a 171×128, RGB."""
    clip = clip_annotate[id_clip[indice]]
    secondi = (clip.inizio_secondi or 0) + int(inizi[indice]) / 25
    comando = ["ffmpeg", "-v", "error", "-ss", f"{secondi:.3f}", "-i", clip.percorso_video, "-t", "1.6",
               "-vf", f"fps=25,scale={LARGHEZZA}:{ALTEZZA}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    grezzi = np.frombuffer(subprocess.run(comando, capture_output=True).stdout, np.uint8)
    numero = grezzi.size // (ALTEZZA * LARGHEZZA * 3)
    return grezzi[: numero * ALTEZZA * LARGHEZZA * 3].reshape(numero, ALTEZZA, LARGHEZZA, 3) if numero >= 16 else None


@torch.no_grad()
def vettore(indice: int) -> np.ndarray | None:
    frame = frame_finestra(indice)
    if frame is None:
        return None
    scelti = frame[np.linspace(0, len(frame) - 1, 16).round().astype(int), 8:120, 29:141]
    clip = (torch.from_numpy(scelti.copy()).permute(3, 0, 1, 2).float() / 255 - MEDIA) / DEVIAZIONE
    return modello(clip[None])[0].numpy()


avvio = time.time()
risultati = {}
for nome, indici in (("test", scelti_test), ("train", scelti_train)):
    vettori, tenuti = [], []
    for numero, indice in enumerate(indici):
        if time.time() - avvio > SECONDI_MASSIMI:
            print("tempo finito", flush=True)
            break
        valori = vettore(int(indice))
        if valori is not None:
            vettori.append(valori)
            tenuti.append(int(indice))
        if numero % 250 == 0:
            print(f"{nome} {numero}/{len(indici)} {time.time() - avvio:.0f} s", flush=True)
    tenuti = np.array(tenuti, np.int64)
    risultati[nome] = (tenuti, np.array(vettori, np.float32))
    np.savez(uscita / f"rgb_{nome}.npz", indici=tenuti, vettori=risultati[nome][1], id_clip=id_clip[tenuti],
             inizio_finestra=inizi[tenuti], etichetta=etichette[tenuti], sorgente=sorgenti[tenuti])

(indici_train, x_train), (indici_test, x_test) = risultati["train"], risultati["test"]
classificatore = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(x_train, violenta[indici_train])
probabilita = classificatore.predict_proba(x_test)[:, 1]
np.save(uscita / "rgb_probabilita_test.npy", probabilita)
vere, predette = violenta[indici_test], probabilita > 0.5
riassunto = {"finestre_train": len(indici_train), "finestre_test": len(indici_test),
             "accuratezza": float((vere == predette).mean()), "recall": float(predette[vere].mean()),
             "falsi_allarmi": float(predette[~vere].mean()), "per_sorgente": {}}
for sorgente in SORGENTI:
    maschera = sorgenti[indici_test] == sorgente
    if maschera.any():
        riassunto["per_sorgente"][sorgente] = {"finestre": int(maschera.sum()),
                                               "accuratezza": float((vere[maschera] == predette[maschera]).mean())}
(uscita / "prova_rgb.json").write_text(json.dumps(riassunto, indent=1))
print(json.dumps(riassunto, indent=1), flush=True)
