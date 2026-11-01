# ===== avvio comune: uguale in tutti i kernel di training/ =====
# Kaggle carica un solo file per kernel, quindi questo blocco è copiato in testa a ognuno.
import os
import shutil
import subprocess
import sys
URL_REPO = "https://github.com/marcozambrella/argo"
# D68: gli esperimenti usano il branch esperimenti-armi-scippi, non main
subprocess.run(["git", "clone", "--depth", "1", "--branch", "esperimenti-armi-scippi", URL_REPO, "/tmp/repo"], check=True)
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

"""D68 strada A, solo CPU: i ritagli delle mani presi DAI VIDEO (D44 li prendeva da foto di armi e sui video non
discriminava). Per ogni finestra scelta: le 4 persone che vede il classificatore, 6 frame su 40, 2 polsi →
[6, 8, 56, 56, 3] uint8 a colori (il bianco e nero si fa dopo, nel kernel mani_video).
Train: tutte le finestre armate del train + altrettante non armate. Val e test: una finestra ogni 4 (inizio
multiplo di 40 frame) di tutte le clip: su queste si misurano tutti i modelli del confronto.
Uscita: ritagli_<split>.npy, presenti_<split>.npy, righe_<split>.npy (indici in finestre.npz)."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from sicurezza.dati import INDICE_CLASSE, nome_file_estratto, prepara_video_esperimenti
from sicurezza.modelli import carica_timeline, tracce_scelte_della_finestra
from sicurezza.visione import ISTANTI_RITAGLI_FINESTRA, LATO_RITAGLI_FINESTRA, MANI_PER_FINESTRA
from sicurezza.visione import ritagli_della_finestra

ingresso = Path("/kaggle/input")
uscita = Path("/kaggle/working")
generatore = np.random.default_rng(0)
clip_annotate = prepara_video_esperimenti(ingresso, Path("/tmp/dati"))
percorso_estratto = {}
for cartella in ingresso.glob("**/estratti"):
    for file in cartella.glob("*.npz"):
        percorso_estratto[file.name] = file
percorso_finestre = next(ingresso.glob("**/finestre.npz"))
with np.load(percorso_finestre) as finestre:
    id_clip = finestre["id_clip"]
    inizi = finestre["inizio_finestra"]
    etichette = finestre["etichetta"]
split = np.load(percorso_finestre.parent / "split.npz")
utilizzabile = np.zeros(len(id_clip), bool)
for riga, clip in enumerate(id_clip.tolist()):
    utilizzabile[riga] = clip in clip_annotate and nome_file_estratto(clip) in percorso_estratto

armata = etichette == INDICE_CLASSE["violenza_con_arma"]
train = np.intersect1d(split["train"], np.where(utilizzabile)[0])
armate = train[armata[train]]
non_armate = generatore.choice(train[~armata[train]], min(len(armate), int((~armata[train]).sum())), replace=False)
scelte = {"train": np.sort(np.concatenate([armate, non_armate]))}
for nome in ("val", "test"):
    indici = np.intersect1d(split[nome], np.where(utilizzabile)[0])
    scelte[nome] = indici[inizi[indici] % 40 == 0]
for nome, indici in scelte.items():
    print(f"{nome}: {len(indici)} finestre, armate {int(armata[indici].sum())}", flush=True)


def ritagli_di_una_clip(clip: str, righe: np.ndarray):
    """Per le finestre di una clip: (righe, ritagli, presenti). Il video si decodifica alla risoluzione dei
    keypoint, una finestra alla volta."""
    timeline = carica_timeline(percorso_estratto[nome_file_estratto(clip)])
    annotata = clip_annotate[clip]
    byte_per_frame = timeline.larghezza * timeline.altezza * 3
    risultati = []
    for riga in righe:
        secondi = int(inizi[riga]) / 25
        if annotata.inizio_secondi is not None:
            secondi += annotata.inizio_secondi
        comando = ["ffmpeg", "-v", "error", "-ss", f"{secondi:.3f}", "-i", annotata.percorso_video, "-t", "1.6",
                   "-vf", f"fps=25,scale={timeline.larghezza}:{timeline.altezza}", "-f", "rawvideo",
                   "-pix_fmt", "rgb24", "-"]
        grezzi = np.frombuffer(subprocess.run(comando, capture_output=True).stdout, np.uint8)
        numero = grezzi.size // byte_per_frame
        frame = grezzi[: numero * byte_per_frame].reshape(numero, timeline.altezza, timeline.larghezza, 3)
        tracce = tracce_scelte_della_finestra(timeline, int(inizi[riga]))
        ritagli, presenti = ritagli_della_finestra(frame, tracce)
        risultati.append((riga, ritagli, presenti))
    return risultati


for nome, indici in scelte.items():
    forma = (len(indici), len(ISTANTI_RITAGLI_FINESTRA), MANI_PER_FINESTRA, LATO_RITAGLI_FINESTRA,
             LATO_RITAGLI_FINESTRA, 3)
    ritagli_split = np.lib.format.open_memmap(uscita / f"ritagli_{nome}.npy", mode="w+", dtype=np.uint8, shape=forma)
    presenti_split = np.zeros(forma[:3], bool)
    posizione_della_riga = {}
    for posizione, riga in enumerate(indici.tolist()):
        posizione_della_riga[riga] = posizione
    clip_della_split = np.unique(id_clip[indici])
    with ThreadPoolExecutor(os.cpu_count() * 2) as esecutore:
        lavori = []
        for clip in clip_della_split:
            lavori.append(esecutore.submit(ritagli_di_una_clip, clip, indici[id_clip[indici] == clip]))
        for numero, lavoro in enumerate(lavori):
            for riga, ritagli, presenti in lavoro.result():
                ritagli_split[posizione_della_riga[int(riga)]] = ritagli
                presenti_split[posizione_della_riga[int(riga)]] = presenti
            lavori[numero] = None
            if numero % 100 == 0:
                print(f"{nome}: clip {numero}/{len(clip_della_split)}", flush=True)
    ritagli_split.flush()
    del ritagli_split
    np.save(uscita / f"presenti_{nome}.npy", presenti_split)
    np.save(uscita / f"righe_{nome}.npy", indici)
    print(f"{nome}: fatto, mani presenti nel {presenti_split.mean():.0%} dei posti", flush=True)
