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

"""D68, solo CPU: i punteggi del sistema di oggi per le armi, sulle stesse finestre di val e test degli altri
modelli (una ogni 4: inizio multiplo di 40 frame). detector_v3 = WARNING ARMA tarato (vicinanza e frame minimi
della taratura sulle finestre v3), mani_d44 = l'ingrandimento sulle mani addestrato sulle foto (D44)."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from sicurezza.dati import nome_file_estratto
from sicurezza.demo import punteggi_detector_finestra, salva_punteggi
from sicurezza.modelli import carica_timeline

ingresso = Path("/kaggle/input")
uscita = Path("/kaggle/working")
percorso_finestre = next(ingresso.glob("**/finestre.npz"))
regola = json.loads((percorso_finestre.parent / "taratura_warning.json").read_text())["migliore"]
print("taratura:", regola, flush=True)
percorso_estratto = {}
for cartella in ingresso.glob("**/estratti"):
    for file in cartella.glob("*.npz"):
        percorso_estratto[file.name] = str(file)
with np.load(percorso_finestre) as finestre:
    campi = {"id_clip": finestre["id_clip"], "inizio_finestra": finestre["inizio_finestra"],
             "etichetta": finestre["etichetta"], "sorgente": finestre["sorgente"]}
split = np.load(percorso_finestre.parent / "split.npz")
CRITERI = {"detector_v3": regola["vicinanza"], "mani_d44": "mani"}


def punteggi_di_una_clip(percorso: str, inizi: list) -> list:
    timeline = carica_timeline(percorso)
    risultati = []
    for inizio in inizi:
        per_criterio = {}
        for nome, vicinanza in CRITERI.items():
            per_criterio[nome] = punteggi_detector_finestra(timeline, inizio, vicinanza, regola["frame_minimi"])
        risultati.append(per_criterio)
    return risultati


for nome_split in ("val", "test"):
    indici = split[nome_split]
    indici = indici[campi["inizio_finestra"][indici] % 40 == 0]
    tenute = []
    for riga in indici:
        if nome_file_estratto(str(campi["id_clip"][riga])) in percorso_estratto:
            tenute.append(riga)
    indici = np.array(tenute)
    valori = {}
    for nome in CRITERI:
        valori[nome] = {"arma": np.zeros(len(indici)), "fuoco": np.zeros(len(indici)), "coltello": np.zeros(len(indici))}
    clip_della_split = np.unique(campi["id_clip"][indici])
    with ProcessPoolExecutor(os.cpu_count()) as esecutore:
        lavori = []
        posizioni = []
        for clip in clip_della_split:
            dove = np.where(campi["id_clip"][indici] == clip)[0]
            inizi = campi["inizio_finestra"][indici[dove]].tolist()
            lavori.append(esecutore.submit(punteggi_di_una_clip, percorso_estratto[nome_file_estratto(str(clip))],
                                           inizi))
            posizioni.append(dove)
        for numero, lavoro in enumerate(lavori):
            for posizione, per_criterio in zip(posizioni[numero], lavoro.result()):
                for nome, punteggi in per_criterio.items():
                    for tipo, valore in punteggi.items():
                        valori[nome][tipo][posizione] = valore
            if numero % 200 == 0:
                print(f"{nome_split}: clip {numero}/{len(clip_della_split)}", flush=True)
    finestre_scelte = {}
    for chiave, colonna in campi.items():
        finestre_scelte[chiave] = colonna[indici]
    for nome in CRITERI:
        (uscita / nome).mkdir(exist_ok=True)
        salva_punteggi(uscita / nome / f"punteggi_{nome_split}.npz", finestre_scelte, valori[nome])
    print(f"{nome_split}: {len(indici)} finestre", flush=True)
print("fatto", flush=True)
