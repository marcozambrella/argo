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

"""D68, solo CPU: finestre di tutte le sorgenti (estrazione principale con armi v3 + mani, normali di folla, dati
nuovi), senza le clip SCVD dove Marco non capiva l'arma, poi un solo split per tutti i modelli del confronto e la
taratura del WARNING ARMA sul validation. Come training/finestre, con in più i dati nuovi e il filtro."""
import json
from pathlib import Path
from sicurezza.dati import FILE_TIPO_ARMA_SCVD, tipi_arma_scvd

esclusi = set()
for id_clip, tipi in tipi_arma_scvd(FILE_TIPO_ARMA_SCVD).items():
    if len(tipi) == 0:
        esclusi.add(id_clip)
righe_annotazioni = {}   # più sorgenti: un id_clip una volta sola
tolte = 0
for file_annotazioni in sorted(Path("/kaggle/input").glob("**/annotazioni.jsonl")):
    for riga in file_annotazioni.read_text().splitlines():
        id_clip = json.loads(riga)["id_clip"]
        if id_clip in esclusi:
            tolte += 1
            continue
        if id_clip not in righe_annotazioni:
            righe_annotazioni[id_clip] = riga
ingresso = Path("/tmp/annotazioni.jsonl")
ingresso.write_text("\n".join(righe_annotazioni.values()) + "\n")
cartelle_estratti = []
for cartella in Path("/kaggle/input").glob("**/estratti"):
    if cartella.is_dir():
        cartelle_estratti.append(str(cartella))
print("clip annotate:", len(righe_annotazioni), "| tolte (arma non chiara):", tolte, "| estratti:", cartelle_estratti,
      flush=True)
uscita = Path("/kaggle/working")
finestre = str(uscita / "finestre.npz")
split = str(uscita / "split.npz")
subprocess.run([sys.executable, "-m", "sicurezza.modelli", "dataset-finestre", "--annotazioni", str(ingresso),
                "--estratti"] + cartelle_estratti + ["--uscita", finestre], check=True, env=ambiente)
subprocess.run([sys.executable, "-m", "sicurezza.modelli", "salva-split", "--finestre", finestre, "--uscita", split],
               check=True, env=ambiente)
subprocess.run([sys.executable, "-m", "sicurezza.demo", "tara-warning", "--finestre", finestre, "--split", split,
                "--estratti"] + cartelle_estratti + ["--uscita", str(uscita / "taratura_warning.json")],
               check=True, env=ambiente)
print(json.loads((uscita / "taratura_warning.json").read_text())["migliore"], flush=True)
