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

"""D68, GPU: estrazione dei dati nuovi scelti da Marco (Snatch 1.0 scippi + pezzi normali, scippi Kaggle annotati,
WVD sintetico), come l'estrazione normali: TrackTrack, detector v3 e mani, 2 processi per GPU. Snatch 1.0 non è
su Kaggle: si scarica da Zenodo (7 zip, 12,5 GB) e si scompatta uno zip alla volta in /tmp."""
from pathlib import Path
import zipfile
from sicurezza.strumenti import scarica_con_riprova

uscita = Path("/kaggle/working")
zenodo = Path("/tmp/dati/zenodo")
zenodo.mkdir(parents=True)
for numero in range(1, 8):
    nome = f"Snatch%201.0-20201231T071009Z-00{numero}.zip"
    archivio = Path("/tmp/snatch.zip")
    scarica_con_riprova(f"https://zenodo.org/records/10812645/files/{nome}?download=1", archivio)
    with zipfile.ZipFile(archivio) as dentro:
        dentro.extractall(zenodo)
    archivio.unlink()
    print("zip", numero, "scompattato", flush=True)
# prepara-elenchi cerca i dataset sotto una sola cartella: lì vanno anche quelli di Kaggle
for cartella in Path("/kaggle/input").iterdir():
    (Path("/tmp/dati") / cartella.name).symlink_to(cartella)
subprocess.run([sys.executable, "-m", "sicurezza.dati", "prepara-elenchi", "--input", "/tmp/dati", "--uscita",
                str(uscita), "--sorgenti", "snatch10", "snatch10_normali", "scippi_kaggle", "wvd"],
               check=True, env=ambiente)

pesi_armi = sorted(Path("/kaggle/input").glob("**/yolo26s_armi/weights/best.pt"))
pesi_mani = sorted(Path("/kaggle/input").glob("**/mani_best.pt"))
opzioni = ["--modello-armi", str(pesi_armi[0]), "--modello-mani", str(pesi_mani[0])]
opzioni += ["--tracker", "tracktrack.yaml"]   # lo stesso dei dati dell'addestramento (D47 mai fatta)
print("modelli:", opzioni, flush=True)
PROCESSI_PER_GPU = 2
NUMERO_GPU = 2
parti = PROCESSI_PER_GPU * NUMERO_GPU
processi = []
for parte in range(parti):
    comando = [sys.executable, "-m", "sicurezza.visione", "estrai", "--elenco", str(uscita / "elenco.jsonl"),
               "--uscita", str(uscita / "estratti"), "--parte", str(parte), "--parti", str(parti),
               "--dispositivo", str(parte % NUMERO_GPU), "--ore-massime", "8"]
    processi.append(subprocess.Popen(comando + opzioni, env=ambiente))
for processo in processi:
    processo.wait()
print("fatti:", len(list((uscita / "estratti").glob("*.npz"))), "file .npz", flush=True)
