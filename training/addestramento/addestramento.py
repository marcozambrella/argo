# ===== avvio comune: uguale in tutti i kernel di training/ =====
# Kaggle carica un solo file per kernel, quindi questo blocco è copiato in testa a ognuno.
import os
import shutil
import subprocess
import sys
URL_REPO = "https://github.com/marcozambrella/argo"
subprocess.run(["git", "clone", "--depth", "1", URL_REPO, "/tmp/repo"], check=True)
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

"""Passi 5-6: finestre etichettate dalle estrazioni, poi Graph-Transformer (GPU 0) e ST-GCN (GPU 1) in parallelo,
poi taratura del WARNING ARMA (D35) e analisi per sorgente. Le estrazioni devono venire da UNA sola sorgente
(aggiungi-armi se il detector è stato riaddestrato, altrimenti estrazione): due cartelle con gli stessi nomi di file
si sovrascriverebbero.
Addestramento 2 (19/10): dopo l'addestramento 1 il Graph-Transformer andava in overfitting dall'epoca 4 → più
regolarizzazione; pazienza 6 (le epoche costano 10-20 minuti); tetto ORE_MASSIME per stare nella quota GPU."""
import json
from pathlib import Path

# per ciascun modello (in parallelo). 20/10 05:40: quota 3,55 h − ~1,2 h tra finestre, un'epoca oltre il tetto,
# taratura e margine
ORE_MASSIME = 2.2
OPZIONI = {"graph_transformer": ["--epoche", "25", "--pazienza", "6", "--dropout", "0.3", "--decadimento-pesi", "0.1"],
           "stgcn": ["--epoche", "30", "--pazienza", "6"]}

# più sorgenti (es. aggiungi-armi + estrazione-normali, D46): si uniscono le annotazioni, un id_clip una volta sola
righe_annotazioni = {}
for file_annotazioni in sorted(Path("/kaggle/input").glob("**/annotazioni.jsonl")):
    for riga in file_annotazioni.read_text().splitlines():
        id_clip = json.loads(riga)["id_clip"]
        if id_clip not in righe_annotazioni:
            righe_annotazioni[id_clip] = riga
ingresso = Path("/tmp/annotazioni.jsonl")
ingresso.write_text("\n".join(righe_annotazioni.values()) + "\n")
cartelle_estratti = []
for cartella in Path("/kaggle/input").glob("**/estratti"):
    if cartella.is_dir():
        cartelle_estratti.append(str(cartella))
print("clip annotate:", len(righe_annotazioni), "| estratti:", cartelle_estratti, flush=True)
# finestre già costruite dal kernel solo CPU `roma-safe-cctv-finestre` (se è tra le sorgenti): niente GPU sprecata
finestre_pronte = sorted(Path("/kaggle/input").glob("**/finestre.npz"))
if finestre_pronte:
    percorso_finestre = str(finestre_pronte[0])
else:
    percorso_finestre = "/tmp/finestre.npz"
    subprocess.run([sys.executable, "-m", "sicurezza.modelli", "dataset-finestre", "--annotazioni", str(ingresso),
                    "--estratti"] + cartelle_estratti + ["--uscita", percorso_finestre], check=True, env=ambiente)
print("finestre:", percorso_finestre, flush=True)

# i due modelli in parallelo, uno per GPU
processi = []
for gpu, modello in enumerate(["graph_transformer", "stgcn"]):
    ambiente_gpu = dict(ambiente)
    ambiente_gpu["CUDA_VISIBLE_DEVICES"] = str(gpu)
    comando = [sys.executable, "-m", "sicurezza.modelli", "addestra", "--finestre", percorso_finestre,
               "--modello", modello, "--uscita", f"/kaggle/working/{modello}", "--ore-massime", str(ORE_MASSIME)]
    processi.append(subprocess.Popen(comando + OPZIONI[modello], env=ambiente_gpu))
for processo in processi:
    processo.wait()

# D35: taratura del WARNING ARMA sul validation set (stesso split del Graph-Transformer)
subprocess.run([sys.executable, "-m", "sicurezza.demo", "tara-warning", "--finestre", percorso_finestre,
                "--split", "/kaggle/working/graph_transformer/split.npz", "--estratti"] + cartelle_estratti
               + ["--uscita", "/kaggle/working/taratura_warning.json"], check=True, env=ambiente)
# il sistema intero sul test: modello + WARNING ARMA tarato + "rissa + warning = rissa con arma" (D6)
for modello in OPZIONI:
    subprocess.run([sys.executable, "-m", "sicurezza.demo", "valuta-sistema", "--finestre", percorso_finestre,
                    "--split", f"/kaggle/working/{modello}/split.npz",
                    "--probabilita", f"/kaggle/working/{modello}/probabilita_test.npy",
                    "--taratura", "/kaggle/working/taratura_warning.json", "--estratti"] + cartelle_estratti
                   + ["--uscita", f"/kaggle/working/{modello}/sistema_test.json"], check=True, env=ambiente)
# dove sbagliano: metriche per sorgente e per video del test
probabilita_dei_modelli = []
for modello in OPZIONI:
    probabilita_dei_modelli.append(f"{modello}=/kaggle/working/{modello}/probabilita_test.npy")
subprocess.run([sys.executable, "-m", "sicurezza.modelli", "analizza-predizioni", "--finestre", percorso_finestre,
                "--probabilita"] + probabilita_dei_modelli + ["--uscita", "/kaggle/working/analisi_test.json"],
               check=True, env=ambiente)
