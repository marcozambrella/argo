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

"""Solo CPU (DA DECIDERE 22): l'addestramento 1 (armi del detector v1, senza folle) e il 2 (v3 + mani + folle) sulle
STESSE finestre di test, quelle dell'addestramento 2 (armi del v3, come nella demo). Due insiemi: tutto il test 2, e
solo le clip che il modello 1 non ha mai visto in addestramento (con le clip nuove lo split per video può cambiare)."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from sicurezza.modelli import analizza_predizioni, carica_finestre, metriche

ambiente = {**os.environ, "PYTHONPATH": "/tmp/repo/src"}
ingresso, uscita = Path("/kaggle/input"), Path("/kaggle/working")
percorso_finestre = next(ingresso.glob("**/finestre.npz"))
percorso_split = percorso_finestre.parent / "split.npz"            # lo stesso split dell'addestramento 2
taratura = percorso_finestre.parent / "taratura_warning.json"
cartelle_estratti = [str(c) for c in ingresso.glob("**/estratti") if c.is_dir()]
dal_kernel_2 = lambda p: "roma-safe-cctv-addestramento" in p.parts   # non "-addestramento-1" (il dataset)
probabilita, visti_dal_modello_1 = {}, {}
for modello in ("stgcn", "graph_transformer"):
    cartella_1 = next(p.parent for p in ingresso.glob(f"**/{modello}/migliore.pt") if "addestramento-1" in str(p))
    subprocess.run([sys.executable, "-m", "sicurezza.modelli", "predici", "--finestre", str(percorso_finestre),
                    "--checkpoint", str(cartella_1 / "migliore.pt"), "--split", str(percorso_split),
                    "--uscita", str(uscita / f"add1_{modello}.npy")], check=True, env=ambiente)
    probabilita[f"add1_{modello}"] = np.load(uscita / f"add1_{modello}.npy")
    split_1 = np.load(cartella_1 / "split.npz", allow_pickle=True)
    visti_dal_modello_1[modello] = set(split_1["clip_train"]) | set(split_1["clip_val"])
    probabilita[f"add2_{modello}"] = np.load(next(p for p in ingresso.glob(f"**/{modello}/probabilita_test.npy")
                                                  if dal_kernel_2(p)))

dati = np.load(percorso_finestre, allow_pickle=True)
indici_test = np.load(percorso_split)["test"]
etichette, clip_test = dati["etichetta"][indici_test], dati["id_clip"][indici_test]
mai_visti = ~np.isin(clip_test, list(visti_dal_modello_1["stgcn"] | visti_dal_modello_1["graph_transformer"]))
confronto = {"finestre_test": int(len(indici_test)), "finestre_mai_viste_dal_modello_1": int(mai_visti.sum())}
for insieme, maschera in (("tutto_il_test_2", np.ones(len(indici_test), bool)), ("mai_viste_dal_modello_1", mai_visti)):
    confronto[insieme] = {nome: metriche(etichette[maschera], valori.argmax(1)[maschera])
                          for nome, valori in probabilita.items()}
in_json = lambda valore: valore.tolist() if hasattr(valore, "tolist") else float(valore)
(uscita / "confronto_addestramenti.json").write_text(json.dumps(confronto, indent=1, default=in_json))
analisi = analizza_predizioni(carica_finestre(percorso_finestre), probabilita)
(uscita / "analisi_per_sorgente.json").write_text(json.dumps(analisi, indent=1, default=in_json))
for nome in ("add1_stgcn", "add1_graph_transformer"):   # il sistema intero (+ WARNING ARMA tarato) col modello 1
    subprocess.run([sys.executable, "-m", "sicurezza.demo", "valuta-sistema", "--finestre", str(percorso_finestre),
                    "--split", str(percorso_split), "--probabilita", str(uscita / f"{nome}.npy"),
                    "--taratura", str(taratura), "--estratti", *cartelle_estratti,
                    "--uscita", str(uscita / f"sistema_{nome}.json")], check=True, env=ambiente)
for insieme in ("tutto_il_test_2", "mai_viste_dal_modello_1"):
    print(insieme, {nome: round(m["macro_f1"], 3) for nome, m in confronto[insieme].items()}, flush=True)
