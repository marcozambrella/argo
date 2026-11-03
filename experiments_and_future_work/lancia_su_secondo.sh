#!/usr/bin/env bash
# D68: lancia un kernel di esperimento sull'account gigibosi senza condivisioni. Le uscite private di
# marcozambrella che servono (finestre v3, Snatch 1.0 ridotto) sono copiate come dataset privati di gigibosi:
# qui si sostituiscono nelle sorgenti. Con un secondo argomento si sceglie la variante della rete video.
# Uso: experiments_and_future_work/lancia_su_secondo.sh addestramento_v3_wvd
#      experiments_and_future_work/lancia_su_secondo.sh rete_video b1
set -euo pipefail
cartella="$(cd "$(dirname "$0")" && pwd)/$1"
copia="$(mktemp -d)/$(basename "$cartella")_secondo"
mkdir -p "$copia"
cp "$cartella"/*.py "$cartella"/kernel-metadata.json "$copia"/
python3 - "$copia/kernel-metadata.json" <<'PY'
import json, sys
percorso = sys.argv[1]
dati = json.load(open(percorso))
kernel = []
for sorgente in dati["kernel_sources"]:
    if sorgente.endswith("/argo-finestre-v3"):
        dati["dataset_sources"].append("gigibosi/argo-finestre-v3")
    else:
        kernel.append(sorgente)
dati["kernel_sources"] = kernel
dataset = []
for sorgente in dati["dataset_sources"]:
    dataset.append(sorgente.replace("marcozambrella/argo-snatch10-ridotto", "gigibosi/argo-snatch10-ridotto"))
dati["dataset_sources"] = dataset
open(percorso, "w").write(json.dumps(dati, indent=2) + "\n")
PY
if [ $# -ge 2 ]; then
    sed -i "s/^VARIANTE = \"b1\"/VARIANTE = \"$2\"/" "$copia"/*.py
    sed -i "s/VARIANTE/$2/g" "$copia/kernel-metadata.json"
fi
sed -i "s#KAGGLE_UTENTE#gigibosi#g" "$copia/kernel-metadata.json"
cat "$copia/kernel-metadata.json" | grep -A12 '"dataset_sources"' | head -20
KAGGLE_API_TOKEN=/home/Marco/.kaggle/token_secondo KAGGLE_USERNAME=gigibosi kaggle kernels push -p "$copia"
