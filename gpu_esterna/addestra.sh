#!/usr/bin/env bash
# Addestramento 2 su una macchina con GPU qualsiasi (RunPod, Vast.ai, Lightning, un PC con NVIDIA, ...), quando la
# quota di Kaggle è finita. Prende finestre.npz + split + taratura dal kernel solo CPU `roma-safe-cctv-finestre`,
# addestra ST-GCN e Graph-Transformer (in parallelo con 2 GPU, altrimenti prima ST-GCN, il migliore) e carica i
# risultati nel dataset Kaggle privato `roma-safe-cctv-addestramento-2`; poi il kernel solo CPU
# `roma-safe-cctv-valutazione` fa valuta-sistema e l'analisi per sorgente (servono le timeline, che stanno su Kaggle).
#
# Uso:   export KAGGLE_API_TOKEN=...      # il token di Kaggle: solo nell'ambiente, MAI in un file del repo
#        ORE_MASSIME=3 bash addestra.sh   # tetto di tempo per modello; SOLO_STGCN=1 per saltare il Graph-Transformer
set -euo pipefail
ORE_MASSIME="${ORE_MASSIME:-3}"
CARTELLA="${CARTELLA:-$HOME/roma-safe-cctv-addestramento}"
mkdir -p "$CARTELLA" && cd "$CARTELLA"
[ -d repo ] || git clone -q --depth 1 https://github.com/marcozambrella/argo repo
git -C repo log -1 --format="commit %h"
pip install -q kaggle lap "ultralytics>=8.4"   # torch c'è già nelle immagini con GPU
export PYTHONPATH="$CARTELLA/repo/src"
[ -f dati/finestre.npz ] || kaggle kernels output marcozambrella/roma-safe-cctv-finestre -p dati \
    --file-pattern "(finestre|split)\.npz|taratura_warning\.json"
ls -la dati

addestra() {   # modello, gpu, opzioni...
    local modello=$1 gpu=$2
    shift 2
    CUDA_VISIBLE_DEVICES=$gpu python -m sicurezza.modelli addestra --finestre dati/finestre.npz --modello "$modello" \
        --uscita "risultati/$modello" --ore-massime "$ORE_MASSIME" "$@" 2>&1 | tee "log_$modello.txt"
}
# stessi iperparametri del kernel kaggle/addestramento (addestramento 2)
OPZIONI_STGCN=(--epoche 30 --pazienza 6)
OPZIONI_GT=(--epoche 25 --pazienza 6 --dropout 0.3 --decadimento-pesi 0.1)
if [ "$(nvidia-smi -L | wc -l)" -ge 2 ]; then
    addestra stgcn 0 "${OPZIONI_STGCN[@]}" > /dev/null &
    addestra graph_transformer 1 "${OPZIONI_GT[@]}" > /dev/null &
    wait
else
    addestra stgcn 0 "${OPZIONI_STGCN[@]}"
    [ "${SOLO_STGCN:-0}" = 1 ] || addestra graph_transformer 0 "${OPZIONI_GT[@]}"
fi

cp dati/taratura_warning.json log_*.txt risultati/
echo '{"title": "roma-safe-cctv addestramento 2", "id": "marcozambrella/roma-safe-cctv-addestramento-2", "licenses": [{"name": "other"}]}' \
    > risultati/dataset-metadata.json
kaggle datasets create -p risultati --dir-mode zip \
    || kaggle datasets version -p risultati -m "addestramento 2" --dir-mode zip
echo "fatto: ora lancia il kernel solo CPU roma-safe-cctv-valutazione (kaggle/lancia.sh valutazione)"
