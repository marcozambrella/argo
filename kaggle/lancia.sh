#!/usr/bin/env bash
# Lancia un kernel Kaggle di questa cartella: sostituisce KAGGLE_UTENTE con il tuo nome utente e fa il push.
# Uso:  kaggle/lancia.sh benchmark        (poi: kaggle kernels status <utente>/roma-safe-cctv-benchmark)
set -euo pipefail
cartella_kernel="$(cd "$(dirname "$0")" && pwd)/$1"
utente="${KAGGLE_USERNAME:-$(kaggle config view 2>/dev/null | sed -n 's/^- username: //p')}"
if [ -z "$utente" ]; then echo "nome utente Kaggle sconosciuto: esporta KAGGLE_USERNAME" >&2; exit 1; fi
copia="$(mktemp -d)"
cp "$cartella_kernel"/* "$copia"/
sed -i "s/KAGGLE_UTENTE/$utente/g" "$copia"/kernel-metadata.json
kaggle kernels push -p "$copia"
echo "stato: kaggle kernels status $utente/$(python3 -c "import json;print(json.load(open('$copia/kernel-metadata.json'))['id'].split('/')[1])")"
