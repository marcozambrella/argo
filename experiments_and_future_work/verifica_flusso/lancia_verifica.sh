#!/usr/bin/env bash
# D71: lancia una parte della verifica del flusso veloce come kernel CPU (4 parti: 0 e 1 sull'account principale,
# 2 e 3 su gigibosi con "secondo").
# Uso: experiments_and_future_work/verifica_flusso/lancia_verifica.sh 2 [secondo] [variante, es. s480-25]
set -euo pipefail
parte="$1"
cartella="$(cd "$(dirname "$0")" && pwd)"
copia="$cartella/../verifica_flusso_$parte"
rm -rf "$copia"
mkdir "$copia"
cp "$cartella/verifica_flusso.py" "$cartella/kernel-metadata.json" "$copia"/
sed -i "s/^PARTE = 0$/PARTE = $parte/" "$copia/verifica_flusso.py"
variante="${3:-d71}"
sed -i "s/^VARIANTE = \"d71\"$/VARIANTE = \"$variante\"/" "$copia/verifica_flusso.py"
prefisso=""
if [ "$variante" != "d71" ]; then prefisso="$variante-"; fi   # i kernel della prima verifica restano com'erano
sed -i "s/VARIANTEPARTE/$prefisso$parte/g" "$copia/kernel-metadata.json"
if [ "${2:-}" = "secondo" ]; then
    "$cartella/../lancia_su_secondo.sh" "verifica_flusso_$parte"
else
    "$cartella/../../training/lancia.sh" "../experiments_and_future_work/verifica_flusso_$parte"
fi
rm -rf "$copia"
