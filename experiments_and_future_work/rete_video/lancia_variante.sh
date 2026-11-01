#!/usr/bin/env bash
# D68: lancia una variante della rete video (b1, b2, b3) come kernel a sé: copia la cartella, mette la variante nel
# codice e nell'id del kernel, poi usa training/lancia.sh.
# Uso: experiments_and_future_work/rete_video/lancia_variante.sh b2
set -euo pipefail
variante="$1"
cartella="$(cd "$(dirname "$0")" && pwd)"
copia="$cartella/../rete_video_$variante"
rm -rf "$copia"
mkdir "$copia"
cp "$cartella/rete_video.py" "$cartella/kernel-metadata.json" "$copia"/
sed -i "s/^VARIANTE = \"b1\"/VARIANTE = \"$variante\"/" "$copia/rete_video.py"
sed -i "s/VARIANTE/$variante/g" "$copia/kernel-metadata.json"
"$cartella/../../training/lancia.sh" "../experiments_and_future_work/rete_video_$variante"
rm -rf "$copia"
