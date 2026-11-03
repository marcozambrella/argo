#!/usr/bin/env bash
# Notte 03→04/11, E3: lancia una variante della pose come kernel CPU a sé (rif, s640_12, s480, s320, n640, s480_12).
# Con "secondo" va sull'account gigibosi (lancia_su_secondo.sh mette la sua copia delle finestre v3).
# Uso: experiments_and_future_work/pose_ridotta/lancia_pose.sh s480 [secondo]
set -euo pipefail
variante="$1"
cartella="$(cd "$(dirname "$0")" && pwd)"
copia="$cartella/../pose_ridotta_$variante"
rm -rf "$copia"
mkdir "$copia"
cp "$cartella/pose_ridotta.py" "$cartella/kernel-metadata.json" "$copia"/
sed -i "s/^VARIANTE = \"rif\"/VARIANTE = \"$variante\"/" "$copia/pose_ridotta.py"
sed -i "s/VARIANTE/${variante//_/-}/g" "$copia/kernel-metadata.json"
if [ "${2:-}" = "secondo" ]; then
    "$cartella/../lancia_su_secondo.sh" "pose_ridotta_$variante"
else
    "$cartella/../../training/lancia.sh" "../experiments_and_future_work/pose_ridotta_$variante"
fi
rm -rf "$copia"
