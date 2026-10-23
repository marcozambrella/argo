#!/usr/bin/env bash
# Copia in public/argo/ le uscite vere di Argo (runs/argo_casi_uso/<nome>: video_annotato.mp4 + eventi.json) e i
# brani candidati, per la composizione ArgoApertura (D61). Da lanciare dalla cartella site/montaggio.
set -e
RADICE=../..
mkdir -p public/argo
cp $RADICE/presentation/musica/candidati/*.mp3 public/argo/
for cartella in $RADICE/runs/argo_casi_uso/*/; do
  nome=$(basename "$cartella")
  [ -f "$cartella/eventi.json" ] || continue
  cp "$cartella/video_annotato.mp4" "public/argo/$nome.mp4"
  cp "$cartella/eventi.json" "public/argo/$nome.json"
done
ls public/argo
