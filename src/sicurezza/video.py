"""Lettura video a fps costante tramite ffmpeg (conversione al volo, nessun file riscritto)."""

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .config import FPS_ANALISI


@dataclass(frozen=True)
class InfoVideo:
    larghezza: int
    altezza: int
    durata_secondi: float


def leggi_info_video(percorso_video: str) -> InfoVideo:
    uscita_ffprobe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration:stream_side_data=rotation:format=duration",
         "-of", "json", str(percorso_video)],
        capture_output=True, text=True, check=True,
    ).stdout
    dati = json.loads(uscita_ffprobe)
    flusso_video = dati["streams"][0]
    larghezza, altezza = int(flusso_video["width"]), int(flusso_video["height"])
    # ffmpeg ruota automaticamente i video con metadato di rotazione: le dimensioni in uscita vanno scambiate
    rotazione = next((int(dato["rotation"]) for dato in flusso_video.get("side_data_list", []) if "rotation" in dato), 0)
    if rotazione % 180:
        larghezza, altezza = altezza, larghezza
    durata = float(flusso_video.get("duration") or dati.get("format", {}).get("duration") or 0.0)
    return InfoVideo(larghezza, altezza, durata)


def leggi_frame(percorso_video: str, fps: int = FPS_ANALISI, inizio_secondi: float | None = None,
                fine_secondi: float | None = None) -> Iterator[np.ndarray]:
    """Frame BGR (altezza, larghezza, 3) uint8 a `fps` costanti, opzionalmente solo tra inizio e fine (secondi)."""
    info = leggi_info_video(percorso_video)
    comando = ["ffmpeg", "-v", "error", "-nostdin"]
    if inizio_secondi:
        comando += ["-ss", f"{inizio_secondi:.3f}"]
    comando += ["-i", str(percorso_video)]
    if fine_secondi is not None:
        comando += ["-t", f"{fine_secondi - (inizio_secondi or 0):.3f}"]
    comando += ["-vf", f"fps={fps}", "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    byte_per_frame = info.larghezza * info.altezza * 3
    processo_ffmpeg = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       bufsize=byte_per_frame)
    try:
        while True:
            dati_frame = processo_ffmpeg.stdout.read(byte_per_frame)
            if len(dati_frame) < byte_per_frame:
                break
            yield np.frombuffer(dati_frame, np.uint8).reshape(info.altezza, info.larghezza, 3)
    finally:
        processo_ffmpeg.stdout.close()
        processo_ffmpeg.kill()
        processo_ffmpeg.wait()
