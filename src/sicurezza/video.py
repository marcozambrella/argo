"""Lettura video a fps costante tramite ffmpeg (conversione al volo, nessun file riscritto)."""

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .config import FPS


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration: float  # secondi


def probe(path: str) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration:stream_side_data=rotation:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    stream = info["streams"][0]
    w, h = int(stream["width"]), int(stream["height"])
    # ffmpeg ruota automaticamente i video con metadato di rotazione: le dimensioni in uscita vanno scambiate
    rotation = next((int(sd["rotation"]) for sd in stream.get("side_data_list", []) if "rotation" in sd), 0)
    if rotation % 180:
        w, h = h, w
    duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0.0)
    return VideoInfo(w, h, duration)


def iter_frames(path: str, fps: int = FPS, start: float | None = None,
                end: float | None = None) -> Iterator[np.ndarray]:
    """Frame BGR (h, w, 3) uint8 a `fps` costanti, opzionalmente solo tra `start` e `end` secondi."""
    info = probe(path)
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if end is not None:
        cmd += ["-t", f"{end - (start or 0):.3f}"]
    cmd += ["-vf", f"fps={fps}", "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    frame_bytes = info.width * info.height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
