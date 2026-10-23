"""Analisi semplice di un brano per il montaggio (D62): BPM, battiti e il tratto da N secondi più energico che inizia
su un attacco forte. Solo ffmpeg + numpy.   Uso: uv run python media/presentation/analizza_musica.py brano.mp3 [secondi]"""
import json
import subprocess
import sys

import numpy as np

CAMPIONI = 22050
PASSO = 512   # ~23 ms


def analizza(percorso: str, secondi: float = 45.0) -> dict:
    grezzo = subprocess.run(["ffmpeg", "-v", "error", "-i", percorso, "-ac", "1", "-ar", str(CAMPIONI), "-f", "f32le", "-"],
                            capture_output=True, check=True).stdout
    audio = np.frombuffer(grezzo, np.float32)
    finestre = np.lib.stride_tricks.sliding_window_view(audio, 2048)[::PASSO] * np.hanning(2048)
    spettro = np.abs(np.fft.rfft(finestre, axis=1))
    flusso = np.maximum(np.diff(np.log1p(spettro), axis=0), 0).sum(1)          # attacchi
    flusso = (flusso - flusso.mean()) / (flusso.std() + 1e-9)
    energia = np.sqrt((finestre ** 2).mean(1))
    tempo = np.arange(len(flusso)) * PASSO / CAMPIONI
    # BPM dall'autocorrelazione degli attacchi, tra 70 e 180
    auto = np.correlate(flusso, flusso, "full")[len(flusso) - 1:]
    ritardi = np.arange(len(auto)) * PASSO / CAMPIONI
    validi = (ritardi > 60 / 180) & (ritardi < 60 / 70)
    periodo = ritardi[validi][np.argmax(auto[validi])]
    # battiti: picchi degli attacchi vicini alla griglia del periodo
    picchi = np.where((flusso[1:-1] > flusso[:-2]) & (flusso[1:-1] >= flusso[2:]) & (flusso[1:-1] > 1.0))[0] + 1
    lunghezza = int(secondi * CAMPIONI / PASSO)
    energia_cumulata = np.concatenate([[0], np.cumsum(energia)])
    candidati = [p for p in picchi if p + lunghezza < len(energia) and flusso[p] > 2.0]
    migliore = max(candidati, key=lambda p: energia_cumulata[p + lunghezza] - energia_cumulata[p]) if candidati else 0
    inizio = tempo[migliore]
    battiti = [round(float(tempo[p] - inizio), 2) for p in picchi if inizio <= tempo[p] < inizio + secondi and flusso[p] > 2.0]
    return {"brano": percorso, "bpm": round(60 / periodo, 1), "inizio_tratto_secondi": round(float(inizio), 2),
            "secondi": secondi, "attacchi_forti_nel_tratto": battiti}


if __name__ == "__main__":
    print(json.dumps(analizza(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 45.0), ensure_ascii=False))
