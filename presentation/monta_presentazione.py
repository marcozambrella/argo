"""Monta la video presentation da presentation/scaletta.json con ffmpeg (D12). Riproducibile: quando il modello
migliora si rigenerano i clip annotati e si rilancia questo script.

Uso:  uv run python presentation/monta_presentazione.py [--scaletta presentation/scaletta.json] [--uscita runs/presentazione.mp4]

SEGNAPOSTO DESIGN: font, colori e transizioni sono provvisori (vedi STILE).
"""

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

STILE = {"sfondo": "0x111111", "colore_testo": "white", "dimensione_titolo": 64, "dimensione_sottotitolo": 30}


def ffmpeg(*argomenti: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *argomenti], check=True)


def testo_sicuro(testo: str) -> str:
    return testo.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")


def scena_titolo(scena, larghezza, altezza, fps, uscita: Path) -> None:
    filtro = (f"drawtext=text='{testo_sicuro(scena['testo'])}':fontcolor={STILE['colore_testo']}:"
              f"fontsize={STILE['dimensione_titolo']}:x=(w-text_w)/2:y=h/2-80")
    if scena.get("sottotitolo"):
        filtro += (f",drawtext=text='{testo_sicuro(scena['sottotitolo'])}':fontcolor={STILE['colore_testo']}:"
                   f"fontsize={STILE['dimensione_sottotitolo']}:x=(w-text_w)/2:y=h/2+20")
    ffmpeg("-f", "lavfi", "-i", f"color=c={STILE['sfondo']}:s={larghezza}x{altezza}:r={fps}:d={scena['secondi']}",
           "-vf", filtro, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(uscita))


def scena_clip(scena, larghezza, altezza, fps, uscita: Path) -> None:
    filtro = (f"scale={larghezza}:{altezza}:force_original_aspect_ratio=decrease,"
              f"pad={larghezza}:{altezza}:(ow-iw)/2:(oh-ih)/2:color={STILE['sfondo']},fps={fps}")
    if scena.get("didascalia"):
        filtro += (f",drawtext=text='{testo_sicuro(scena['didascalia'])}':fontcolor=white:fontsize=26:"
                   f"box=1:boxcolor=black@0.5:x=(w-text_w)/2:y=h-60")
    ffmpeg("-i", scena["percorso"], "-vf", filtro, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(uscita))


def scena_grafico_probabilita(scena, larghezza, altezza, fps, uscita: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(scena["percorso_csv"]) as file_csv:
        righe = list(csv.DictReader(file_csv))
    classi = [nome for nome in righe[0] if nome not in {"inizio_secondi", "fine_secondi", "classe", "warning_arma"}]
    tempi = [float(riga["fine_secondi"]) for riga in righe]
    figura, asse = plt.subplots(figsize=(larghezza / 100, altezza / 100), dpi=100)
    for classe in classi:  # SEGNAPOSTO DESIGN: colori di matplotlib
        asse.plot(tempi, [float(riga[classe]) for riga in righe], label=classe, linewidth=2)
    asse.set_xlabel("secondi")
    asse.set_ylabel("probabilità")
    asse.set_ylim(0, 1)
    asse.legend(loc="upper right")
    figura.tight_layout()
    immagine = uscita.with_suffix(".png")
    figura.savefig(immagine)
    plt.close(figura)
    ffmpeg("-loop", "1", "-i", str(immagine), "-t", str(scena["secondi"]), "-r", str(fps),
           "-vf", f"scale={larghezza}:{altezza}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(uscita))


SCENE = {"titolo": scena_titolo, "clip": scena_clip, "grafico_probabilita": scena_grafico_probabilita}


def monta(percorso_scaletta: Path, percorso_uscita: Path) -> None:
    scaletta = json.loads(percorso_scaletta.read_text())
    larghezza, altezza = scaletta["risoluzione"]
    with tempfile.TemporaryDirectory() as cartella_temporanea:
        pezzi = []
        for numero, scena in enumerate(scaletta["scene"]):
            pezzo = Path(cartella_temporanea) / f"{numero:03d}.mp4"
            percorso_necessario = scena.get("percorso") or scena.get("percorso_csv")
            if percorso_necessario and not Path(percorso_necessario).exists():
                print(f"salto la scena {numero} ({scena['tipo']}): manca {percorso_necessario}")
                continue
            SCENE[scena["tipo"]](scena, larghezza, altezza, scaletta["fps"], pezzo)
            pezzi.append(pezzo)
        elenco = Path(cartella_temporanea) / "elenco.txt"
        elenco.write_text("".join(f"file '{pezzo}'\n" for pezzo in pezzi))
        percorso_uscita.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg("-f", "concat", "-safe", "0", "-i", str(elenco), "-c", "copy", str(percorso_uscita))
    print(f"presentazione: {percorso_uscita} ({len(pezzi)} scene)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scaletta", default="presentation/scaletta.json")
    parser.add_argument("--uscita", default="runs/presentazione.mp4")
    argomenti = parser.parse_args()
    monta(Path(argomenti.scaletta), Path(argomenti.uscita))
