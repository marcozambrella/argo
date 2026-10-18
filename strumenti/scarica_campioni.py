"""Scarica pochi video veri dai dataset Kaggle per provare la catena in locale (senza login: API pubblica).

Kaggle a volte consegna i file grandi dentro uno zip con lo stesso nome: qui vengono estratti.
Uso:  uv run python strumenti/scarica_campioni.py [--cartella data/campioni]
      uv run python strumenti/scarica_campioni.py --dataset-intero aniketsahu09/chain-snatching-cctv-dataset
"""

import argparse
import io
import json
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

API = "https://www.kaggle.com/api/v1/datasets"
CAMPIONI = {  # dataset → file da scaricare (percorsi interni al dataset)
    "vulamnguyen/rwf2000": ["RWF-2000/val/Fight/0Ow4cotKOuw_0.avi", "RWF-2000/val/Fight/0Ow4cotKOuw_1.avi",
                            "RWF-2000/val/NonFight/10uSOcwS_0.avi", "RWF-2000/val/NonFight/1AURh0Wj_0.avi"],
    "mohamedtariki/vsd-cctv-fights": ["groundtruth.json", "original_format-001-200/fight_0001.webm",
                                      "original_format-001-200/fight_0002.mp4"],
    "toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd": [
        "SCVD/SCVD_converted/Test/Normal/t_n001_converted.avi", "SCVD/SCVD_converted/Test/Violence/t_v001_converted.avi",
        "SCVD/SCVD_converted/Test/Weaponized/t_w001_converted.avi"],
}


def scarica_file(riferimento_dataset: str, percorso_interno: str, destinazione: Path) -> None:
    url = f"{API}/download/{riferimento_dataset}/{urllib.parse.quote(percorso_interno, safe='')}"
    contenuto = urllib.request.urlopen(url, timeout=300).read()
    if contenuto[:4] == b"PK\x03\x04" and not percorso_interno.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(contenuto)) as archivio:
            contenuto = archivio.read(archivio.namelist()[0])
    destinazione.parent.mkdir(parents=True, exist_ok=True)
    destinazione.write_bytes(contenuto)


def elenca_file_dataset(riferimento_dataset: str) -> list[str]:
    nomi, segnalibro = [], None
    while True:
        url = f"{API}/list/{riferimento_dataset}?pageSize=200"
        if segnalibro:
            url += f"&pageToken={urllib.parse.quote(segnalibro)}"
        risposta = json.load(urllib.request.urlopen(url, timeout=60))
        nomi += [file["name"] for file in risposta.get("datasetFiles", [])]
        segnalibro = risposta.get("nextPageTokenNullable")
        if not segnalibro:
            return nomi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cartella", default="data/campioni")
    parser.add_argument("--dataset-intero", default=None, help="scarica tutti i file di questo dataset Kaggle")
    argomenti = parser.parse_args()
    cartella = Path(argomenti.cartella)
    campioni = CAMPIONI
    if argomenti.dataset_intero:
        campioni = {argomenti.dataset_intero: elenca_file_dataset(argomenti.dataset_intero)}
    for riferimento_dataset, percorsi in campioni.items():
        for percorso_interno in percorsi:
            destinazione = cartella / riferimento_dataset.split("/")[1] / percorso_interno
            if destinazione.exists():
                continue
            scarica_file(riferimento_dataset, percorso_interno, destinazione)
            print(f"{destinazione} ({destinazione.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps({"cartella": str(cartella)}))


if __name__ == "__main__":
    main()
