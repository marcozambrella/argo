"""Chain-snatching (Kaggle aniketsahu09/chain-snatching-cctv-dataset): 170 video di scippi di collane (P1).

Il dataset non ha i tempi dell'evento: li annota Marco a mano (D15) con strumenti/annota_scippi.py in
annotations/chain_snatching/tempi_scippo.json. Diventano finestre "scippo" solo i tratti annotati; il resto del
video si scarta (prima e dopo lo scippo la scena non è affidabilmente "normale"). Tutti i video restano comunque
nell'elenco di estrazione, così l'estrazione non deve aspettare la fine dell'annotazione.
"""

import json
from pathlib import Path

from . import ClipAnnotata

FILE_TEMPI_SCIPPO = Path(__file__).resolve().parents[3] / "annotations" / "chain_snatching" / "tempi_scippo.json"


def elenca_clip(radice_dataset: str | Path, file_tempi: Path = FILE_TEMPI_SCIPPO) -> list[ClipAnnotata]:
    tempi = json.loads(Path(file_tempi).read_text()) if Path(file_tempi).exists() else {}
    clip = []
    for percorso in sorted(Path(radice_dataset).glob("**/Chain_Snatching_Videos/*.mp4")):
        annotazione = tempi.get(percorso.name)
        if annotazione is None:
            eventi, nota = [], "non ancora annotato: nessuna finestra"
        elif annotazione["nessuno_scippo"]:
            eventi, nota = [], "nessuno scippo visibile: nessuna finestra"
        else:
            eventi, nota = [(float(inizio), float(fine), "scippo") for inizio, fine in annotazione["eventi"]], ""
        clip.append(ClipAnnotata(
            id_clip=f"chain_snatching/{percorso.stem}", sorgente="chain_snatching",
            gruppo=f"chain_snatching/{percorso.stem}", percorso_video=str(percorso), eventi_secondi=eventi,
            classe_sfondo=None, note=nota))
    return clip
