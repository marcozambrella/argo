"""Costruisce annotazioni.jsonl (ClipAnnotata) ed elenco.jsonl (VideoDaEstrarre) da tutte le sorgenti trovate.

Su Kaggle i dataset sono montati sotto /kaggle/input: ogni modulo di etichette cerca i propri file da lì.
Si estrae un SOVRAINSIEME (es. CCTV-Fights con anche i video da cellulare): così le scelte su cosa usare per
l'addestramento restano aperte e non serve rifare l'estrazione.

Uso:
    python -m sicurezza.prepara_elenchi --input /kaggle/input --uscita /kaggle/working
"""

import argparse
from collections import Counter
from pathlib import Path

from .estrazione import scrivi_elenco
from .etichette import scrivi_annotazioni
from .etichette import cctv_fights, chain_snatching, rwf2000, scvd, ucf_uca

SORGENTI = {"rwf2000": rwf2000.elenca_clip, "cctv_fights": cctv_fights.elenca_clip, "scvd": scvd.elenca_clip,
            "ucf_uca": ucf_uca.elenca_clip, "chain_snatching": chain_snatching.elenca_clip}


def prepara(cartella_input: Path, cartella_uscita: Path, sorgenti=tuple(SORGENTI)):
    clip = []
    for nome in sorgenti:
        try:
            clip_sorgente = SORGENTI[nome](cartella_input)
        except StopIteration:  # es. groundtruth.json non trovato perché il dataset non è montato
            clip_sorgente = []
        print(f"{nome:16s} {len(clip_sorgente)} clip")
        clip += clip_sorgente
    cartella_uscita.mkdir(parents=True, exist_ok=True)
    scrivi_annotazioni(clip, cartella_uscita / "annotazioni.jsonl")
    # ordine alternato per sorgente: se la sessione si ferma a metà, ogni sorgente ha già una parte estratta
    per_sorgente = {}
    for singola_clip in clip:
        per_sorgente.setdefault(singola_clip.sorgente, []).append(singola_clip)
    alternate = [gruppo[indice] for indice in range(max(map(len, per_sorgente.values()), default=0))
                 for gruppo in per_sorgente.values() if indice < len(gruppo)]
    scrivi_elenco([singola_clip.da_estrarre() for singola_clip in alternate], cartella_uscita / "elenco.jsonl")
    secondi = Counter()
    for singola_clip in clip:
        if singola_clip.fine_secondi is not None:
            secondi[singola_clip.sorgente] += singola_clip.fine_secondi - (singola_clip.inizio_secondi or 0)
    for sorgente, totale in sorted(secondi.items()):
        print(f"  {sorgente}: {totale / 3600:.2f} ore note in anticipo (segmenti)")
    return clip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--sorgenti", nargs="*", default=list(SORGENTI))
    argomenti = parser.parse_args()
    prepara(Path(argomenti.input), Path(argomenti.uscita), argomenti.sorgenti)


if __name__ == "__main__":
    main()
