"""Goal 80% (07/11, Marco: "per i test di accuracy integra anche i test sui nuovi dataset").
Per ogni modello: soglia scelta sul val unito (finestre v3 + argo-anomalie, come sarebbe nel sistema vero), poi
accuratezza bilanciata a pezzi da 10 s separata sul test vecchio (finestre v3) e sul test nuovo (argo-anomalie: NWPU,
CHAD, ATMA-V, telecamere mai viste); più la misura incrociata (val ↔ test) di ciascuno.
Tipi: arma, fuoco, coltello, scippo (come valuta-armi-scippi) e violenza, anomalia (dove il modello le dà).

Un modello è "nome=cartella[,cartella]": cartelle con punteggi_val.npz, punteggi_test.npz e violenza_*.npy; le finestre
di argo-anomalie si riconoscono dalla sorgente "anomalie_*", quindi b1a e clip_video stanno in una cartella sola.
Uso:
    PYTHONPATH=src .venv/bin/python experiments_and_future_work/clip_video/misura_vecchi_e_nuovi.py uscita.json \
        b2=runs/.../rete_video_b2/punteggi_migliore,runs/.../punteggi_nuovi/b2 \
        b1a=runs/.../rete_video_b1a/punteggi_migliore
"""
import json
import sys
from pathlib import Path

import numpy as np

from sicurezza.dati import INDICE_CLASSE, tipi_arma_scvd, tipi_arma_ucf
from sicurezza.demo import (accuratezza_bilanciata, carica_punteggi, intervallo_bootstrap, pezzi_armi_scippi,
                            pezzi_di_video, scegli_soglia)

TIPI = ("arma", "fuoco", "coltello", "scippo", "violenza", "anomalia")


def carica(cartelle: list, split: str) -> dict:
    """I punteggi di uno split, uniti da più cartelle (con la violenza dal suo .npy)."""
    parti = []
    for cartella in cartelle:
        punteggi = carica_punteggi(Path(cartella) / f"punteggi_{split}.npz")
        percorso_violenza = Path(cartella) / f"violenza_{split}.npy"
        if percorso_violenza.exists():
            punteggi["violenza"] = np.load(percorso_violenza)
        parti.append(punteggi)
    chiavi = set(parti[0].keys())
    for parte in parti[1:]:
        chiavi = chiavi & set(parte.keys())
    unito = {}
    for chiave in chiavi:
        valori = []
        for parte in parti:
            valori.append(parte[chiave])
        unito[chiave] = np.concatenate(valori)
    return unito


def pezzi(punteggi: dict, tipi_ucf: dict, tipi_scvd: dict) -> dict:
    """pezzi_armi_scippi più violenza e anomalia (positivo: almeno una finestra di quel tipo; negativo: nessuna)."""
    risultato = pezzi_armi_scippi(punteggi, 10.0, tipi_ucf, tipi_scvd)
    nomi = pezzi_di_video(punteggi["id_clip"], punteggi["inizio_finestra"], 10.0)
    nomi_pezzi, posizione = np.unique(nomi, return_inverse=True)
    violenta = np.isin(punteggi["etichetta"], [INDICE_CLASSE["violenza"], INDICE_CLASSE["violenza_con_arma"]])
    anomala = punteggi["etichetta"] != INDICE_CLASSE["normale"]
    for tipo, di_tipo in (("violenza", violenta), ("anomalia", anomala)):
        positivo = np.zeros(len(nomi_pezzi), bool)
        np.logical_or.at(positivo, posizione, di_tipo)
        risultato[f"positivo_{tipo}"] = positivo
        risultato[f"negativo_{tipo}"] = ~positivo
        if tipo in punteggi:
            massimo = np.full(len(nomi_pezzi), -np.inf)
            np.maximum.at(massimo, posizione, np.asarray(punteggi[tipo], np.float64))
            risultato[f"punteggio_{tipo}"] = massimo
    return risultato


def sottoinsieme(p: dict, nuovi: bool) -> dict:
    """I pezzi di argo-anomalie (nuovi=True) o gli altri."""
    tenere = np.char.startswith(p["sorgente"].astype(str), "anomalie_") == nuovi
    risultato = {}
    for chiave, valori in p.items():
        risultato[chiave] = valori[tenere]
    return risultato


def misura_tipo(val: dict, test: dict, tipo: str, soglia: float) -> dict:
    positivi = test[f"positivo_{tipo}"]
    negativi = test[f"negativo_{tipo}"]
    if positivi.sum() == 0 or negativi.sum() == 0:
        return None
    punteggi = test[f"punteggio_{tipo}"]
    risultato = accuratezza_bilanciata(punteggi, positivi, negativi, soglia)
    risultato["intervallo_95"] = intervallo_bootstrap(punteggi, positivi, negativi, test["clip"], soglia)
    risultato["clip_positive"] = len(set(test["clip"][positivi].tolist()))
    # incrociata dentro il sottoinsieme: soglia scelta su uno split e misurata sull'altro
    parti = []
    for scelta, misura in ((val, test), (test, val)):
        if scelta[f"positivo_{tipo}"].sum() == 0 or misura[f"positivo_{tipo}"].sum() == 0:
            continue
        s = scegli_soglia(scelta[f"punteggio_{tipo}"], scelta[f"positivo_{tipo}"], scelta[f"negativo_{tipo}"])
        parti.append((misura[f"punteggio_{tipo}"] - s, misura[f"positivo_{tipo}"], misura[f"negativo_{tipo}"]))
    if len(parti) == 2:
        p = np.concatenate([parti[0][0], parti[1][0]])
        pos = np.concatenate([parti[0][1], parti[1][1]])
        neg = np.concatenate([parti[0][2], parti[1][2]])
        risultato["incrociata"] = accuratezza_bilanciata(p, pos, neg, 0.0)["bilanciata"]
        risultato["clip_incrociata"] = int(pos.sum())
    return risultato


def main() -> None:
    uscita = Path(sys.argv[1])
    tipi_ucf = tipi_arma_ucf()
    tipi_scvd = tipi_arma_scvd("annotations/armi/tipo_arma_scvd.json")
    risultati = {}
    for argomento in sys.argv[2:]:
        nome, elenco = argomento.split("=")
        cartelle = elenco.split(",")
        val = pezzi(carica(cartelle, "val"), tipi_ucf, tipi_scvd)
        test = pezzi(carica(cartelle, "test"), tipi_ucf, tipi_scvd)
        risultati[nome] = {}
        for tipo in TIPI:
            if f"punteggio_{tipo}" not in val:
                continue
            soglia = scegli_soglia(val[f"punteggio_{tipo}"], val[f"positivo_{tipo}"], val[f"negativo_{tipo}"])
            risultati[nome][tipo] = {"soglia": round(soglia, 4)}
            for nome_parte, nuovi in (("vecchi", False), ("nuovi", True)):
                val_parte = sottoinsieme(val, nuovi)
                test_parte = sottoinsieme(test, nuovi)
                if len(test_parte["clip"]) == 0:
                    continue
                misura = misura_tipo(val_parte, test_parte, tipo, soglia)
                if misura is None:
                    continue
                risultati[nome][tipo][nome_parte] = misura
                print(f"{nome:10s} {tipo:9s} {nome_parte:6s} test {misura['bilanciata']:.3f} {misura['intervallo_95']} "
                      f"({misura['clip_positive']} clip) incrociata {misura.get('incrociata', float('nan')):.3f}",
                      flush=True)
    uscita.write_text(json.dumps(risultati, indent=1))


main()
