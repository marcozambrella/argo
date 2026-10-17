"""Addestramento di un classificatore di finestre (Graph-Transformer o ST-GCN) con split per video.

Uso:
    python -m sicurezza.addestramento --finestre finestre.npz --modello graph_transformer --uscita runs/gt

Tutti gli iperparametri predefiniti sono PROVVISORI (DA DECIDERE con Marco dopo i primi esperimenti).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .aumenti import aumenta_lotto
from .config import CLASSI
from .modelli import conta_parametri, crea_modello
from .suddivisione import NOMI_SPLIT, dividi_per_gruppi
from .valutazione import latenza_cpu_millisecondi, metriche


def carica_finestre(percorso: str | Path) -> dict[str, np.ndarray]:
    with np.load(percorso, allow_pickle=False) as dati:
        return {chiave: dati[chiave] for chiave in dati.files}


def lotti(indici: np.ndarray, dimensione_lotto: int, mescola: bool, generatore: np.random.Generator):
    ordine = generatore.permutation(indici) if mescola else indici
    for partenza in range(0, len(ordine), dimensione_lotto):
        yield ordine[partenza:partenza + dimensione_lotto]


def tensori_lotto(finestre, indici, dispositivo):
    return (torch.from_numpy(finestre["nodi"][indici].astype(np.float32)).to(dispositivo),
            torch.from_numpy(finestre["maschera"][indici]).to(dispositivo),
            torch.from_numpy(finestre["classe_arma"][indici].astype(np.int64)).to(dispositivo),
            torch.from_numpy(finestre["etichetta"][indici]).to(dispositivo))


@torch.no_grad()
def predici(modello, finestre, indici, dispositivo, dimensione_lotto=256) -> tuple[np.ndarray, np.ndarray]:
    modello.eval()
    probabilita = []
    for indici_lotto in lotti(indici, dimensione_lotto, False, None):
        nodi, maschera, classe_arma, _ = tensori_lotto(finestre, indici_lotto, dispositivo)
        with torch.autocast(dispositivo.type, enabled=dispositivo.type == "cuda"):
            probabilita.append(torch.softmax(modello(nodi, maschera, classe_arma).float(), -1).cpu().numpy())
    probabilita = np.concatenate(probabilita) if probabilita else np.empty((0, len(CLASSI)))
    return probabilita, finestre["etichetta"][indici]


def addestra(finestre: dict, nome_modello: str, cartella_uscita: Path, epoche: int = 50, dimensione_lotto: int = 64,
             tasso_apprendimento: float = 5e-4, decadimento_pesi: float = 0.05, pazienza: int = 10,
             bilanciamento: str = "pesi", seme: int = 0, iperparametri_modello: dict | None = None) -> dict:
    torch.manual_seed(seme)
    generatore = np.random.default_rng(seme)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cartella_uscita.mkdir(parents=True, exist_ok=True)

    split = dividi_per_gruppi(finestre["gruppo"], finestre["etichetta"], finestre["sorgente"], seme=seme)
    indici_split = {nome: np.flatnonzero(split == numero) for numero, nome in enumerate(NOMI_SPLIT)}
    np.savez(cartella_uscita / "split.npz", **indici_split)

    iperparametri_modello = iperparametri_modello or {}
    modello = crea_modello(nome_modello, **iperparametri_modello).to(dispositivo)
    conteggio_train = np.bincount(finestre["etichetta"][indici_split["train"]], minlength=len(CLASSI))
    if bilanciamento == "pesi":  # pesi inversamente proporzionali alla frequenza (classi assenti: peso 0)
        pesi = np.where(conteggio_train > 0, conteggio_train.sum() / np.maximum(conteggio_train, 1) / len(CLASSI), 0)
        funzione_perdita = nn.CrossEntropyLoss(weight=torch.tensor(pesi, dtype=torch.float32, device=dispositivo))
    else:
        funzione_perdita = nn.CrossEntropyLoss()
    ottimizzatore = torch.optim.AdamW(modello.parameters(), lr=tasso_apprendimento, weight_decay=decadimento_pesi)
    passi_totali = epoche * math.ceil(len(indici_split["train"]) / dimensione_lotto)
    passi_riscaldamento = max(1, passi_totali // 20)
    pianificatore = torch.optim.lr_scheduler.LambdaLR(ottimizzatore, lambda passo: min(
        (passo + 1) / passi_riscaldamento, 0.5 * (1 + math.cos(math.pi * min(passo / passi_totali, 1.0)))))
    scalatore = torch.amp.GradScaler(enabled=dispositivo.type == "cuda")

    print(f"{nome_modello}: {conta_parametri(modello):,} parametri, dispositivo {dispositivo}, finestre "
          + ", ".join(f"{nome}={len(indici)}" for nome, indici in indici_split.items()), flush=True)
    storia, miglior_f1, epoche_senza_miglioramento = [], -1.0, 0
    for epoca in range(1, epoche + 1):
        modello.train()
        inizio, perdita_totale = time.perf_counter(), 0.0
        indici_train = indici_split["train"]
        if bilanciamento == "campionamento":  # ogni classe estratta con la stessa probabilità
            peso_finestra = 1.0 / conteggio_train[finestre["etichetta"][indici_train]]
            indici_train = generatore.choice(indici_train, len(indici_train), p=peso_finestra / peso_finestra.sum())
        for indici_lotto in lotti(indici_train, dimensione_lotto, True, generatore):
            nodi, maschera, classe_arma, etichetta = tensori_lotto(finestre, indici_lotto, dispositivo)
            nodi, maschera, classe_arma = aumenta_lotto(nodi, maschera, classe_arma)
            with torch.autocast(dispositivo.type, enabled=dispositivo.type == "cuda"):
                perdita = funzione_perdita(modello(nodi, maschera, classe_arma).float(), etichetta)
            ottimizzatore.zero_grad(set_to_none=True)
            scalatore.scale(perdita).backward()
            scalatore.unscale_(ottimizzatore)
            nn.utils.clip_grad_norm_(modello.parameters(), 1.0)
            scalatore.step(ottimizzatore)
            scalatore.update()
            pianificatore.step()
            perdita_totale += perdita.item() * len(indici_lotto)

        probabilita_val, vere_val = predici(modello, finestre, indici_split["val"], dispositivo)
        metriche_val = metriche(vere_val, probabilita_val.argmax(1))
        storia.append({"epoca": epoca, "perdita_train": perdita_totale / max(len(indici_train), 1),
                       "macro_f1_val": metriche_val["macro_f1"], "secondi": time.perf_counter() - inizio})
        print(f"epoca {epoca}: perdita {storia[-1]['perdita_train']:.4f}, macro-F1 val {metriche_val['macro_f1']:.4f}"
              f" ({storia[-1]['secondi']:.0f} s)", flush=True)
        if metriche_val["macro_f1"] > miglior_f1:
            miglior_f1, epoche_senza_miglioramento = metriche_val["macro_f1"], 0
            torch.save({"nome_modello": nome_modello, "iperparametri_modello": iperparametri_modello,
                        "classi": CLASSI, "stato": modello.state_dict(), "epoca": epoca}, cartella_uscita / "migliore.pt")
        else:
            epoche_senza_miglioramento += 1
            if epoche_senza_miglioramento >= pazienza:
                print(f"stop anticipato: {pazienza} epoche senza miglioramento", flush=True)
                break

    salvato = torch.load(cartella_uscita / "migliore.pt", map_location=dispositivo)
    modello.load_state_dict(salvato["stato"])
    risultati = {"modello": nome_modello, "parametri": conta_parametri(modello), "epoca_migliore": salvato["epoca"],
                 "storia": storia,
                 "impostazioni": {"epoche": epoche, "dimensione_lotto": dimensione_lotto,
                                  "tasso_apprendimento": tasso_apprendimento, "decadimento_pesi": decadimento_pesi,
                                  "bilanciamento": bilanciamento, "seme": seme, **iperparametri_modello}}
    for nome in ("val", "test"):
        probabilita, vere = predici(modello, finestre, indici_split[nome], dispositivo)
        risultati[nome] = metriche(vere, probabilita.argmax(1)) if len(vere) else None
        np.save(cartella_uscita / f"probabilita_{nome}.npy", probabilita)
    risultati["latenza_cpu_ms_per_finestra"] = latenza_cpu_millisecondi(modello)
    (cartella_uscita / "risultati.json").write_text(json.dumps(risultati, indent=2, ensure_ascii=False))
    return risultati


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--finestre", required=True)
    parser.add_argument("--modello", choices=["graph_transformer", "stgcn"], required=True)
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--epoche", type=int, default=50)
    parser.add_argument("--lotto", type=int, default=64)
    parser.add_argument("--tasso-apprendimento", type=float, default=5e-4)
    parser.add_argument("--bilanciamento", choices=["pesi", "campionamento", "nessuno"], default="pesi")
    parser.add_argument("--seme", type=int, default=0)
    argomenti = parser.parse_args()
    risultati = addestra(carica_finestre(argomenti.finestre), argomenti.modello, Path(argomenti.uscita),
                         argomenti.epoche, argomenti.lotto, argomenti.tasso_apprendimento,
                         bilanciamento=argomenti.bilanciamento, seme=argomenti.seme)
    print(json.dumps({chiave: risultati[chiave] for chiave in ("val", "test", "latenza_cpu_ms_per_finestra")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
