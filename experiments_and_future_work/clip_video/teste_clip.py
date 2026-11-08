"""Goal 80% G3: teste piccole sulle feature CLIP congelate del kernel clip_video.
Per ogni finestra 12 token (4 frame × 3 viste). Testa: lo stesso MLP su ogni token (D → 256), poi massimo e media sui
token, poi 6 uscite sì/no (violenza, arma, fuoco, coltello, scippo, anomalia) con bersagli e maschere come b1a.
Configurazione fissa (niente ricerca di iperparametri sul val: con poche clip armate il val va in sovradattamento,
G1); sul val si sceglie solo l'epoca. Misura: la stessa di `demo valuta-armi-scippi` (pezzi da 10 s).
Scrive i punteggi nel formato delle reti video, così si possono anche mettere insieme a b2/b4.

Le teste di v1.3 (pesi/clip_teste.pt, via impacchetta_teste_clip.py): MODELLI = ("ViT-B-16",) e FRAME = (0, 2);
armi con DIFFERENZE = False (G8), scippo con DIFFERENZE = True (G9).

Uso (checkout su esperimenti-armi-scippi, feature del kernel clip_video in runs/esperimenti_armi_scippi/clip_video):
    PYTHONPATH=src .venv/bin/python experiments_and_future_work/clip_video/teste_clip.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from sicurezza.dati import INDICE_CLASSE, categoria_ucf, tipi_arma_scvd, tipi_arma_ucf
from sicurezza.demo import (TIPI_MISURATI, misura_incrociata, misura_un_tipo, pezzi_armi_scippi, salva_punteggi,
                            tipi_della_clip)

CARTELLA = Path("runs/esperimenti_armi_scippi/clip_video")
MODELLI = ("ViT-L-14-336", "ViT-B-16")
USCITE = ("violenza", "arma", "fuoco", "coltello", "scippo", "anomalia")
EPOCHE = 15
LOTTO = 256
NASCOSTI = 256
SEMI = (0, 1, 2)
FRAME = (0, 1, 2, 3)   # quali dei 4 frame (4, 14, 24, 34); (0, 2) = un frame ogni 0,8 s, metà del costo
DIFFERENZE = False   # 08/11: differenze fra frame consecutivi (movimento) come token in più
VISTE = (0, 1, 2)   # 0 = intero, 1 = quadrato a sinistra, 2 = a destra; con (0,) costa un terzo sulla CPU


def bersagli_e_maschere(finestre: dict, tipi_ucf: dict, tipi_scvd: dict) -> tuple:
    etichette = finestre["etichetta"]
    id_clip = finestre["id_clip"]
    sorgenti = finestre["sorgente"]
    tipo_nuovo = finestre["tipo_nuovo"]
    nuova = tipo_nuovo != ""
    armata = etichette == INDICE_CLASSE["violenza_con_arma"]
    bersagli = np.zeros((len(etichette), len(USCITE)), np.float32)
    maschere = np.ones((len(etichette), len(USCITE)), np.float32)
    bersagli[:, 0] = np.isin(etichette, [INDICE_CLASSE["violenza"], INDICE_CLASSE["violenza_con_arma"]])
    bersagli[:, 1] = armata
    bersagli[:, 4] = etichette == INDICE_CLASSE["scippo"]
    bersagli[:, 5] = (etichette != INDICE_CLASSE["normale"]) | (nuova & (tipo_nuovo != "normale"))
    for riga in np.where((sorgenti == "ucf_uca") & (etichette == INDICE_CLASSE["normale"]))[0]:
        id_video = str(id_clip[riga])[len("ucf/"):].split("@")[0]
        if categoria_ucf(id_video) != "Normal":
            maschere[riga, 5] = 0
    for riga in np.where(armata)[0]:
        tipi = tipi_della_clip(str(id_clip[riga]), tipi_ucf, tipi_scvd)
        if len(tipi) == 0:
            maschere[riga, 2] = 0
            maschere[riga, 3] = 0
        if "fuoco" in tipi:
            bersagli[riga, 2] = 1
        if "coltello" in tipi:
            bersagli[riga, 3] = 1
    # furto e altra di argo-anomalie: solo anomalia (le altre uscite restano 0, come in b1a)
    return bersagli, maschere


class Testa(torch.nn.Module):
    """Lo stesso MLP su ogni token, poi massimo e media. Con DIFFERENZE anche un MLP sulle differenze fra frame
    consecutivi della stessa vista (il movimento: uno scippo è uno strappo rapido), anch'esse con massimo e media."""
    def __init__(self, dimensione: int):
        super().__init__()
        self.token = torch.nn.Sequential(torch.nn.LayerNorm(dimensione), torch.nn.Linear(dimensione, NASCOSTI),
                                         torch.nn.GELU(), torch.nn.Dropout(0.2))
        ingresso_uscita = 2 * NASCOSTI
        if DIFFERENZE:
            self.differenze = torch.nn.Sequential(torch.nn.LayerNorm(dimensione), torch.nn.Linear(dimensione, NASCOSTI),
                                                  torch.nn.GELU(), torch.nn.Dropout(0.2))
            ingresso_uscita = 4 * NASCOSTI
        self.uscita = torch.nn.Linear(ingresso_uscita, len(USCITE))

    def forward(self, token):   # [B, frame × viste, D], prima tutte le viste del primo frame, poi del secondo...
        nascosti = self.token(token)
        parti = [nascosti.max(dim=1).values, nascosti.mean(dim=1)]
        if DIFFERENZE:
            per_frame = token.view(token.shape[0], len(FRAME), len(VISTE), token.shape[-1])
            salti = (per_frame[:, 1:] - per_frame[:, :-1]).reshape(token.shape[0], -1, token.shape[-1])
            nascosti_salti = self.differenze(salti)
            parti.append(nascosti_salti.max(dim=1).values)
            parti.append(nascosti_salti.mean(dim=1))
        return self.uscita(torch.cat(parti, dim=1))


def probabilita(testa, feature: np.ndarray) -> np.ndarray:
    testa.eval()
    uscite = []
    with torch.no_grad():
        for primo in range(0, len(feature), 2048):
            lotto = torch.from_numpy(feature[primo: primo + 2048].astype(np.float32))
            uscite.append(torch.sigmoid(testa(lotto)).numpy())
    return np.concatenate(uscite)


def punteggi_finestre(finestre: dict, p: np.ndarray) -> dict:
    risultato = {"id_clip": finestre["id_clip"], "inizio_finestra": finestre["inizio_finestra"],
                 "etichetta": finestre["etichetta"], "sorgente": finestre["sorgente"]}
    for numero, nome in enumerate(USCITE):
        risultato[nome] = p[:, numero]
    return risultato


def misura(val: dict, test: dict, tipi_ucf, tipi_scvd) -> dict:
    pezzi_val = pezzi_armi_scippi(val, 10.0, tipi_ucf, tipi_scvd)
    pezzi_test = pezzi_armi_scippi(test, 10.0, tipi_ucf, tipi_scvd)
    risultato = {}
    for tipo in TIPI_MISURATI:
        m = misura_un_tipo(pezzi_val, pezzi_test, tipo)
        incrociata = misura_incrociata(pezzi_val, pezzi_test, tipo)
        risultato[tipo] = {"val": m["val"]["bilanciata"], "test": m["bilanciata"], "intervallo": m["intervallo_95"],
                           "incrociata": incrociata["bilanciata"], "clip": incrociata["clip_positive"],
                           "falsi_folle": m.get("falsi_allarmi_folle")}
    return risultato


def addestra_testa(nome: str, seme: int, feature: dict, bersagli, maschere, peso_positivi, finestre_val: dict,
                   tipi_ucf, tipi_scvd) -> Testa:
    """Una testa addestrata sul train; l'epoca si sceglie sul val (media dei 4 tipi a pezzi, soglia del val)."""
    torch.manual_seed(seme)
    testa = Testa(feature["train"].shape[-1])
    ottimizzatore = torch.optim.AdamW(testa.parameters(), lr=1e-3, weight_decay=1e-2)
    pianificatore = torch.optim.lr_scheduler.CosineAnnealingLR(ottimizzatore, EPOCHE)
    generatore = np.random.default_rng(seme)
    migliore = -1.0
    stato_migliore = None
    for epoca in range(1, EPOCHE + 1):
        testa.train()
        ordine = generatore.permutation(len(feature["train"]))
        for primo in range(0, len(ordine), LOTTO):
            scelte = ordine[primo: primo + LOTTO]
            ingresso = torch.from_numpy(feature["train"][scelte].astype(np.float32))
            uscita = testa(ingresso)
            perdita = torch.nn.functional.binary_cross_entropy_with_logits(
                uscita, torch.from_numpy(bersagli[scelte]), pos_weight=peso_positivi, reduction="none")
            maschera = torch.from_numpy(maschere[scelte])
            perdita = (perdita * maschera).sum() / maschera.sum().clamp(min=1)
            ottimizzatore.zero_grad()
            perdita.backward()
            ottimizzatore.step()
        pianificatore.step()
        # l'epoca si sceglie sul val: media dell'accuratezza bilanciata dei pezzi sui 4 tipi (soglia del val)
        val = punteggi_finestre(finestre_val, probabilita(testa, feature["val"]))
        pezzi_val = pezzi_armi_scippi(val, 10.0, tipi_ucf, tipi_scvd)
        valori_val = []
        for tipo in TIPI_MISURATI:
            valori_val.append(misura_un_tipo(pezzi_val, pezzi_val, tipo)["bilanciata"])
        valore = float(np.mean(valori_val))
        print(f"{nome} seme {seme} epoca {epoca}: perdita {perdita.item():.4f}, val {valore:.4f} "
              f"({' '.join(f'{v:.3f}' for v in valori_val)})", flush=True)
        if valore > migliore:
            migliore = valore
            stato_migliore = {}
            for chiave, tensore in testa.state_dict().items():
                stato_migliore[chiave] = tensore.clone()
    testa.load_state_dict(stato_migliore)
    return testa


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(4)
    tipi_ucf = tipi_arma_ucf()
    tipi_scvd = tipi_arma_scvd("annotations/armi/tipo_arma_scvd.json")
    finestre = {}
    for split in ("train", "val", "test"):
        with np.load(CARTELLA / f"finestre_{split}.npz") as file:
            finestre[split] = {}
            for chiave in file.files:
                finestre[split][chiave] = file[chiave]
    bersagli, maschere = bersagli_e_maschere(finestre["train"], tipi_ucf, tipi_scvd)
    print("finestre:", {split: len(finestre[split]["riga"]) for split in finestre})
    positivi = (bersagli * maschere).sum(0)
    negativi = ((1 - bersagli) * maschere).sum(0)
    peso_positivi = torch.tensor(negativi / np.maximum(positivi, 1), dtype=torch.float32).clamp(max=30)
    print("positivi per uscita:", dict(zip(USCITE, positivi.astype(int).tolist())))

    risultati = {}
    for nome in MODELLI:
        feature = {}
        for split in ("train", "val", "test"):
            valori = np.load(CARTELLA / f"feature_{nome}_{split}.npy")
            valori = valori[:, list(FRAME)][:, :, list(VISTE)]                  # [N, frame, viste, D]
            feature[split] = valori.reshape(len(valori), -1, valori.shape[-1])   # [N, 4 × viste, D]
        # 07/11: 3 semi e media delle probabilità (con un seme solo i numeri ballano di 1-2 punti)
        probabilita_split = {"val": np.zeros((len(feature["val"]), len(USCITE))),
                             "test": np.zeros((len(feature["test"]), len(USCITE)))}
        for seme in SEMI:
            testa = addestra_testa(nome, seme, feature, bersagli, maschere, peso_positivi, finestre["val"],
                                   tipi_ucf, tipi_scvd)
            nome_file = f"testa_{nome}_{seme}_viste{len(VISTE)}_frame{len(FRAME)}_diff{int(DIFFERENZE)}.pt"
            torch.save(testa.state_dict(), CARTELLA / nome_file)
            for split in ("val", "test"):
                probabilita_split[split] += probabilita(testa, feature[split]) / len(SEMI)
        suffisso = ""
        if VISTE != (0, 1, 2):
            suffisso = "_viste" + "".join(str(v) for v in VISTE)
        if DIFFERENZE:
            suffisso = suffisso + "_differenze"
        if FRAME != (0, 1, 2, 3):
            suffisso = suffisso + "_frame" + "".join(str(f) for f in FRAME)
        cartella_punteggi = CARTELLA / f"punteggi_{nome}{suffisso}"
        cartella_punteggi.mkdir(exist_ok=True)
        uscite_split = {}
        for split in ("val", "test"):
            p = probabilita_split[split]
            uscite_split[split] = punteggi_finestre(finestre[split], p)
            punteggi = {}
            for numero, nome_uscita in enumerate(USCITE):
                if nome_uscita in ("arma", "fuoco", "coltello", "scippo", "anomalia"):
                    punteggi[nome_uscita] = p[:, numero]
            salva_punteggi(cartella_punteggi / f"punteggi_{split}.npz",
                           {"id_clip": finestre[split]["id_clip"],
                            "inizio_finestra": finestre[split]["inizio_finestra"],
                            "etichetta": finestre[split]["etichetta"], "sorgente": finestre[split]["sorgente"]},
                           punteggi)
            np.save(cartella_punteggi / f"violenza_{split}.npy", p[:, 0])
        risultati[nome] = misura(uscite_split["val"], uscite_split["test"], tipi_ucf, tipi_scvd)
        for tipo, r in risultati[nome].items():
            print(f"{nome} {tipo:9s} test {r['test']:.3f} {r['intervallo']} | incrociata {r['incrociata']:.3f} "
                  f"({r['clip']} clip) | folle {r['falsi_folle']}", flush=True)
    nome_json = f"teste_clip_viste{len(VISTE)}_frame{len(FRAME)}_diff{int(DIFFERENZE)}.json"
    (CARTELLA / nome_json).write_text(json.dumps(risultati, indent=1))


main()
