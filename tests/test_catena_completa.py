"""Catena completa su dati finti: estrazioni → annotazioni → finestre → addestramento dei due modelli."""

import json

import numpy as np
import pytest

from aiuti_sintetici import estrazione_finta_con_agitazione
from sicurezza.addestramento import addestra
from sicurezza.dataset_finestre import costruisci_dataset, riepilogo
from sicurezza.estrazione import nome_file_estratto
from sicurezza.etichette import ClipAnnotata


@pytest.fixture(scope="module")
def dataset_finto(tmp_path_factory):
    cartella = tmp_path_factory.mktemp("estratti")
    clip = []
    for numero in range(40):
        violenta = numero % 2 == 0
        id_clip = f"finto/clip{numero}"
        np.savez(cartella / nome_file_estratto(id_clip),
                 **estrazione_finta_con_agitazione(agitazione=40.0 if violenta else 0.0, seme=numero))
        clip.append(ClipAnnotata(id_clip=id_clip, sorgente="finto", gruppo=f"finto/{numero}", percorso_video="-",
                                 classe_intera="violenza" if violenta else "normale"))
    return costruisci_dataset(clip, [cartella])


def test_dataset(dataset_finto):
    assert len(dataset_finto["etichetta"]) == 40 * 3   # 60 frame → finestre a 0, 10, 20
    assert dataset_finto["nodi"].dtype == np.float16
    assert "violenza" in riepilogo(dataset_finto)


# il Graph-Transformer piccolo è veloce ma impara più lentamente (~150 passi); l'ST-GCN è lento su CPU ma rapido
CONFIGURAZIONI = {"graph_transformer": dict(epoche=20, dimensione_lotto=8, iperparametri_modello={"dimensione": 32,
                                                                                                   "blocchi": 1}),
                  "stgcn": dict(epoche=12, dimensione_lotto=16)}


@pytest.mark.lento
@pytest.mark.parametrize("nome_modello", list(CONFIGURAZIONI))
def test_addestramento_impara(dataset_finto, nome_modello, tmp_path):
    risultati = addestra(dataset_finto, nome_modello, tmp_path, tasso_apprendimento=2e-3, pazienza=20,
                         **CONFIGURAZIONI[nome_modello])
    assert (tmp_path / "migliore.pt").exists() and (tmp_path / "split.npz").exists()
    assert json.loads((tmp_path / "risultati.json").read_text())["modello"] == nome_modello
    # polsi agitati vs fermi: il compito è facilissimo, un modello che funziona lo impara
    assert risultati["val"]["macro_f1"] >= 0.8, risultati["storia"]
