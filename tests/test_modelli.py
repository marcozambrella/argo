import pytest
import torch

from sicurezza.caratteristiche import NODI_PERSONE, NODI_PER_FRAME
from sicurezza.modelli import MODELLI, conta_parametri, crea_modello
from sicurezza.modelli.graph_transformer import (ARCO_ARMA_POLSO, ARCO_OSSO, ARCO_PERSONE_DIVERSE,
                                                 matrice_tipi_arco)


def lotto_finto(lotto=3, frame=40):
    nodi = torch.randn(lotto, frame, NODI_PER_FRAME, 5)
    maschera = torch.zeros(lotto, frame, NODI_PER_FRAME, dtype=torch.bool)
    maschera[0, :, :2 * 17] = True                  # 2 persone
    maschera[1, :, :17] = True                       # 1 persona + arma
    maschera[1, ::2, NODI_PERSONE] = True
    # il terzo esempio è una scena vuota
    nodi = nodi * maschera[..., None]
    classe_arma = torch.zeros(lotto, frame, 2, dtype=torch.long)
    classe_arma[1, ::2, 0] = 2
    return nodi, maschera, classe_arma


@pytest.mark.parametrize("nome", list(MODELLI))
def test_uscita_e_gradienti(nome):
    modello = crea_modello(nome)
    logit = modello(*lotto_finto())
    assert logit.shape == (3, 4) and torch.isfinite(logit).all()
    logit.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in modello.parameters() if p.grad is not None)


def test_nodi_mascherati_non_contano_nel_graph_transformer():
    """Cambiare i valori dei nodi assenti non deve cambiare l'uscita.
    (L'ST-GCN, come l'originale, non usa la maschera dentro le convoluzioni: escluso.)"""
    modello = crea_modello("graph_transformer").eval()
    nodi, maschera, classe_arma = lotto_finto()
    rumore = torch.randn_like(nodi) * (~maschera[..., None])
    with torch.no_grad():
        prima = modello(nodi, maschera, classe_arma)
        dopo = modello(nodi + rumore, maschera, classe_arma)
    torch.testing.assert_close(prima, dopo, atol=1e-4, rtol=1e-4)


def test_tipi_arco():
    tipi = matrice_tipi_arco()
    assert tipi[5, 7] == ARCO_OSSO                     # spalla_sx – gomito_sx
    assert tipi[0, 17] == ARCO_PERSONE_DIVERSE         # naso persona 0 – naso persona 1
    assert tipi[NODI_PERSONE, 9] == ARCO_ARMA_POLSO    # arma – polso_sx persona 0
    assert (tipi == tipi.T).all()


def test_dimensioni_ragionevoli():
    assert 0.5e6 < conta_parametri(crea_modello("graph_transformer")) < 2e6
    assert 2e6 < conta_parametri(crea_modello("stgcn")) < 4e6
