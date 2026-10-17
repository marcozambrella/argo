import numpy as np

from aiuti_sintetici import estrazione_finta
from sicurezza.allarme_arma import arma_vicina_a_persona, classe_finale, frame_con_arma, warning_arma
from sicurezza.caratteristiche import carica_timeline


def test_arma_nella_mano_conta_arma_lontana_no():
    persona = np.array([[100, 100, 200, 400]], float)
    assert arma_vicina_a_persona(np.array([190, 250, 210, 270]), persona)
    assert not arma_vicina_a_persona(np.array([500, 250, 520, 270]), persona)
    assert not arma_vicina_a_persona(np.array([190, 250, 210, 270]), np.empty((0, 4)))


def test_warning_su_finestra(tmp_path):
    np.savez(tmp_path / "f.npz", **estrazione_finta(con_arma=True))   # arma nei frame pari, confidenza 0,8
    timeline = carica_timeline(tmp_path / "f.npz")
    assert frame_con_arma(timeline, 0) == 20
    assert warning_arma(timeline, 0)
    assert not warning_arma(timeline, 0, soglia_confidenza=0.9)


def test_regola_violenza_piu_arma():
    probabilita_violenza = np.array([0.1, 0.7, 0.1, 0.1])
    assert classe_finale(probabilita_violenza, arma_confermata=True) == "violenza_con_arma"
    assert classe_finale(probabilita_violenza, arma_confermata=False) == "violenza"
    assert classe_finale(np.array([0.7, 0.1, 0.1, 0.1]), arma_confermata=True) == "normale"  # solo warning
