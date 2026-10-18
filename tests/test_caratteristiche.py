import numpy as np

from aiuti_sintetici import estrazione_finta
from sicurezza.caratteristiche import (CARATTERISTICHE_PER_NODO, NODI_PERSONE, NODI_PER_FRAME, carica_timeline,
                                       costruisci_finestra)


def salva(tmp_path, **opzioni):
    percorso = tmp_path / "finta.npz"
    np.savez_compressed(percorso, **estrazione_finta(**opzioni))
    return percorso


def test_forme_e_maschera(tmp_path):
    timeline = carica_timeline(salva(tmp_path, numero_persone=3))
    nodi, maschera, classe_arma = costruisci_finestra(timeline, 10)
    assert nodi.shape == (40, NODI_PER_FRAME, CARATTERISTICHE_PER_NODO)
    assert maschera[:, :3 * 17].all() and not maschera[:, 3 * 17:NODI_PERSONE].any()  # 4ª persona assente
    assert np.isfinite(nodi).all()


def test_normalizzazione_invariante_a_zoom(tmp_path):
    """Stessa scena ripresa a risoluzione doppia → stesse caratteristiche."""
    piccola = estrazione_finta(con_arma=False)
    grande = {chiave: valore for chiave, valore in piccola.items()}
    grande["persone_keypoint"] = piccola["persone_keypoint"] * np.array([2, 2, 1], np.float32)
    grande["persone_box"] = piccola["persone_box"] * 2
    np.savez(tmp_path / "p.npz", **piccola)
    np.savez(tmp_path / "g.npz", **grande)
    nodi_piccola, _, _ = costruisci_finestra(carica_timeline(tmp_path / "p.npz"), 0)
    nodi_grande, _, _ = costruisci_finestra(carica_timeline(tmp_path / "g.npz"), 0)
    np.testing.assert_allclose(nodi_piccola, nodi_grande, atol=1e-4)


def test_arma_nei_frame_pari(tmp_path):
    timeline = carica_timeline(salva(tmp_path, con_arma=True))
    nodi, maschera, classe_arma = costruisci_finestra(timeline, 0)
    assert maschera[0::2, NODI_PERSONE].all() and not maschera[1::2, NODI_PERSONE].any()
    assert (classe_arma[0::2, 0] == 1).all()  # classe 0 del detector → 1 (0 = nessuna arma)
    assert np.allclose(nodi[0::2, NODI_PERSONE, 2], 0.8, atol=1e-3)


def test_finestra_senza_persone(tmp_path):
    timeline = carica_timeline(salva(tmp_path, numero_persone=0, con_arma=False))
    nodi, maschera, _ = costruisci_finestra(timeline, 0)
    assert not maschera.any() and not nodi.any()


def test_ricucitura_unisce_frammenti_ma_non_persone_diverse():
    from sicurezza.caratteristiche import ricuci_tracce
    frame = 60

    def traccia(dal, al, centro_x):
        box = np.full((frame, 4), np.nan, np.float32)
        box[dal:al] = [centro_x - 40, 100, centro_x + 40, 300]   # persona alta 200 px
        keypoint = np.full((frame, 17, 3), np.nan, np.float32)
        keypoint[dal:al] = 1.0
        return keypoint, box

    tracce = {1: traccia(0, 20, 100), 2: traccia(25, 60, 110),    # stessa persona: buco di 5 frame, 10 px più in là
              3: traccia(0, 60, 400),                              # un'altra persona, presente per tutto il tempo
              4: traccia(28, 60, 700)}                             # parte lontano: non è il seguito di nessuno
    keypoint, box = ricuci_tracce({k: v[0] for k, v in tracce.items()}, {k: v[1] for k, v in tracce.items()})
    assert sorted(box) == [1, 3, 4]
    assert (~np.isnan(box[1][:, 0])).sum() == 55 and not np.isnan(keypoint[1][59]).any()
