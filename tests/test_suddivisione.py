import numpy as np

from sicurezza.suddivisione import dividi_per_gruppi


def test_nessun_gruppo_in_due_split_e_quote_rispettate():
    generatore = np.random.default_rng(1)
    gruppi = np.repeat([f"v{i}" for i in range(200)], generatore.integers(5, 30, 200))
    etichette = np.array([hash(g) % 4 for g in gruppi])
    sorgenti = np.array(["a" if int(g[1:]) % 2 else "b" for g in gruppi])
    split = dividi_per_gruppi(gruppi, etichette, sorgenti)
    for gruppo in np.unique(gruppi):
        assert len(np.unique(split[gruppi == gruppo])) == 1
    quote = np.bincount(split, minlength=3) / len(split)
    np.testing.assert_allclose(quote, [0.70, 0.15, 0.15], atol=0.05)


def test_riproducibile():
    gruppi = np.repeat([f"v{i}" for i in range(50)], 10)
    etichette, sorgenti = np.zeros(500, int), np.array(["a"] * 500)
    assert (dividi_per_gruppi(gruppi, etichette, sorgenti) == dividi_per_gruppi(gruppi, etichette, sorgenti)).all()
