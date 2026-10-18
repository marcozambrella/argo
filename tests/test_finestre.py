from sicurezza.config import INDICE_CLASSE
from sicurezza.finestre import etichetta_finestra, inizi_finestre

NORMALE, VIOLENZA = INDICE_CLASSE["normale"], INDICE_CLASSE["violenza"]


def test_prime_finestre_0_39_e_10_49():
    inizi = inizi_finestre(100)
    assert [(inizio, inizio + 39) for inizio in inizi[:2]] == [(0, 39), (10, 49)]
    # l'ultima finestra deve stare tutta nel video
    assert inizi[-1] == 60 and inizi[-1] + 40 <= 100


def test_video_troppo_corto_nessuna_finestra():
    assert len(inizi_finestre(39)) == 0
    assert list(inizi_finestre(40)) == [0]


def test_etichetta_soglia_20_frame():
    eventi = [(30, 200, VIOLENZA)]
    assert etichetta_finestra(0, eventi, classe_sfondo=NORMALE) is None        # 10 frame di evento: ambigua
    assert etichetta_finestra(10, eventi, classe_sfondo=NORMALE) == VIOLENZA   # 20 frame: esattamente la soglia
    assert etichetta_finestra(200, eventi, classe_sfondo=NORMALE) == NORMALE   # nessun evento
    assert etichetta_finestra(200, eventi, classe_sfondo=None) is None         # sorgente senza sfondo affidabile


def test_eventi_sovrapposti_non_contano_doppio():
    # due eventi lunghi che nella finestra 0-39 coprono gli stessi 15 frame (25-39): 15, non 25 → ambigua
    eventi = [(25, 80, VIOLENZA), (30, 80, VIOLENZA)]
    assert etichetta_finestra(0, eventi, classe_sfondo=NORMALE) is None


def test_evento_breve_conta_se_contenuto_per_intero():
    """D22: uno scippo di 12 frame (0,5 s) prende la finestra che lo contiene tutto, non quelle che lo tagliano."""
    SCIPPO = INDICE_CLASSE["scippo"]
    eventi = [(45, 57, SCIPPO)]
    assert etichetta_finestra(20, eventi, classe_sfondo=None) == SCIPPO     # 20-59: contiene 45-56
    assert etichetta_finestra(40, eventi, classe_sfondo=None) == SCIPPO     # 40-79: contiene 45-56
    assert etichetta_finestra(50, eventi, classe_sfondo=None) is None       # 50-89: lo taglia → ambigua
    assert etichetta_finestra(0, eventi, classe_sfondo=None) is None        # 0-39: non lo tocca, sfondo None
