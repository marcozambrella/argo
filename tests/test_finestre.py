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
    eventi = [(0, 15, VIOLENZA), (5, 15, VIOLENZA)]
    assert etichetta_finestra(0, eventi, classe_sfondo=NORMALE) is None  # 15 frame unici, non 25
