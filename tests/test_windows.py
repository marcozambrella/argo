from sicurezza.config import CLASS_ID
from sicurezza.windows import label_window, window_starts

NORMALE, VIOLENZA = CLASS_ID["normale"], CLASS_ID["violenza"]


def test_prime_finestre_0_39_e_10_49():
    starts = window_starts(100)
    assert list(starts[:2]) == [0, 10]
    assert [(s, s + 39) for s in starts[:2]] == [(0, 39), (10, 49)]
    # l'ultima finestra deve stare tutta nel video
    assert starts[-1] + 40 <= 100 and starts[-1] == 60


def test_video_troppo_corto_nessuna_finestra():
    assert len(window_starts(39)) == 0
    assert list(window_starts(40)) == [0]


def test_etichetta_soglia_20_frame():
    events = [(30, 200, VIOLENZA)]
    assert label_window(0, events, background=NORMALE) is None        # 10 frame di evento: ambigua
    assert label_window(10, events, background=NORMALE) == VIOLENZA   # 20 frame: esattamente la soglia
    assert label_window(200, events, background=NORMALE) == NORMALE   # nessun evento
    assert label_window(200, events, background=None) is None         # sorgente senza sfondo affidabile


def test_eventi_sovrapposti_non_contano_doppio():
    events = [(0, 15, VIOLENZA), (5, 15, VIOLENZA)]
    assert label_window(0, events, background=NORMALE) is None  # 15 frame unici, non 25
