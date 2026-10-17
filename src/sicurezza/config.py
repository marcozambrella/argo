"""Costanti condivise da estrazione, training e demo (decisioni D2 e D5 nel README)."""

FPS = 25            # tutti i video vengono letti a 25 fps
WINDOW = 40         # frame per finestra (1,6 s)
STRIDE = 10         # passo tra finestre: 0-39, 10-49, 20-59, ...
MIN_COVERAGE = 20   # frame di evento necessari perché la finestra prenda la classe dell'evento

CLASSES = ("normale", "violenza", "violenza_con_arma", "scippo")
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}
