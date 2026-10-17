"""Costanti condivise da estrazione, training e demo (decisioni D2 e D5 nel diario)."""

FPS_ANALISI = 25                 # tutti i video vengono letti a 25 fps
FRAME_PER_FINESTRA = 40          # 1,6 secondi
PASSO_FINESTRE = 10              # finestre 0-39, 10-49, 20-59, ...
FRAME_MINIMI_EVENTO = 20         # frame di evento necessari perché la finestra prenda la classe dell'evento

CLASSI = ("normale", "violenza", "violenza_con_arma", "scippo")
INDICE_CLASSE = {nome: indice for indice, nome in enumerate(CLASSI)}
