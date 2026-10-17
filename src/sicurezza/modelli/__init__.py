"""Classificatori delle finestre: stesso ingresso (nodi, maschera, classe_arma), stessa uscita (logit 4 classi)."""

from .graph_transformer import GraphTransformer
from .stgcn import StGcn

MODELLI = {"graph_transformer": GraphTransformer, "stgcn": StGcn}


def crea_modello(nome: str, **iperparametri):
    return MODELLI[nome](**iperparametri)


def conta_parametri(modello) -> int:
    return sum(parametro.numel() for parametro in modello.parameters() if parametro.requires_grad)
