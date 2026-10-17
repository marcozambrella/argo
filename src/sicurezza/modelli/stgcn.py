"""ST-GCN (Yan et al., AAAI 2018) in PyTorch puro, come termine di confronto per il Graph-Transformer (D11).

Grafo dello scheletro COCO per ogni persona, partizione "spaziale" (radice, verso il centro, verso l'esterno),
10 blocchi con convoluzione temporale 9, importanza degli archi appresa. Le 4 persone sono elaborate separatamente
e poi unite con max-pool (le persone assenti sono escluse); le armi passano da un ramo a parte e vengono aggiunte
prima del classificatore. Iperparametri del paper originale; PROVVISORI per questo progetto.
"""

import numpy as np
import torch
from torch import nn

from ..caratteristiche import CARATTERISTICHE_PER_NODO, NODI_PERSONE, NUMERO_ARMI, NUMERO_PERSONE
from ..config import CLASSI
from ..scheletro import CENTRO_SCHELETRO, NUMERO_KEYPOINT, OSSA


def distanze_in_salti(numero_nodi: int, archi) -> np.ndarray:
    adiacenza = np.eye(numero_nodi, dtype=bool)
    for i, j in archi:
        adiacenza[i, j] = adiacenza[j, i] = True
    distanze = np.full((numero_nodi, numero_nodi), np.inf)
    raggiunti = np.eye(numero_nodi, dtype=bool)
    distanze[raggiunti] = 0
    for salti in range(1, numero_nodi):
        raggiunti = (raggiunti.astype(int) @ adiacenza.astype(int)) > 0
        distanze[raggiunti & np.isinf(distanze)] = salti
    return distanze


def adiacenza_partizione_spaziale() -> np.ndarray:
    """[3, 17, 17]: sottoinsiemi radice / centripeto / centrifugo, normalizzati per grado."""
    salti = distanze_in_salti(NUMERO_KEYPOINT, OSSA)
    vicini = salti <= 1
    grado = vicini.sum(0)
    normalizzata = vicini / grado[None, :]  # colonna j normalizzata per il grado di j
    distanza_dal_centro = salti[:, CENTRO_SCHELETRO]
    partizioni = np.zeros((3, NUMERO_KEYPOINT, NUMERO_KEYPOINT), np.float32)
    for i in range(NUMERO_KEYPOINT):
        for j in range(NUMERO_KEYPOINT):
            if not vicini[i, j]:
                continue
            if distanza_dal_centro[j] == distanza_dal_centro[i]:
                partizioni[0, j, i] = normalizzata[j, i]
            elif distanza_dal_centro[j] > distanza_dal_centro[i]:
                partizioni[1, j, i] = normalizzata[j, i]
            else:
                partizioni[2, j, i] = normalizzata[j, i]
    return partizioni


class BloccoStGcn(nn.Module):
    def __init__(self, canali_ingresso: int, canali_uscita: int, numero_partizioni: int, passo: int = 1,
                 dropout: float = 0.0, residuo: bool = True, kernel_temporale: int = 9):
        super().__init__()
        self.numero_partizioni = numero_partizioni
        self.convoluzione_grafo = nn.Conv2d(canali_ingresso, canali_uscita * numero_partizioni, kernel_size=1)
        self.convoluzione_tempo = nn.Sequential(
            nn.BatchNorm2d(canali_uscita), nn.ReLU(inplace=True),
            nn.Conv2d(canali_uscita, canali_uscita, (kernel_temporale, 1), (passo, 1),
                      ((kernel_temporale - 1) // 2, 0)),
            nn.BatchNorm2d(canali_uscita), nn.Dropout(dropout, inplace=True))
        if not residuo:
            self.residuo = lambda x: 0
        elif canali_ingresso == canali_uscita and passo == 1:
            self.residuo = lambda x: x
        else:
            self.residuo = nn.Sequential(nn.Conv2d(canali_ingresso, canali_uscita, 1, (passo, 1)),
                                         nn.BatchNorm2d(canali_uscita))
        self.attivazione = nn.ReLU(inplace=True)

    def forward(self, x, adiacenza):
        """x [N, C, T, V], adiacenza [K, V, V]."""
        residuo = self.residuo(x)
        y = self.convoluzione_grafo(x)
        lotto, _, frame, nodi = y.shape
        y = y.view(lotto, self.numero_partizioni, -1, frame, nodi)
        y = torch.einsum("nkctv,kvw->nctw", y, adiacenza)
        return self.attivazione(self.convoluzione_tempo(y) + residuo)


class StGcn(nn.Module):
    def __init__(self, dropout: float = 0.5, numero_classi: int = len(CLASSI), numero_classi_arma: int = 3,
                 dimensione_ramo_armi: int = 64):
        super().__init__()
        self.register_buffer("adiacenza", torch.from_numpy(adiacenza_partizione_spaziale()))
        partizioni = self.adiacenza.shape[0]
        self.normalizzazione_dati = nn.BatchNorm1d(CARATTERISTICHE_PER_NODO * NUMERO_KEYPOINT)
        configurazione = [(CARATTERISTICHE_PER_NODO, 64, 1, False), (64, 64, 1, True), (64, 64, 1, True),
                          (64, 64, 1, True), (64, 128, 2, True), (128, 128, 1, True), (128, 128, 1, True),
                          (128, 256, 2, True), (256, 256, 1, True), (256, 256, 1, True)]
        self.blocchi = nn.ModuleList(
            BloccoStGcn(ingresso, uscita, partizioni, passo, dropout if numero else 0.0, residuo)
            for numero, (ingresso, uscita, passo, residuo) in enumerate(configurazione))
        self.importanza_archi = nn.ParameterList(nn.Parameter(torch.ones_like(self.adiacenza))
                                                 for _ in self.blocchi)

        self.classe_arma = nn.Embedding(numero_classi_arma + 1, 16)
        self.ramo_armi = nn.Sequential(nn.Linear(CARATTERISTICHE_PER_NODO + 16, dimensione_ramo_armi), nn.ReLU(),
                                       nn.Linear(dimensione_ramo_armi, dimensione_ramo_armi))
        self.classificatore = nn.Linear(256 + 2 * dimensione_ramo_armi, numero_classi)

    def forward(self, nodi: torch.Tensor, maschera: torch.Tensor, classe_arma: torch.Tensor) -> torch.Tensor:
        """Stessi ingressi del Graph-Transformer: nodi [B, T, 70, 5], maschera [B, T, 70], classe_arma [B, T, 2]."""
        lotto, frame, _, canali = nodi.shape
        persone = nodi[:, :, :NODI_PERSONE].reshape(lotto, frame, NUMERO_PERSONE, NUMERO_KEYPOINT, canali)
        x = persone.permute(0, 2, 3, 4, 1).reshape(lotto * NUMERO_PERSONE, NUMERO_KEYPOINT * canali, frame)
        x = self.normalizzazione_dati(x)
        x = x.view(lotto * NUMERO_PERSONE, NUMERO_KEYPOINT, canali, frame).permute(0, 2, 3, 1)  # [N*M, C, T, V]
        for blocco, importanza in zip(self.blocchi, self.importanza_archi):
            x = blocco(x, self.adiacenza * importanza)
        per_persona = x.mean((2, 3)).view(lotto, NUMERO_PERSONE, -1)
        persona_presente = maschera[:, :, :NODI_PERSONE].reshape(lotto, frame, NUMERO_PERSONE, -1).any(-1).any(1)
        per_persona = per_persona.masked_fill(~persona_presente[..., None], float("-inf")).amax(1)
        per_persona = torch.where(torch.isfinite(per_persona), per_persona, torch.zeros_like(per_persona))

        armi = torch.cat([nodi[:, :, NODI_PERSONE:], self.classe_arma(classe_arma.long())], dim=-1)
        armi = self.ramo_armi(armi) * maschera[:, :, NODI_PERSONE:, None]                  # [B, T, 2, D]
        armi_presenti = maschera[:, :, NODI_PERSONE:].float().sum((1, 2)).clamp(min=1.0)[:, None]
        sintesi_armi = torch.cat([armi.sum((1, 2)) / armi_presenti, armi.amax((1, 2))], dim=-1)
        return self.classificatore(torch.cat([per_persona, sintesi_armi], dim=-1))
