"""Graph-Transformer sulle finestre-grafo (D11): relazioni tra keypoint della stessa persona, tra persone diverse
e tra persone e armi.

Ogni blocco fa:
  1. attenzione SPAZIALE tra i 70 nodi di ogni frame, con un bias appreso per tipo di arco (osso, stessa persona,
     persone diverse, arma-polso, ...) e un bias che penalizza la distanza per gli archi tra entità diverse;
  2. attenzione TEMPORALE di ogni nodo lungo i 40 frame;
  3. rete feed-forward.
Iperparametri PROVVISORI (dimensione 128, 4 teste, 4 blocchi): da decidere con Marco dopo i primi esperimenti.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..caratteristiche import CARATTERISTICHE_PER_NODO, NODI_PERSONE, NODI_PER_FRAME, NUMERO_ARMI, NUMERO_PERSONE
from ..config import CLASSI, FRAME_PER_FINESTRA
from ..scheletro import NUMERO_KEYPOINT, OSSA, POLSI

# tipi di arco tra due nodi dello stesso frame
ARCO_SE_STESSO, ARCO_OSSO, ARCO_STESSA_PERSONA, ARCO_PERSONE_DIVERSE, ARCO_ARMA_POLSO, ARCO_ARMA_CORPO, \
    ARCO_ARMA_ARMA = range(7)
NUMERO_TIPI_ARCO = 7
ARCHI_TRA_ENTITA_DIVERSE = (ARCO_PERSONE_DIVERSE, ARCO_ARMA_POLSO, ARCO_ARMA_CORPO, ARCO_ARMA_ARMA)


def matrice_tipi_arco() -> torch.Tensor:
    """[70, 70] con il tipo di arco tra ogni coppia di nodi."""
    persona_del_nodo = [nodo // NUMERO_KEYPOINT for nodo in range(NODI_PERSONE)] + [-1] * NUMERO_ARMI
    keypoint_del_nodo = [nodo % NUMERO_KEYPOINT for nodo in range(NODI_PERSONE)] + [-1] * NUMERO_ARMI
    ossa = {frozenset(osso) for osso in OSSA}
    tipi = torch.empty(NODI_PER_FRAME, NODI_PER_FRAME, dtype=torch.long)
    for i in range(NODI_PER_FRAME):
        for j in range(NODI_PER_FRAME):
            arma_i, arma_j = persona_del_nodo[i] < 0, persona_del_nodo[j] < 0
            if i == j:
                tipo = ARCO_SE_STESSO
            elif arma_i and arma_j:
                tipo = ARCO_ARMA_ARMA
            elif arma_i or arma_j:
                keypoint_persona = keypoint_del_nodo[j] if arma_i else keypoint_del_nodo[i]
                tipo = ARCO_ARMA_POLSO if keypoint_persona in POLSI else ARCO_ARMA_CORPO
            elif persona_del_nodo[i] != persona_del_nodo[j]:
                tipo = ARCO_PERSONE_DIVERSE
            elif frozenset((keypoint_del_nodo[i], keypoint_del_nodo[j])) in ossa:
                tipo = ARCO_OSSO
            else:
                tipo = ARCO_STESSA_PERSONA
            tipi[i, j] = tipo
    return tipi


def maschera_attenzione(chiavi_valide: torch.Tensor) -> torch.Tensor:
    """[gruppi, L] bool → [gruppi, 1, L, L] float con -inf sulle chiavi non valide.
    Ogni nodo vede sempre se stesso, così una riga tutta mascherata non produce NaN."""
    lunghezza = chiavi_valide.shape[-1]
    vede = chiavi_valide[:, None, :] | torch.eye(lunghezza, dtype=torch.bool, device=chiavi_valide.device)
    return torch.zeros(vede.shape, device=chiavi_valide.device).masked_fill(~vede, float("-inf"))[:, None]


class BloccoSpazioTempo(nn.Module):
    def __init__(self, dimensione: int, teste: int, dropout: float):
        super().__init__()
        self.teste = teste
        self.norma_spazio = nn.LayerNorm(dimensione)
        self.qkv_spazio = nn.Linear(dimensione, 3 * dimensione)
        self.uscita_spazio = nn.Linear(dimensione, dimensione)
        self.bias_tipo_arco = nn.Parameter(torch.zeros(teste, NUMERO_TIPI_ARCO))
        self.peso_distanza = nn.Parameter(torch.zeros(teste))   # softplus(0) ≈ 0,69 per testa all'inizio
        self.norma_tempo = nn.LayerNorm(dimensione)
        self.qkv_tempo = nn.Linear(dimensione, 3 * dimensione)
        self.uscita_tempo = nn.Linear(dimensione, dimensione)
        self.norma_ff = nn.LayerNorm(dimensione)
        self.feed_forward = nn.Sequential(nn.Linear(dimensione, 4 * dimensione), nn.GELU(), nn.Dropout(dropout),
                                          nn.Linear(4 * dimensione, dimensione))
        self.dropout = nn.Dropout(dropout)

    def attenzione(self, x, qkv, uscita, bias):
        gruppi, lunghezza, dimensione = x.shape
        q, k, v = qkv(x).view(gruppi, lunghezza, 3, self.teste, -1).permute(2, 0, 3, 1, 4)
        risultato = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                                   dropout_p=self.dropout.p if self.training else 0.0)
        return uscita(risultato.transpose(1, 2).reshape(gruppi, lunghezza, dimensione))

    def forward(self, x, maschera, tipi_arco, distanze):
        """x [B, T, N, D], maschera [B, T, N], tipi_arco [N, N], distanze [B, T, N, N]."""
        lotto, frame, nodi, dimensione = x.shape

        # 1. spazio: bias per tipo di arco + penalità di distanza sugli archi tra entità diverse
        bias_tipo = self.bias_tipo_arco[:, tipi_arco]                                     # [H, N, N]
        tra_entita = torch.zeros_like(tipi_arco, dtype=torch.bool)
        for tipo in ARCHI_TRA_ENTITA_DIVERSE:
            tra_entita |= tipi_arco == tipo
        penalita = -F.softplus(self.peso_distanza)[None, :, None, None] * (distanze * tra_entita).reshape(
            lotto * frame, 1, nodi, nodi)                                                # [B*T, H, N, N]
        bias = bias_tipo[None] + penalita + maschera_attenzione(maschera.reshape(lotto * frame, nodi))
        spazio = self.norma_spazio(x).reshape(lotto * frame, nodi, dimensione)
        x = x + self.dropout(self.attenzione(spazio, self.qkv_spazio, self.uscita_spazio, bias)).view_as(x)

        # 2. tempo: ogni nodo guarda se stesso negli altri frame in cui è presente
        tempo = self.norma_tempo(x).transpose(1, 2).reshape(lotto * nodi, frame, dimensione)
        bias_tempo = maschera_attenzione(maschera.transpose(1, 2).reshape(lotto * nodi, frame))
        aggiornamento = self.attenzione(tempo, self.qkv_tempo, self.uscita_tempo, bias_tempo)
        x = x + self.dropout(aggiornamento.view(lotto, nodi, frame, dimensione).transpose(1, 2))

        # 3. feed-forward
        return x + self.dropout(self.feed_forward(self.norma_ff(x)))


class GraphTransformer(nn.Module):
    def __init__(self, dimensione: int = 128, teste: int = 4, blocchi: int = 4, dropout: float = 0.1,
                 numero_classi: int = len(CLASSI), numero_classi_arma: int = 3):
        super().__init__()
        self.proiezione_ingresso = nn.Linear(CARATTERISTICHE_PER_NODO, dimensione)
        self.tipo_nodo = nn.Embedding(NUMERO_KEYPOINT + 1, dimensione)        # 17 keypoint + "arma"
        self.posto_nodo = nn.Embedding(NUMERO_PERSONE + 1, dimensione)        # 4 persone + "posto armi"
        self.classe_arma = nn.Embedding(numero_classi_arma + 1, dimensione)   # 0 = nessuna arma
        self.posizione_temporale = nn.Parameter(torch.zeros(FRAME_PER_FINESTRA, dimensione))
        nn.init.normal_(self.posizione_temporale, std=0.02)
        self.blocchi = nn.ModuleList(BloccoSpazioTempo(dimensione, teste, dropout) for _ in range(blocchi))
        self.norma_finale = nn.LayerNorm(dimensione)
        self.testa = nn.Sequential(nn.Linear(2 * dimensione, dimensione), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(dimensione, numero_classi))

        indice_nodo = torch.arange(NODI_PER_FRAME)
        self.register_buffer("tipo_di_ogni_nodo", torch.where(indice_nodo < NODI_PERSONE,
                                                              indice_nodo % NUMERO_KEYPOINT, NUMERO_KEYPOINT))
        self.register_buffer("posto_di_ogni_nodo", torch.where(indice_nodo < NODI_PERSONE,
                                                               indice_nodo // NUMERO_KEYPOINT, NUMERO_PERSONE))
        self.register_buffer("tipi_arco", matrice_tipi_arco())

    def forward(self, nodi: torch.Tensor, maschera: torch.Tensor, classe_arma: torch.Tensor) -> torch.Tensor:
        """nodi [B, T, 70, 5], maschera [B, T, 70] bool, classe_arma [B, T, 2] → logit [B, classi]."""
        lotto, frame, _, _ = nodi.shape
        x = self.proiezione_ingresso(nodi) + self.tipo_nodo(self.tipo_di_ogni_nodo) \
            + self.posto_nodo(self.posto_di_ogni_nodo) + self.posizione_temporale[:frame, None, :]
        classe_per_nodo = torch.zeros(lotto, frame, NODI_PER_FRAME, dtype=torch.long, device=nodi.device)
        classe_per_nodo[:, :, NODI_PERSONE:] = classe_arma.long()
        x = x + self.classe_arma(classe_per_nodo) * (classe_per_nodo > 0)[..., None]

        posizioni = nodi[..., :2]
        distanze = torch.cdist(posizioni.reshape(lotto * frame, NODI_PER_FRAME, 2),
                               posizioni.reshape(lotto * frame, NODI_PER_FRAME, 2)).view(
            lotto, frame, NODI_PER_FRAME, NODI_PER_FRAME)
        for blocco in self.blocchi:
            x = blocco(x, maschera, self.tipi_arco, distanze)

        x = self.norma_finale(x)
        peso = maschera[..., None].float()
        media = (x * peso).sum((1, 2)) / peso.sum((1, 2)).clamp(min=1.0)
        massimo = x.masked_fill(~maschera[..., None], float("-inf")).amax((1, 2))
        massimo = torch.where(torch.isfinite(massimo), massimo, torch.zeros_like(massimo))  # scena vuota
        return self.testa(torch.cat([media, massimo], dim=-1))
