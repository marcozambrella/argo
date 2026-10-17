"""Data augmentation sui lotti di finestre (su GPU). Probabilità e ampiezze PROVVISORIE."""

import torch

from .caratteristiche import NODI_PERSONE, NUMERO_PERSONE
from .scheletro import NUMERO_KEYPOINT, SPECCHIO_KEYPOINT

PROBABILITA_RIBALTAMENTO = 0.5
INTERVALLO_SCALA = (0.9, 1.1)
TRASLAZIONE_MASSIMA = 0.2          # in altezze-corpo
PROBABILITA_SPEGNI_KEYPOINT = 0.05


def aumenta_lotto(nodi: torch.Tensor, maschera: torch.Tensor, classe_arma: torch.Tensor):
    """nodi [B, T, 70, 5] (x, y, confidenza, vx, vy). Ritorna nuovi tensori, gli originali non vengono toccati."""
    nodi, maschera, classe_arma = nodi.clone(), maschera.clone(), classe_arma.clone()
    lotto, frame = nodi.shape[:2]
    dispositivo = nodi.device
    visibile = nodi[..., 2] > 0

    # 1. ribaltamento orizzontale: x e vx cambiano segno, sinistra e destra si scambiano
    ribalta = torch.rand(lotto, device=dispositivo) < PROBABILITA_RIBALTAMENTO
    if ribalta.any():
        specchio = torch.tensor(SPECCHIO_KEYPOINT, device=dispositivo)
        persone = nodi[ribalta, :, :NODI_PERSONE].view(-1, frame, NUMERO_PERSONE, NUMERO_KEYPOINT, 5)
        nodi[ribalta, :, :NODI_PERSONE] = persone[:, :, :, specchio].reshape(-1, frame, NODI_PERSONE, 5)
        nodi[ribalta, ..., 0] *= -1
        nodi[ribalta, ..., 3] *= -1
        visibile = nodi[..., 2] > 0

    # 2. scala e traslazione (solo sui nodi visibili: quelli invisibili restano a zero)
    scala = torch.empty(lotto, 1, 1, 1, device=dispositivo).uniform_(*INTERVALLO_SCALA)
    traslazione = torch.empty(lotto, 1, 1, 2, device=dispositivo).uniform_(-TRASLAZIONE_MASSIMA, TRASLAZIONE_MASSIMA)
    posizioni = nodi[..., 0:2] * scala + traslazione
    nodi[..., 0:2] = torch.where(visibile[..., None], posizioni, nodi[..., 0:2])
    nodi[..., 3:5] *= scala

    # 3. keypoint spenti a caso (come se YOLO non li vedesse)
    spegni = (torch.rand(lotto, frame, NODI_PERSONE, device=dispositivo) < PROBABILITA_SPEGNI_KEYPOINT)
    nodi[:, :, :NODI_PERSONE][spegni] = 0.0

    # 4. ordine delle persone casuale: il modello non deve dipendere dal "posto"
    for indice in range(lotto):
        ordine = torch.randperm(NUMERO_PERSONE, device=dispositivo)
        persone = nodi[indice, :, :NODI_PERSONE].view(frame, NUMERO_PERSONE, NUMERO_KEYPOINT, 5)
        nodi[indice, :, :NODI_PERSONE] = persone[:, ordine].reshape(frame, NODI_PERSONE, 5)
        maschera_persone = maschera[indice, :, :NODI_PERSONE].view(frame, NUMERO_PERSONE, NUMERO_KEYPOINT)
        maschera[indice, :, :NODI_PERSONE] = maschera_persone[:, ordine].reshape(frame, NODI_PERSONE)
    return nodi, maschera, classe_arma
