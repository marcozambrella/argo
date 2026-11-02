# ===== avvio comune: uguale in tutti i kernel di training/ =====
# Kaggle carica un solo file per kernel, quindi questo blocco è copiato in testa a ognuno.
import os
import shutil
import subprocess
import sys
URL_REPO = "https://github.com/marcozambrella/argo"
# D68: gli esperimenti usano il branch esperimenti-armi-scippi, non main
subprocess.run(["git", "clone", "--depth", "1", "--branch", "esperimenti-armi-scippi", URL_REPO, "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4", "lap"], check=True)
sys.path.insert(0, "/tmp/repo/src")
commit = subprocess.run(["git", "-C", "/tmp/repo", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
print("commit:", commit.stdout.strip(), flush=True)
if shutil.which("ffprobe") is None:  # il lettore video usa ffmpeg/ffprobe: build statica se l'immagine non li ha
    subprocess.run("curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | "
                   "tar -xJ -C /tmp && cp /tmp/ffmpeg-*-static/ff* /usr/local/bin/", shell=True, check=True)
# i comandi `python -m sicurezza...` lanciati dai kernel trovano il pacchetto in /tmp/repo/src
ambiente = dict(os.environ)
ambiente["PYTHONPATH"] = "/tmp/repo/src"
# =====

"""D68 strada A, GPU: classificatore delle armi in mano sui ritagli presi dai video (kernel ritagli_video).
Ogni ritaglio passa in una ResNet-18 (ImageNet); la finestra è armata se lo è almeno una mano in almeno un istante
(massimo su mani e istanti: sappiamo che la finestra è armata, non quale mano). Varianti (richiesta di Marco:
strati convoluzionali in più, colori contro bianco e nero):
    a1: ResNet-18 → testa su ogni ritaglio
    a2: + 2 strati conv 1D nel tempo sulla sequenza di ogni mano
    a3: + 2 strati conv 2D in più sulle mappe della ResNet, poi i conv nel tempo
ognuna a colori (v1: anche in bianco e nero), un thread per GPU, varianti di seguito, EPOCHE fisse
(niente early stop). Uscite:
arma, fuoco, coltello (fuoco/coltello non addestrati sulle armi di tipo ignoto). Per ogni modello si salvano la
migliore sul val e l'ultima, e i punteggi di val e test per `demo valuta-armi-scippi`.
02/11: la v1 con multiprocessing (fork dopo la prova su CPU) non ha stampato nessuna epoca in 3,7 h: fork dopo che
torch ha avviato i suoi thread della CPU può bloccare i figli. Ora un thread per GPU (niente fork) e una stampa ogni
100 lotti."""
import json
import threading
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sicurezza.dati import INDICE_CLASSE, FILE_TIPO_ARMA_SCVD, tipi_arma_scvd, tipi_arma_ucf
from sicurezza.demo import tipi_della_clip, salva_punteggi

EPOCHE = 12
LOTTO = 32
LATO_RETE = 112           # i ritagli da 56 px si ingrandiscono: la ResNet vede meglio
USCITE = ("arma", "fuoco", "coltello")
# 02/11 Marco: per la quota rimasta solo a colori, 12 epoche (il grigio contro i colori si vede con la rete video b3)
VARIANTI_PER_GPU = (("a1_colori", "a2_colori"), ("a3_colori",))
MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DEVIAZIONE = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
ingresso = Path("/kaggle/input")
uscita = Path("/kaggle/working")


class ModelloMani(nn.Module):
    def __init__(self, conv_tempo: bool, conv_in_piu: bool):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.corpo = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1,
                                   resnet.layer2, resnet.layer3, resnet.layer4)
        dimensione = 512
        self.conv_in_piu = None
        if conv_in_piu:
            self.conv_in_piu = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                             nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
            dimensione = 256
        self.conv_tempo = None
        if conv_tempo:
            self.conv_tempo = nn.Sequential(nn.Conv1d(dimensione, 256, 3, padding=1), nn.ReLU(),
                                            nn.Conv1d(256, 256, 3, padding=1), nn.ReLU())
            dimensione = 256
        self.testa = nn.Linear(dimensione, len(USCITE))

    def forward(self, ritagli: torch.Tensor, presenti: torch.Tensor) -> torch.Tensor:
        """ritagli [B, istanti, mani, 3, L, L] normalizzati, presenti [B, istanti, mani] → logit [B, 3]."""
        lotto, istanti, mani = presenti.shape
        mappe = self.corpo(ritagli.flatten(0, 2))
        if self.conv_in_piu is not None:
            mappe = self.conv_in_piu(mappe)
        caratteristiche = mappe.mean(dim=(2, 3)).view(lotto, istanti, mani, -1)
        caratteristiche = caratteristiche * presenti.unsqueeze(-1)   # una mano assente non conta
        if self.conv_tempo is not None:
            # ogni mano è una sequenza nel tempo: [B * mani, dimensione, istanti]
            sequenze = caratteristiche.permute(0, 2, 3, 1).flatten(0, 1)
            sequenze = self.conv_tempo(sequenze)
            caratteristiche = sequenze.view(lotto, mani, -1, istanti).permute(0, 3, 1, 2)
        logit = self.testa(caratteristiche)                                   # [B, istanti, mani, 3]
        logit = logit.masked_fill(~presenti.unsqueeze(-1), -1e4)             # senza mani: nessuna arma vista
        return logit.amax(dim=(1, 2))


def prepara_lotto(ritagli: np.ndarray, grigio: bool, allena: bool, dispositivo) -> torch.Tensor:
    """[B, istanti, mani, L, L, 3] uint8 → [B, istanti, mani, 3, LATO_RETE, LATO_RETE] normalizzato."""
    forma = ritagli.shape
    immagini = torch.from_numpy(np.ascontiguousarray(ritagli)).to(dispositivo)
    immagini = immagini.view(-1, forma[3], forma[4], 3).permute(0, 3, 1, 2).float() / 255
    if allena and np.random.random() < 0.5:
        immagini = immagini.flip(3)   # specchio
    if grigio:
        valore = 0.299 * immagini[:, 0:1] + 0.587 * immagini[:, 1:2] + 0.114 * immagini[:, 2:3]
        immagini = valore.repeat(1, 3, 1, 1)
    immagini = F.interpolate(immagini, size=(LATO_RETE, LATO_RETE), mode="bilinear", align_corners=False)
    immagini = (immagini - MEDIA.to(dispositivo)) / DEVIAZIONE.to(dispositivo)
    return immagini.view(forma[0], forma[1], forma[2], 3, LATO_RETE, LATO_RETE)


@torch.no_grad()
def probabilita(modello, ritagli, presenti, grigio: bool, dispositivo) -> np.ndarray:
    modello.eval()
    uscite = []
    for primo in range(0, len(presenti), LOTTO):
        ingresso_rete = prepara_lotto(ritagli[primo:primo + LOTTO], grigio, False, dispositivo)
        mani = torch.from_numpy(presenti[primo:primo + LOTTO]).to(dispositivo)
        uscite.append(torch.sigmoid(modello(ingresso_rete, mani)).cpu().numpy())
    return np.concatenate(uscite)


def bilanciata_media(probabilita_val: np.ndarray, bersagli: np.ndarray, maschere: np.ndarray) -> float:
    valori = []
    for numero in range(len(USCITE)):
        valide = maschere[:, numero] > 0
        positivi = valide & (bersagli[:, numero] > 0.5)
        negativi = valide & (bersagli[:, numero] < 0.5)
        if positivi.sum() == 0 or negativi.sum() == 0:
            continue
        previsti = probabilita_val[:, numero] > 0.5
        valori.append((previsti[positivi].mean() + (~previsti[negativi]).mean()) / 2)
    return float(np.mean(valori))


def carica_dati() -> dict:
    cartella = next(ingresso.glob("**/ritagli_train.npy")).parent
    percorso_finestre = next(ingresso.glob("**/finestre.npz"))
    with np.load(percorso_finestre) as finestre:
        campi = {"id_clip": finestre["id_clip"], "inizio_finestra": finestre["inizio_finestra"],
                 "etichetta": finestre["etichetta"], "sorgente": finestre["sorgente"]}
    tipi_ucf = tipi_arma_ucf()
    tipi_scvd = tipi_arma_scvd(FILE_TIPO_ARMA_SCVD)
    dati = {"campi": campi}
    for nome in ("train", "val", "test"):
        righe = np.load(cartella / f"righe_{nome}.npy")
        armata = campi["etichetta"][righe] == INDICE_CLASSE["violenza_con_arma"]
        bersagli = np.zeros((len(righe), len(USCITE)), np.float32)
        maschere = np.ones((len(righe), len(USCITE)), np.float32)
        bersagli[:, 0] = armata
        for posizione in np.where(armata)[0]:
            tipi = tipi_della_clip(str(campi["id_clip"][righe[posizione]]), tipi_ucf, tipi_scvd)
            if len(tipi) == 0:
                maschere[posizione, 1:] = 0
            if "fuoco" in tipi:
                bersagli[posizione, 1] = 1
            if "coltello" in tipi:
                bersagli[posizione, 2] = 1
        dati[nome] = {"righe": righe, "bersagli": bersagli, "maschere": maschere,
                      "ritagli": np.load(cartella / f"ritagli_{nome}.npy", mmap_mode="r"),
                      "presenti": np.load(cartella / f"presenti_{nome}.npy")}
    # il train sta in memoria una volta sola (~8 GB), condiviso dai due thread
    dati["train"]["ritagli"] = np.asarray(dati["train"]["ritagli"])
    return dati


def addestra_variante(nome: str, dati: dict, dispositivo) -> None:
    grigio = nome.endswith("grigio")
    tipo = nome.split("_")[0]
    modello = ModelloMani(conv_tempo=tipo in ("a2", "a3"), conv_in_piu=tipo == "a3").to(dispositivo)
    ottimizzatore = torch.optim.AdamW(modello.parameters(), lr=3e-4, weight_decay=1e-4)
    train = dati["train"]
    passi_per_epoca = len(train["righe"]) // LOTTO
    pianificatore = torch.optim.lr_scheduler.CosineAnnealingLR(ottimizzatore, EPOCHE * passi_per_epoca)
    ritagli_train = train["ritagli"]
    cartella = uscita / nome
    cartella.mkdir(exist_ok=True)
    storia = []
    migliore = -1.0
    for epoca in range(1, EPOCHE + 1):
        inizio = time.time()
        modello.train()
        ordine = np.random.permutation(len(train["righe"]))
        perdite = []
        for passo in range(passi_per_epoca):
            scelte = np.sort(ordine[passo * LOTTO:(passo + 1) * LOTTO])
            ingresso_rete = prepara_lotto(ritagli_train[scelte], grigio, True, dispositivo)
            mani = torch.from_numpy(train["presenti"][scelte]).to(dispositivo)
            bersagli = torch.from_numpy(train["bersagli"][scelte]).to(dispositivo)
            maschere = torch.from_numpy(train["maschere"][scelte]).to(dispositivo)
            perdita = F.binary_cross_entropy_with_logits(modello(ingresso_rete, mani), bersagli, reduction="none")
            perdita = (perdita * maschere).sum() / maschere.sum().clamp(min=1)
            ottimizzatore.zero_grad(set_to_none=True)
            perdita.backward()
            ottimizzatore.step()
            pianificatore.step()
            perdite.append(perdita.item())
            if passo % 100 == 0:
                print(f"{nome} epoca {epoca}: lotto {passo}/{passi_per_epoca} ({time.time() - inizio:.0f} s)",
                      flush=True)
        val = dati["val"]
        probabilita_val = probabilita(modello, val["ritagli"], val["presenti"], grigio, dispositivo)
        valore = bilanciata_media(probabilita_val, val["bersagli"], val["maschere"])
        storia.append({"epoca": epoca, "perdita": float(np.mean(perdite)), "bilanciata_val": valore,
                       "secondi": time.time() - inizio})
        print(f"{nome} epoca {epoca}: perdita {np.mean(perdite):.4f}, bilanciata val {valore:.4f} "
              f"({time.time() - inizio:.0f} s)", flush=True)
        if valore > migliore:
            migliore = valore
            torch.save(modello.state_dict(), cartella / "migliore.pt")
    torch.save(modello.state_dict(), cartella / "ultima.pt")
    (cartella / "storia.json").write_text(json.dumps(storia, indent=1))
    campi = dati["campi"]
    for nome_pesi in ("migliore", "ultima"):
        modello.load_state_dict(torch.load(cartella / f"{nome_pesi}.pt", map_location=dispositivo))
        cartella_punteggi = cartella / f"punteggi_{nome_pesi}"
        cartella_punteggi.mkdir(exist_ok=True)
        for nome_split in ("val", "test"):
            parte = dati[nome_split]
            valori = probabilita(modello, parte["ritagli"], parte["presenti"], grigio, dispositivo)
            finestre_scelte = {}
            for chiave, colonna in campi.items():
                finestre_scelte[chiave] = colonna[parte["righe"]]
            punteggi = {}
            for numero, nome_uscita in enumerate(USCITE):
                punteggi[nome_uscita] = valori[:, numero]
            salva_punteggi(cartella_punteggi / f"punteggi_{nome_split}.npz", finestre_scelte, punteggi)


def lavora_su_una_gpu(numero_gpu: int, dati: dict) -> None:
    dispositivo = torch.device(f"cuda:{numero_gpu}")
    for nome in VARIANTI_PER_GPU[numero_gpu]:
        addestra_variante(nome, dati, dispositivo)


if __name__ == "__main__":
    # prova di un passo su CPU con dati finti: un errore di forme si vede subito, prima delle ore di GPU
    for tipo in ("a1", "a2", "a3"):
        prova = ModelloMani(conv_tempo=tipo != "a1", conv_in_piu=tipo == "a3")
        finti = prepara_lotto(np.zeros((2, 6, 8, 56, 56, 3), np.uint8), True, True, torch.device("cpu"))
        presenti = torch.zeros(2, 6, 8, dtype=torch.bool)
        presenti[0, :, 1] = True
        print(tipo, "prova:", tuple(prova(finti, presenti).shape), flush=True)
    dati_comuni = carica_dati()
    print("dati caricati", flush=True)
    # un thread per GPU: torch lascia libero il GIL durante i calcoli sulla GPU
    lavori = []
    for numero_gpu in range(len(VARIANTI_PER_GPU)):
        lavoro = threading.Thread(target=lavora_su_una_gpu, args=(numero_gpu, dati_comuni))
        lavoro.start()
        lavori.append(lavoro)
    for lavoro in lavori:
        lavoro.join()
    print("fatto", flush=True)
