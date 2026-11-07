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
sys.stdout.reconfigure(line_buffering=True)   # anche le stampe senza flush arrivano subito nel log

"""D68 strada B, GPU: R(2+1)D-18 (Kinetics-400) fine-tunata con 5 uscite sì/no indipendenti (violenza, arma, arma
da fuoco, coltello, scippo) invece della sola violenza di training/finetune_rgb. Le armi di tipo ignoto (UCF
"weapon", WVD mazze e martelli) non addestrano le uscite fuoco/coltello. Varianti (VARIANTE, la cambia
lancia_variante.sh): b1 = 112 px a colori come oggi; b2 = 224 px a colori; b3 = 224 px in bianco e nero.
Marco: niente early stop → tutte le epoche; si salvano la migliore sul val e l'ultima. Alla fine: punteggi su una
finestra ogni due di val e test (punteggi_val.npz, punteggi_test.npz per `demo valuta-armi-scippi`), per la
migliore e per l'ultima.
D69: b1a = b1 con una sesta uscita "anomalia" (violenza, arma, scippo o un'altra anomalia rilevante) e le finestre
del dataset argo-anomalie (NWPU Campus, CHAD, ATMA-V: tipo segnato da Marco, normali delle stesse telecamere; le
uscite che lo spezzone non dice restano mascherate)."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torchvision
from sicurezza.dati import INDICE_CLASSE, FILE_TIPO_ARMA_SCVD, prepara_video_esperimenti, tipi_arma_scvd, tipi_arma_ucf
from sicurezza.dati import ClipAnnotata, categoria_ucf
from sicurezza.demo import tipi_della_clip, salva_punteggi

VARIANTE = "b1"
LATI = {"b1": 112, "b2": 224, "b3": 224, "b1a": 112, "b5": 192}
LATO = LATI[VARIANTE]
LATO_X = LATO
if VARIANTE == "b5":   # goal 80% G4: tutto il fotogramma in 4:3 (192×256), non il quadrato centrale; stessi pixel di b2
    LATO_X = LATO * 4 // 3
GRIGIO = VARIANTE == "b3"
EPOCHE = {"b1": 8, "b2": 5, "b3": 5, "b1a": 8, "b5": 5}[VARIANTE]
ALTEZZA = LATO * 8 // 7          # 128 per 112, 256 per 224: margine per il ritaglio a caso
LARGHEZZA = ALTEZZA * 4 // 3 + (ALTEZZA * 4 // 3) % 2   # 4:3 come la maggior parte delle CCTV (171 → 170 per 112)
ORE_MASSIME = 11.0
FINESTRE_PER_GRUPPO = {"arma": 4000, "scippo": 4000, "violenza": 3000, "normale": 3000,
                       "anomalie_nuove": 4000, "normali_nuove": 5000}   # le ultime due solo in b1a
# D69: normali_nuove 5000 e non 3000: con i video normali di CHAD (corse senza motivo, come in metro) NWPU ne avrebbe ~700
FINESTRE_VAL = 2000
LOTTO = {"b1": 32, "b2": 8, "b3": 8, "b1a": 32, "b5": 8}[VARIANTE]   # a 224 px anche 16 va fuori memoria sulle T4 (02/11, b2 v1)
FRAME_PER_BLOCCO = 600   # punteggi finali: le clip lunghe si decodificano a blocchi da 24 s
CON_ANOMALIE = VARIANTE == "b1a"
USCITE = ("violenza", "arma", "fuoco", "coltello", "scippo")
if CON_ANOMALIE:
    USCITE = ("violenza", "arma", "fuoco", "coltello", "scippo", "anomalia")
ETICHETTA_DEL_TIPO = {"normale": INDICE_CLASSE["normale"], "violenza": INDICE_CLASSE["violenza"],
                      "arma": INDICE_CLASSE["violenza_con_arma"], "scippo": INDICE_CLASSE["scippo"],
                      "furto": -1, "altra": -1}   # furto e altra: anomalie senza una classe dei 4
MEDIA = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
DEVIAZIONE = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)
avvio = time.time()


def trascorse() -> float:
    return (time.time() - avvio) / 3600


def sedici(frame: np.ndarray) -> np.ndarray:
    """16 frame distribuiti in modo uniforme tra quelli dati."""
    return frame[np.linspace(0, len(frame) - 1, 16).round().astype(int)]


def ritaglia(finestre: np.ndarray, allena: bool, generatore) -> torch.Tensor:
    """[B, 16, ALTEZZA, LARGHEZZA, 3] uint8 → [B, 3, 16, LATO, LATO] normalizzato; in train ritaglio a caso e
    specchio. In bianco e nero (b3) il grigio va copiato sui 3 canali: la rete preaddestrata ne vuole 3."""
    pezzi = []
    for finestra in finestre:
        if allena:
            y = generatore.integers(0, ALTEZZA - LATO + 1)
            x = generatore.integers(0, LARGHEZZA - LATO_X + 1)
            pezzo = finestra[:, y:y + LATO, x:x + LATO_X]
            if generatore.random() < 0.5:
                pezzo = pezzo[:, :, ::-1]
        else:
            y = (ALTEZZA - LATO) // 2
            x = (LARGHEZZA - LATO_X) // 2
            pezzo = finestra[:, y:y + LATO, x:x + LATO_X]
        pezzi.append(pezzo)
    lotto = torch.from_numpy(np.ascontiguousarray(np.stack(pezzi))).permute(0, 4, 1, 2, 3).float() / 255
    if GRIGIO:
        grigio = 0.299 * lotto[:, 0:1] + 0.587 * lotto[:, 1:2] + 0.114 * lotto[:, 2:3]
        lotto = grigio.repeat(1, 3, 1, 1, 1)
    return ((lotto - MEDIA) / DEVIAZIONE).contiguous()


class Finestre(torch.utils.data.Dataset):
    """Finestre già decodificate in un .npy su disco, con bersagli [5] e maschera [5] (0 = uscita non addestrata)."""
    def __init__(self, percorso: str, posizioni: np.ndarray, bersagli: np.ndarray, maschere: np.ndarray):
        self.percorso = percorso
        self.posizioni = posizioni
        self.bersagli = bersagli
        self.maschere = maschere
        self.archivio = None

    def __len__(self):
        return len(self.posizioni)

    def __getitem__(self, numero):
        if self.archivio is None:
            self.archivio = np.load(self.percorso, mmap_mode="r")
        return np.asarray(self.archivio[self.posizioni[numero]]), self.bersagli[numero], self.maschere[numero]


def unisci_lotto(elementi, allena: bool, generatore):
    finestre = []
    bersagli = []
    maschere = []
    for finestra, bersaglio, maschera in elementi:
        finestre.append(finestra)
        bersagli.append(bersaglio)
        maschere.append(maschera)
    return (ritaglia(np.stack(finestre), allena, generatore), torch.tensor(np.stack(bersagli)),
            torch.tensor(np.stack(maschere)))


def unisci_lotto_train(elementi):
    return unisci_lotto(elementi, True, np.random.default_rng())


def unisci_lotto_val(elementi):
    return unisci_lotto(elementi, False, None)


def crea_rete() -> torch.nn.Module:
    rete = torchvision.models.video.r2plus1d_18(weights=torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1)
    rete.fc = torch.nn.Linear(512, len(USCITE))
    return rete


def un_epoca(rete, caricatore, ottimizzatore, dispositivo) -> float:
    rete.train()
    perdite = []
    for ingresso, bersaglio, maschera in caricatore:
        uscita_rete = rete(ingresso.to(dispositivo, non_blocking=True))
        bersaglio = bersaglio.to(dispositivo)
        maschera = maschera.to(dispositivo)
        perdita_per_uscita = torch.nn.functional.binary_cross_entropy_with_logits(uscita_rete, bersaglio,
                                                                                  reduction="none")
        perdita = (perdita_per_uscita * maschera).sum() / maschera.sum().clamp(min=1)
        ottimizzatore.zero_grad(set_to_none=True)
        perdita.backward()
        ottimizzatore.step()
        perdite.append(perdita.item())
    return float(np.mean(perdite))


@torch.no_grad()
def probabilita_lotto(rete, ingresso, dispositivo) -> np.ndarray:
    rete.eval()
    return torch.sigmoid(rete(ingresso.to(dispositivo))).cpu().numpy()


def accuratezza_bilanciata_val(probabilita: np.ndarray, bersagli: np.ndarray, maschere: np.ndarray) -> float:
    """Media sulle 5 uscite dell'accuratezza bilanciata a soglia 0,5 (le uscite senza positivi non contano)."""
    valori = []
    for uscita in range(len(USCITE)):
        valide = maschere[:, uscita] > 0
        positivi = valide & (bersagli[:, uscita] > 0.5)
        negativi = valide & (bersagli[:, uscita] < 0.5)
        if positivi.sum() == 0 or negativi.sum() == 0:
            continue
        previsti = probabilita[:, uscita] > 0.5
        valori.append((previsti[positivi].mean() + (~previsti[negativi]).mean()) / 2)
    return float(np.mean(valori))


def stato_in_mezza_precisione(base) -> dict:
    stato = {}
    for chiave, valore in base.state_dict().items():
        stato[chiave] = valore.half().cpu()
    return stato
# --- esecuzione ---


if __name__ == "__main__":
    ingresso = Path("/kaggle/input")
    uscita = Path("/kaggle/working")
    dati = Path("/tmp/dati")
    generatore = np.random.default_rng(0)

    clip_annotate = prepara_video_esperimenti(ingresso, dati)
    print(f"video pronti: {len(clip_annotate)} clip, {trascorse():.2f} h", flush=True)

    percorso_finestre = next(ingresso.glob("**/finestre.npz"))
    with np.load(percorso_finestre) as finestre:
        id_clip = finestre["id_clip"]
        inizi = finestre["inizio_finestra"]
        etichette = finestre["etichetta"]
        sorgenti = finestre["sorgente"]
    split = {}
    with np.load(percorso_finestre.parent / "split.npz") as file_split:
        for nome_split in file_split.files:
            split[nome_split] = file_split[nome_split]
    tipo_nuovo = np.full(len(etichette), "", dtype=object)   # tipo di Marco per le finestre di argo-anomalie

    if CON_ANOMALIE:
        # finestre di argo-anomalie in coda a quelle di finestre v3, con i loro video e il loro split
        percorso_anomalie = next(ingresso.glob("**/anomalie.json"))
        nuovi_id = []
        nuovi_inizi = []
        nuove_etichette = []
        nuove_sorgenti = []
        nuovi_tipi = []
        nuovi_split = {"train": [], "val": [], "test": []}
        for voce in json.loads(percorso_anomalie.read_text()):
            id_video = "anomalie/" + voce["chiave"]
            sorgente = "anomalie_" + voce["chiave"].split("__")[0]
            clip_annotate[id_video] = ClipAnnotata(id_clip=id_video, sorgente=sorgente, gruppo=id_video,
                                                   percorso_video=str(percorso_anomalie.parent / voce["video"]))
            for inizio_secondi, tipo in voce["finestre"]:
                nuovi_split[voce["split"]].append(len(etichette) + len(nuovi_id))
                nuovi_id.append(id_video)
                nuovi_inizi.append(int(round(inizio_secondi * 25)))
                nuove_etichette.append(ETICHETTA_DEL_TIPO[tipo])
                nuove_sorgenti.append(sorgente)
                nuovi_tipi.append(tipo)
        id_clip = np.concatenate([id_clip.astype(str), np.array(nuovi_id, dtype=str)])   # stringhe: npz senza pickle
        inizi = np.concatenate([inizi, np.array(nuovi_inizi, inizi.dtype)])
        etichette = np.concatenate([etichette, np.array(nuove_etichette, etichette.dtype)])
        sorgenti = np.concatenate([sorgenti.astype(str), np.array(nuove_sorgenti, dtype=str)])
        tipo_nuovo = np.concatenate([tipo_nuovo, np.array(nuovi_tipi, dtype=object)])
        for nome_split in ("train", "val", "test"):
            split[nome_split] = np.concatenate([split[nome_split], np.array(nuovi_split[nome_split], np.int64)])
        print(f"argo-anomalie: {len(nuovi_id)} finestre", flush=True)
    nuova = tipo_nuovo != ""

    # bersagli e maschere per ogni finestra
    tipi_ucf = tipi_arma_ucf()
    tipi_scvd = tipi_arma_scvd(FILE_TIPO_ARMA_SCVD)
    armata = etichette == INDICE_CLASSE["violenza_con_arma"]
    bersagli = np.zeros((len(etichette), len(USCITE)), np.float32)
    maschere = np.ones((len(etichette), len(USCITE)), np.float32)
    bersagli[:, 0] = np.isin(etichette, [INDICE_CLASSE["violenza"], INDICE_CLASSE["violenza_con_arma"]])
    bersagli[:, 1] = armata
    bersagli[:, 4] = etichette == INDICE_CLASSE["scippo"]
    if CON_ANOMALIE:
        bersagli[:, 5] = (etichette != INDICE_CLASSE["normale"]) | (nuova & (tipo_nuovo != "normale"))
        # UCF: le finestre "normali" dei video di rapine, furti, incendi... non sono normali per l'uscita anomalia
        for riga in np.where((sorgenti == "ucf_uca") & (etichette == INDICE_CLASSE["normale"]))[0]:
            id_video = str(id_clip[riga])[len("ucf/"):].split("@")[0]
            if categoria_ucf(id_video) != "Normal":
                maschere[riga, 5] = 0
    for riga in np.where(armata)[0]:
        tipi = tipi_della_clip(str(id_clip[riga]), tipi_ucf, tipi_scvd)
        if len(tipi) == 0:
            maschere[riga, 2] = 0   # tipo ignoto: non si sa se è fuoco o coltello
            maschere[riga, 3] = 0
        if "fuoco" in tipi:
            bersagli[riga, 2] = 1
        if "coltello" in tipi:
            bersagli[riga, 3] = 1

    clip_note = []
    for clip in id_clip:
        clip_note.append(clip in clip_annotate)
    utilizzabile = np.array(clip_note)
    # train: tutte (fino a un tetto) le finestre armate e di scippo, poi violenza e normali
    train = np.where(np.isin(np.arange(len(etichette)), split["train"]) & utilizzabile)[0]
    vecchie = train[~nuova[train]]
    gruppo_finestra = {"arma": vecchie[armata[vecchie]],
                       "scippo": vecchie[etichette[vecchie] == INDICE_CLASSE["scippo"]],
                       "violenza": vecchie[etichette[vecchie] == INDICE_CLASSE["violenza"]],
                       "normale": vecchie[etichette[vecchie] == INDICE_CLASSE["normale"]]}
    if CON_ANOMALIE:
        nuove = train[nuova[train]]
        gruppo_finestra["anomalie_nuove"] = nuove[tipo_nuovo[nuove] != "normale"]
        gruppo_finestra["normali_nuove"] = nuove[tipo_nuovo[nuove] == "normale"]
    scelti = []
    for nome, indici in gruppo_finestra.items():
        quante = min(FINESTRE_PER_GRUPPO[nome], len(indici))
        scelti.append(generatore.choice(indici, quante, replace=False))
        print(f"train {nome}: {quante} di {len(indici)}", flush=True)
    scelti_train = np.concatenate(scelti)
    val = np.where(np.isin(np.arange(len(etichette)), split["val"]) & utilizzabile)[0]
    # val per scegliere l'epoca: tutte le armate e gli scippi, il resto a caso
    anomala_nuova = nuova & (tipo_nuovo != "normale")
    val_rari = val[armata[val] | (etichette[val] == INDICE_CLASSE["scippo"]) | anomala_nuova[val]]
    val_altri = generatore.choice(val[~np.isin(val, val_rari)], max(0, FINESTRE_VAL - len(val_rari)), replace=False)
    scelti_val = np.sort(np.concatenate([val_rari, val_altri]))

    def frame_finestra(indice: int):
        """I 40 frame della finestra a 25 fps → i 16 campionati; None se il video ne dà meno di 16."""
        clip = clip_annotate[id_clip[indice]]
        secondi = 0.0
        if clip.inizio_secondi is not None:
            secondi = clip.inizio_secondi
        secondi += int(inizi[indice]) / 25
        comando = ["ffmpeg", "-v", "error", "-ss", f"{secondi:.3f}", "-i", clip.percorso_video, "-t", "1.6",
                   "-vf", f"fps=25,scale={LARGHEZZA}:{ALTEZZA}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        grezzi = np.frombuffer(subprocess.run(comando, capture_output=True).stdout, np.uint8)
        numero = grezzi.size // (ALTEZZA * LARGHEZZA * 3)
        if numero < 16:
            return None
        return sedici(grezzi[: numero * ALTEZZA * LARGHEZZA * 3].reshape(numero, ALTEZZA, LARGHEZZA, 3))

    def decodifica(indici: np.ndarray, nome: str):
        """Decodifica le finestre in un .npy su disco. Ritorna (percorso, posizioni riuscite)."""
        percorso = f"/tmp/{nome}.npy"
        archivio = np.lib.format.open_memmap(percorso, mode="w+", dtype=np.uint8,
                                             shape=(len(indici), 16, ALTEZZA, LARGHEZZA, 3))
        tenute = np.zeros(len(indici), bool)
        with ThreadPoolExecutor(os.cpu_count() * 2) as esecutore:
            for posizione, frame in enumerate(esecutore.map(frame_finestra, indici.tolist())):
                if frame is not None:
                    archivio[posizione] = frame
                    tenute[posizione] = True
                if posizione % 2000 == 0:
                    print(f"{nome}: decodificate {posizione}/{len(indici)}, {trascorse():.2f} h", flush=True)
        archivio.flush()
        del archivio
        return percorso, np.where(tenute)[0]

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rete = crea_rete()
    if torch.cuda.device_count() > 1:
        rete = torch.nn.DataParallel(rete)
    rete = rete.to(dispositivo)
    if isinstance(rete, torch.nn.DataParallel):
        base = rete.module
    else:
        base = rete
    testa = list(base.fc.parameters())
    corpo = []
    for nome, parametro in base.named_parameters():
        if not nome.startswith("fc."):
            corpo.append(parametro)

    # prova di un passo sulla GPU prima della decodifica (D53: un errore così si vede in 2 minuti)
    ottimizzatore = torch.optim.AdamW([{"params": corpo, "lr": 1e-4}, {"params": testa, "lr": 1e-3}],
                                      weight_decay=1e-4)
    finte = np.random.default_rng(0).integers(0, 256, (LOTTO, 16, ALTEZZA, LARGHEZZA, 3), dtype=np.uint8)
    lotto_finto = [(ritaglia(finte, True, np.random.default_rng(0)), torch.ones(LOTTO, len(USCITE)),
                    torch.ones(LOTTO, len(USCITE)))]
    un_epoca(rete, lotto_finto, ottimizzatore, dispositivo)
    base.load_state_dict(crea_rete().state_dict())   # si riparte dai pesi Kinetics
    ottimizzatore = torch.optim.AdamW([{"params": corpo, "lr": 1e-4}, {"params": testa, "lr": 1e-3}],
                                      weight_decay=1e-4)
    pianificatore = torch.optim.lr_scheduler.CosineAnnealingLR(ottimizzatore, EPOCHE)
    print(f"variante {VARIANTE} (lato {LATO}, grigio {GRIGIO}, {EPOCHE} epoche): prova di un passo ok, "
          f"{torch.cuda.device_count()} GPU, {trascorse():.2f} h", flush=True)

    percorso_val, posizioni_val = decodifica(scelti_val, "val")
    percorso_train, posizioni_train = decodifica(scelti_train, "train")
    righe_train = scelti_train[posizioni_train]
    righe_val = scelti_val[posizioni_val]
    caricatore = torch.utils.data.DataLoader(
        Finestre(percorso_train, posizioni_train, bersagli[righe_train], maschere[righe_train]), batch_size=LOTTO,
        shuffle=True, num_workers=os.cpu_count(), drop_last=True, pin_memory=True, collate_fn=unisci_lotto_train)
    caricatore_val = torch.utils.data.DataLoader(
        Finestre(percorso_val, posizioni_val, bersagli[righe_val], maschere[righe_val]), batch_size=LOTTO,
        num_workers=os.cpu_count(), collate_fn=unisci_lotto_val)
    print(f"train {len(posizioni_train)} finestre, val {len(posizioni_val)}, {trascorse():.2f} h", flush=True)

    storia = []
    migliore = -1.0
    for epoca in range(1, EPOCHE + 1):
        if trascorse() > ORE_MASSIME - 2.5:   # solo sicurezza: serve tempo per i punteggi finali
            print("tetto di tempo: fine dell'addestramento", flush=True)
            break
        inizio_epoca = time.time()
        perdita = un_epoca(rete, caricatore, ottimizzatore, dispositivo)
        pianificatore.step()
        uscite_val = []
        for ingresso_val, bersaglio_val, maschera_val in caricatore_val:
            uscite_val.append(probabilita_lotto(rete, ingresso_val, dispositivo))
        probabilita_val = np.concatenate(uscite_val)
        valore = accuratezza_bilanciata_val(probabilita_val, bersagli[righe_val], maschere[righe_val])
        storia.append({"epoca": epoca, "perdita": perdita, "bilanciata_val": valore,
                       "secondi": time.time() - inizio_epoca})
        print(f"epoca {epoca}: perdita {perdita:.4f}, accuratezza bilanciata val {valore:.4f} "
              f"({time.time() - inizio_epoca:.0f} s, {trascorse():.2f} h)", flush=True)
        if valore > migliore:
            migliore = valore
            torch.save(stato_in_mezza_precisione(base), uscita / "migliore.pt")
    torch.save(stato_in_mezza_precisione(base), uscita / "ultima.pt")
    (uscita / "storia.json").write_text(json.dumps(storia, indent=1))
    os.remove(percorso_train)
    os.remove(percorso_val)   # 8,4 GB a 224 px: b2 v2 è stata uccisa (RAM) nei punteggi finali

    # punteggi per il confronto: una finestra ogni due (inizio multiplo di 20 frame) di val e test, un flusso
    # ffmpeg per clip. Il confronto usa solo le finestre che hanno tutti i modelli.
    byte_per_frame = ALTEZZA * LARGHEZZA * 3

    def frame_del_blocco(clip: str, primo_frame: int, frame_necessari: int) -> np.ndarray:
        """I frame [primo_frame, primo_frame + frame_necessari) della clip, a 25 fps."""
        annotata = clip_annotate[clip]
        secondi = primo_frame / 25
        if annotata.inizio_secondi is not None:
            secondi += annotata.inizio_secondi
        comando = ["ffmpeg", "-v", "error", "-ss", f"{secondi:.3f}", "-i", annotata.percorso_video,
                   "-t", f"{frame_necessari / 25 + 0.2:.3f}", "-vf", f"fps=25,scale={LARGHEZZA}:{ALTEZZA}",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        grezzi = np.frombuffer(subprocess.run(comando, capture_output=True).stdout, np.uint8)
        frame_completi = grezzi.size // byte_per_frame
        return grezzi[: frame_completi * byte_per_frame].reshape(-1, ALTEZZA, LARGHEZZA, 3)

    # goal 80% (07/11): solo la migliore; con Kaggle lento anche i punteggi dell'ultima sforavano le 12 h
    for nome_pesi in ("migliore",):
        stato = {}
        for chiave, valore in torch.load(uscita / f"{nome_pesi}.pt", map_location="cpu").items():
            stato[chiave] = valore.float()
        base.load_state_dict(stato)
        cartella_punteggi = uscita / f"punteggi_{nome_pesi}"
        cartella_punteggi.mkdir(exist_ok=True)
        for nome_split in ("val", "test"):
            indici = np.where(np.isin(np.arange(len(etichette)), split[nome_split]) & utilizzabile
                              & (inizi % 20 == 0))[0]
            # blocchi = (clip, primo frame, righe delle finestre che cominciano nel blocco)
            blocchi = []
            for clip in np.unique(id_clip[indici]):
                righe = indici[id_clip[indici] == clip]
                righe = righe[np.argsort(inizi[righe])]
                numero_blocco = inizi[righe] // (FRAME_PER_BLOCCO - 40)
                for valore in np.unique(numero_blocco):
                    blocchi.append((clip, int(valore) * (FRAME_PER_BLOCCO - 40), righe[numero_blocco == valore]))
            tutte_le_righe = []
            tutte_le_p = []
            # decodifica di pochi blocchi alla volta in parallelo (la memoria resta piccola), rete sulla GPU
            with ThreadPoolExecutor(os.cpu_count()) as esecutore:
                for primo_gruppo in range(0, len(blocchi), os.cpu_count()):
                    gruppo = blocchi[primo_gruppo: primo_gruppo + os.cpu_count()]
                    lavori = []
                    for clip, primo_frame, righe in gruppo:
                        lavori.append(esecutore.submit(frame_del_blocco, clip, primo_frame, FRAME_PER_BLOCCO))
                    for numero, (clip, primo_frame, righe) in enumerate(gruppo):
                        frame = lavori[numero].result()
                        finestre_blocco = []
                        for riga in righe:
                            inizio_nel_blocco = int(inizi[riga]) - primo_frame
                            pezzo = frame[inizio_nel_blocco: inizio_nel_blocco + 40]
                            if len(pezzo) >= 16:
                                tutte_le_righe.append(riga)
                                finestre_blocco.append(sedici(pezzo))
                        for primo in range(0, len(finestre_blocco), LOTTO):
                            lotto = ritaglia(np.stack(finestre_blocco[primo: primo + LOTTO]), False, None)
                            tutte_le_p.append(probabilita_lotto(rete, lotto, dispositivo))
                    print(f"punteggi {nome_pesi} {nome_split}: blocchi {primo_gruppo + len(gruppo)}/{len(blocchi)}, "
                          f"{trascorse():.2f} h", flush=True)
            righe = np.array(tutte_le_righe, np.int64)
            probabilita = np.concatenate(tutte_le_p)
            finestre_scelte = {"id_clip": id_clip[righe], "inizio_finestra": inizi[righe],
                               "etichetta": etichette[righe], "sorgente": sorgenti[righe]}
            punteggi = {}
            for numero_uscita, nome_uscita in enumerate(USCITE):
                if nome_uscita != "violenza":
                    punteggi[nome_uscita] = probabilita[:, numero_uscita]
            salva_punteggi(cartella_punteggi / f"punteggi_{nome_split}.npz", finestre_scelte, punteggi)
            np.save(cartella_punteggi / f"violenza_{nome_split}.npy", probabilita[:, 0])
    print("fatto", flush=True)
