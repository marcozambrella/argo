# ===== avvio comune (da kaggle/avvio_comune.py) =====
# Blocco iniziale comune ai kernel (copiato in testa a ogni script: Kaggle carica un solo file per kernel).
import os, shutil, subprocess, sys
URL_REPO = "https://github.com/marcozambrella/roma-safe-cctv"
subprocess.run(["git", "clone", "--depth", "1", URL_REPO, "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4", "lap"], check=True)
sys.path.insert(0, "/tmp/repo/src")
print("commit:", subprocess.run(["git", "-C", "/tmp/repo", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip(), flush=True)
if shutil.which("ffprobe") is None:  # il lettore video usa ffmpeg/ffprobe: build statica se l'immagine non li ha
    subprocess.run("curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | "
                   "tar -xJ -C /tmp && cp /tmp/ffmpeg-*-static/ff* /usr/local/bin/", shell=True, check=True)
# =====

"""D53 (1), GPU: fine-tuning di TUTTA la R(2+1)D-18 (Kinetics-400) sulla violenza sì/no, invece della rete congelata
+ regressione logistica di D48. Stesse finestre e stesso split dell'addestramento 2 (RWF-2000, CCTV-Fights, SCVD,
UCF), stessa preparazione della demo (40 frame a 25 fps → 16 frame, 171×128, ritaglio 112×112; in train ritaglio a
caso e specchio). L'epoca si sceglie su finestre sparse del validation; poi si rimisurano le STESSE finestre di clip
intere di `prova-rgb-clip` (test mai visto dall'add. 1 + validation, indici in kaggle/finetune_rgb/finestre_clip.json),
decodificate come nella demo, così in locale la fusione con gli scheletri si confronta alla pari con D48.
Uscite: rete_violenza.pt (rete senza testa, fp16), testa_violenza.npz (stesso formato di pesi/testa_rgb.npz),
prob_clip_test.npz / prob_clip_val.npz (indici, probabilità), storia.json."""
import json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch, torchvision
from sicurezza.dati import INDICE_CLASSE, prepara

ORE_MASSIME = 8.0                  # tutta la sessione; l'addestramento si ferma ~1,5 h prima per la misura finale
FINESTRE_TRAIN_PER_CLASSE = 7000   # 14.000 finestre × ~1 MB in /tmp (si riduce se il disco non basta)
FINESTRE_VAL = 2000
EPOCHE, PAZIENZA, LOTTO = 12, 3, 32
# 21/10 07:52, versione 1: con fp16 (autocast) le conv3d sulle T4 danno "CUDA error: misaligned address" → fp32
MEZZA_PRECISIONE = False
SORGENTI = ("rwf2000", "cctv_fights", "scvd", "ucf_uca")
ALTEZZA, LARGHEZZA = 128, 171
MEDIA = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
DEVIAZIONE = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)
avvio = time.time()
trascorse = lambda: (time.time() - avvio) / 3600


# --- funzioni ---
def sedici(frame: np.ndarray) -> np.ndarray:
    return frame[np.linspace(0, len(frame) - 1, 16).round().astype(int)]


def ritaglia(finestre: np.ndarray, allena: bool, generatore: np.random.Generator) -> torch.Tensor:
    """[B, 16, 128, 171, 3] uint8 → [B, 3, 16, 112, 112] normalizzato. In train ritaglio a caso + specchio."""
    pezzi = []
    for finestra in finestre:
        if allena:
            y, x = generatore.integers(0, ALTEZZA - 112 + 1), generatore.integers(0, LARGHEZZA - 112 + 1)
            pezzo = finestra[:, y:y + 112, x:x + 112]
            pezzi.append(pezzo[:, :, ::-1] if generatore.random() < 0.5 else pezzo)
        else:
            pezzi.append(finestra[:, 8:120, 29:141])
    lotto = torch.from_numpy(np.ascontiguousarray(np.stack(pezzi))).permute(0, 4, 1, 2, 3).float() / 255
    return ((lotto - MEDIA) / DEVIAZIONE).contiguous()   # senza, resta la disposizione permutata (v1: errore CUDA)


class Finestre(torch.utils.data.Dataset):
    """Finestre già decodificate in un .npy su disco (aperto in sola lettura in ogni worker)."""
    def __init__(self, percorso: str, posizioni: np.ndarray, etichette: np.ndarray):
        self.percorso, self.posizioni, self.etichette, self.archivio = percorso, posizioni, etichette, None

    def __len__(self):
        return len(self.posizioni)

    def __getitem__(self, numero):
        if self.archivio is None:
            self.archivio = np.load(self.percorso, mmap_mode="r")
        return np.asarray(self.archivio[self.posizioni[numero]]), self.etichette[numero]


def unisci_lotto(elementi, allena: bool, generatore):
    return ritaglia(np.stack([e[0] for e in elementi]), allena, generatore), torch.tensor([float(e[1]) for e in elementi])


def crea_rete() -> torch.nn.Module:
    rete = torchvision.models.video.r2plus1d_18(weights=torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1)
    rete.fc = torch.nn.Linear(512, 1)
    return rete


def un_epoca(rete, caricatore, ottimizzatore, scalatore, dispositivo) -> float:
    rete.train()
    perdite = []
    for ingresso, etichetta in caricatore:
        with torch.autocast(dispositivo.type, enabled=MEZZA_PRECISIONE and dispositivo.type == "cuda"):
            uscita_rete = rete(ingresso.to(dispositivo, non_blocking=True))[:, 0]
            perdita = torch.nn.functional.binary_cross_entropy_with_logits(uscita_rete.float(), etichetta.to(dispositivo))
        ottimizzatore.zero_grad(set_to_none=True)
        scalatore.scale(perdita).backward()
        scalatore.step(ottimizzatore)
        scalatore.update()
        perdite.append(perdita.item())
    return float(np.mean(perdite))


@torch.no_grad()
def probabilita(rete, lotti, dispositivo) -> np.ndarray:
    rete.eval()
    uscite = []
    for ingresso in lotti:
        with torch.autocast(dispositivo.type, enabled=MEZZA_PRECISIONE and dispositivo.type == "cuda"):
            uscite.append(torch.sigmoid(rete(ingresso.to(dispositivo))[:, 0].float()).cpu().numpy())
    return np.concatenate(uscite) if uscite else np.zeros(0, np.float32)
# --- esecuzione ---


if __name__ == "__main__":
    ingresso, uscita = Path("/kaggle/input"), Path("/kaggle/working")
    generatore = np.random.default_rng(0)
    clip_annotate = {clip.id_clip: clip for clip in prepara(ingresso, Path("/tmp/elenchi"), SORGENTI)}
    percorso_finestre = next(ingresso.glob("**/finestre.npz"))
    with np.load(percorso_finestre) as finestre:
        id_clip, inizi, etichette, sorgenti = (finestre[k] for k in ("id_clip", "inizio_finestra", "etichetta", "sorgente"))
    split = np.load(percorso_finestre.parent / "split.npz")
    violenta = np.isin(etichette, [INDICE_CLASSE["violenza"], INDICE_CLASSE["violenza_con_arma"]])
    utilizzabile = np.where(np.array([clip in clip_annotate for clip in id_clip]))[0]
    libero_gb = shutil.disk_usage("/tmp").free / 1e9
    per_classe = int(min(FINESTRE_TRAIN_PER_CLASSE, libero_gb * 0.7 / 1.05e-3 / 2 - FINESTRE_VAL))
    print(f"{len(clip_annotate)} clip, {len(utilizzabile)} finestre utilizzabili, /tmp libero {libero_gb:.0f} GB "
          f"→ {per_classe} finestre di train per classe", flush=True)
    train = np.intersect1d(split["train"], utilizzabile)
    scelti_train = np.concatenate([generatore.choice(train[violenta[train] == valore],
                                                     min(per_classe, int((violenta[train] == valore).sum())),
                                                     replace=False) for valore in (True, False)])
    val = np.intersect1d(split["val"], utilizzabile)
    scelti_val = np.sort(generatore.choice(val, min(FINESTRE_VAL, len(val)), replace=False))

    def frame_finestra(indice: int) -> np.ndarray | None:
        """I 40 frame della finestra a 25 fps, 171×128 RGB → i 16 campionati (come prova-rgb)."""
        clip = clip_annotate[id_clip[indice]]
        secondi = (clip.inizio_secondi or 0) + int(inizi[indice]) / 25
        comando = ["ffmpeg", "-v", "error", "-ss", f"{secondi:.3f}", "-i", clip.percorso_video, "-t", "1.6",
                   "-vf", f"fps=25,scale={LARGHEZZA}:{ALTEZZA}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        grezzi = np.frombuffer(subprocess.run(comando, capture_output=True).stdout, np.uint8)
        numero = grezzi.size // (ALTEZZA * LARGHEZZA * 3)
        if numero < 16:
            return None
        return sedici(grezzi[: numero * ALTEZZA * LARGHEZZA * 3].reshape(numero, ALTEZZA, LARGHEZZA, 3))

    def decodifica(indici: np.ndarray, nome: str) -> tuple[str, np.ndarray]:
        percorso = f"/tmp/{nome}.npy"
        archivio = np.lib.format.open_memmap(percorso, mode="w+", dtype=np.uint8,
                                             shape=(len(indici), 16, ALTEZZA, LARGHEZZA, 3))
        tenute = np.zeros(len(indici), bool)
        with ThreadPoolExecutor(os.cpu_count() * 2) as esecutore:
            for posizione, frame in enumerate(esecutore.map(frame_finestra, indici.tolist())):
                if frame is not None:
                    archivio[posizione], tenute[posizione] = frame, True
                if posizione % 1000 == 0:
                    print(f"{nome}: decodificate {posizione}/{len(indici)}, {trascorse():.2f} h", flush=True)
        archivio.flush()
        del archivio
        return percorso, np.where(tenute)[0]

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rete = crea_rete()
    if torch.cuda.device_count() > 1:
        rete = torch.nn.DataParallel(rete)
    rete = rete.to(dispositivo)
    base = rete.module if isinstance(rete, torch.nn.DataParallel) else rete
    testa = list(base.fc.parameters())
    corpo = [p for nome, p in base.named_parameters() if not nome.startswith("fc.")]
    ottimizzatore = torch.optim.AdamW([{"params": corpo, "lr": 1e-4}, {"params": testa, "lr": 1e-3}],
                                      weight_decay=1e-4)
    pianificatore = torch.optim.lr_scheduler.CosineAnnealingLR(ottimizzatore, EPOCHE)
    scalatore = torch.amp.GradScaler("cuda", enabled=MEZZA_PRECISIONE and dispositivo.type == "cuda")
    finte = np.random.default_rng(0).integers(0, 256, (LOTTO, 16, ALTEZZA, LARGHEZZA, 3), dtype=np.uint8)
    un_epoca(rete, [(ritaglia(finte, True, np.random.default_rng(0)), torch.ones(LOTTO))], ottimizzatore, scalatore,
             dispositivo)   # stesso percorso dei dati veri
    ottimizzatore.zero_grad(set_to_none=True)
    base.load_state_dict(crea_rete().state_dict())   # si riparte dai pesi Kinetics: la prova serviva solo a fallire presto
    ottimizzatore = torch.optim.AdamW([{"params": corpo, "lr": 1e-4}, {"params": testa, "lr": 1e-3}],
                                      weight_decay=1e-4)
    pianificatore = torch.optim.lr_scheduler.CosineAnnealingLR(ottimizzatore, EPOCHE)
    print(f"prova di un passo sulla GPU: ok ({torch.cuda.device_count()} GPU), {trascorse():.2f} h", flush=True)

    percorso_val, posizioni_val = decodifica(scelti_val, "val")
    percorso_train, posizioni_train = decodifica(scelti_train, "train")
    print(f"train {len(posizioni_train)} finestre, val {len(posizioni_val)}, {trascorse():.2f} h", flush=True)

    caricatore = torch.utils.data.DataLoader(
        Finestre(percorso_train, posizioni_train, violenta[scelti_train[posizioni_train]]), batch_size=LOTTO,
        shuffle=True, num_workers=os.cpu_count(), drop_last=True, pin_memory=True,
        collate_fn=lambda elementi: unisci_lotto(elementi, True, np.random.default_rng()))
    caricatore_val = torch.utils.data.DataLoader(
        Finestre(percorso_val, posizioni_val, violenta[scelti_val[posizioni_val]]), batch_size=LOTTO,
        num_workers=os.cpu_count(), collate_fn=lambda elementi: unisci_lotto(elementi, False, None))
    vere_val = violenta[scelti_val[posizioni_val]]

    storia, migliore, senza_miglioramenti = [], -1.0, 0
    for epoca in range(1, EPOCHE + 1):
        if trascorse() > ORE_MASSIME - 1.5:
            print("tetto di tempo: fine dell'addestramento", flush=True)
            break
        inizio_epoca = time.time()
        perdita = un_epoca(rete, caricatore, ottimizzatore, scalatore, dispositivo)
        pianificatore.step()
        p_val = probabilita(rete, (lotto for lotto, _ in caricatore_val), dispositivo)
        accuratezza = float(((p_val > 0.5) == vere_val).mean())
        storia.append({"epoca": epoca, "perdita": perdita, "accuratezza_val": accuratezza,
                       "recall_val": float((p_val > 0.5)[vere_val].mean()), "secondi": time.time() - inizio_epoca})
        print(f"epoca {epoca}: perdita {perdita:.4f}, accuratezza val {accuratezza:.4f} "
              f"({time.time() - inizio_epoca:.0f} s, {trascorse():.2f} h)", flush=True)
        if accuratezza > migliore:
            migliore, senza_miglioramenti = accuratezza, 0
            torch.save({k: v.half().cpu() for k, v in base.state_dict().items()}, uscita / "migliore_completo.pt")
        else:
            senza_miglioramenti += 1
            if senza_miglioramenti >= PAZIENZA:
                print("pazienza finita", flush=True)
                break
    (uscita / "storia.json").write_text(json.dumps(storia, indent=1))

    # v2 (21/10 10:05): senza map_location i tensori restavano su GPU e .numpy() falliva dopo l'addestramento
    stato = {k: v.float() for k, v in torch.load(uscita / "migliore_completo.pt", map_location="cpu").items()}
    base.load_state_dict(stato)
    torch.save({k: v.half() for k, v in stato.items() if not k.startswith("fc.")}, uscita / "rete_violenza.pt")
    np.savez(uscita / "testa_violenza.npz", media=np.zeros(512), scala=np.ones(512),
             pesi=stato["fc.weight"][0].numpy().astype(np.float32), bias=np.float32(stato["fc.bias"][0]))
    os.remove(percorso_train)   # libera il disco per la misura finale

    # Misura finale sulle finestre di clip intere di prova-rgb-clip, decodificate come nella demo (un flusso per clip)
    indici_clip = json.loads(Path("/tmp/repo/kaggle/finetune_rgb/finestre_clip.json").read_text())
    for nome in ("test", "val"):
        indici = np.array(indici_clip[nome])
        tutte_le_righe, tutte_le_p = [], []
        for numero, clip in enumerate(dict.fromkeys(id_clip[indici])):
            righe = indici[id_clip[indici] == clip]
            righe = righe[np.argsort(inizi[righe])]
            annotata = clip_annotate[clip]
            frame_necessari = int(inizi[righe].max()) + 40
            comando = ["ffmpeg", "-v", "error", "-ss", f"{annotata.inizio_secondi or 0:.3f}", "-i", annotata.percorso_video,
                       "-t", f"{frame_necessari / 25 + 0.2:.3f}", "-vf", f"fps=25,scale={LARGHEZZA}:{ALTEZZA}",
                       "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
            grezzi = np.frombuffer(subprocess.run(comando, capture_output=True).stdout, np.uint8)
            frame = grezzi[: grezzi.size // (ALTEZZA * LARGHEZZA * 3) * ALTEZZA * LARGHEZZA * 3].reshape(-1, ALTEZZA, LARGHEZZA, 3)
            tenute = [riga for riga in righe if len(frame[int(inizi[riga]): int(inizi[riga]) + 40]) >= 16]
            finestre_clip = [sedici(frame[int(inizi[riga]): int(inizi[riga]) + 40]) for riga in tenute]
            lotti = (ritaglia(np.stack(finestre_clip[i: i + LOTTO]), False, None) for i in range(0, len(finestre_clip), LOTTO))
            tutte_le_righe += tenute
            tutte_le_p.append(probabilita(rete, lotti, dispositivo))
            if numero % 20 == 0:
                print(f"misura {nome}: clip {numero}, finestre {len(tutte_le_righe)}, {trascorse():.2f} h", flush=True)
        np.savez(uscita / f"prob_clip_{nome}.npz", indici=np.array(tutte_le_righe, np.int64),
                 probabilita=np.concatenate(tutte_le_p) if tutte_le_p else np.zeros(0, np.float32))
        print(f"misura {nome}: {len(tutte_le_righe)} finestre, {trascorse():.2f} h", flush=True)
    print("fatto", flush=True)
