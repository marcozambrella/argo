# ===== avvio comune: uguale in tutti i kernel di training/ =====
# Kaggle carica un solo file per kernel, quindi questo blocco è copiato in testa a ognuno.
import os
import shutil
import subprocess
import sys
URL_REPO = "https://github.com/marcozambrella/argo"
# D68: gli esperimenti usano il branch esperimenti-armi-scippi, non main
subprocess.run(["git", "clone", "--depth", "1", "--branch", "esperimenti-armi-scippi", URL_REPO, "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4", "lap", "open_clip_torch"], check=True)
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

"""Goal 80% (07/11), G3: feature CLIP congelate per le finestre, al posto di una rete video addestrata.
Ricerca: le feature CLIP congelate sono lo stato dell'arte sulle anomalie di UCF-Crime (VadCLIP) e CLIP "conosce"
pistole e coltelli. Per ogni finestra 4 frame (4, 14, 24, 34 dei 40), ognuno visto 3 volte: intero (schiacciato
in un quadrato) e due quadrati sovrapposti a sinistra e a destra (il doppio della risoluzione per le armi piccole;
b2 invece taglia via il 34% della larghezza). Due modelli: ViT-L/14 a 336 px e ViT-B/16 a 224 px (OpenAI).
Le teste (arma, fuoco, coltello, scippo, violenza, anomalia) si addestrano poi in locale sulle feature salvate.
Finestre: train come b1a ma con più finestre per gruppo; val e test tutte quelle che cominciano a multipli di 20
frame, come i punteggi delle reti video (così il confronto è sulle stesse finestre)."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import open_clip
import torch
from sicurezza.dati import INDICE_CLASSE, prepara_video_esperimenti
from sicurezza.dati import ClipAnnotata

LARGHEZZA = 640
ALTEZZA = 480          # 4:3 come b2 (le CCTV sono quasi tutte 4:3 o 16:9)
FRAME_SCELTI = (4, 14, 24, 34)
FINESTRE_PER_GRUPPO = {"arma": 8000, "scippo": 8000, "violenza": 6000, "normale": 8000,
                       "anomalie_nuove": 6000, "normali_nuove": 6000}
FRAME_PER_BLOCCO = 600
MODELLI = (("ViT-L-14-336", "openai", 336), ("ViT-B-16", "openai", 224))
ETICHETTA_DEL_TIPO = {"normale": INDICE_CLASSE["normale"], "violenza": INDICE_CLASSE["violenza"],
                      "arma": INDICE_CLASSE["violenza_con_arma"], "scippo": INDICE_CLASSE["scippo"],
                      "furto": -1, "altra": -1}
MEDIA_CLIP = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
DEVIAZIONE_CLIP = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
avvio = time.time()


def trascorse() -> float:
    return (time.time() - avvio) / 3600


def viste(frame: np.ndarray, lato: int) -> torch.Tensor:
    """[M, 480, 640, 3] uint8 → [M * 3, 3, lato, lato]: intero, quadrato a sinistra, quadrato a destra."""
    immagini = torch.from_numpy(np.ascontiguousarray(frame)).permute(0, 3, 1, 2).float() / 255
    intero = torch.nn.functional.interpolate(immagini, size=(lato, lato), mode="bilinear", antialias=True)
    sinistra = torch.nn.functional.interpolate(immagini[:, :, :, :ALTEZZA], size=(lato, lato), mode="bilinear",
                                               antialias=True)
    destra = torch.nn.functional.interpolate(immagini[:, :, :, LARGHEZZA - ALTEZZA:], size=(lato, lato),
                                             mode="bilinear", antialias=True)
    tutte = torch.stack([intero, sinistra, destra], dim=1).reshape(-1, 3, lato, lato)
    return (tutte - MEDIA_CLIP) / DEVIAZIONE_CLIP


@torch.no_grad()
def feature(modelli: list, frame: np.ndarray, dispositivo) -> list:
    """Per ogni modello: [M, 3, D] float16 normalizzate (M frame, 3 viste)."""
    risultati = []
    for rete, lato in modelli:
        pezzi = []
        lotto_viste = viste(frame, lato)
        for primo in range(0, len(lotto_viste), 96):
            ingresso = lotto_viste[primo: primo + 96].to(dispositivo).half()
            uscita_rete = rete(ingresso).float()
            uscita_rete = uscita_rete / uscita_rete.norm(dim=-1, keepdim=True)
            pezzi.append(uscita_rete.cpu().half().numpy())
        tutte = np.concatenate(pezzi)
        risultati.append(tutte.reshape(len(frame), 3, -1))
    return risultati


class SoloImmagini(torch.nn.Module):
    """encode_image come forward, per DataParallel sulle 2 GPU."""
    def __init__(self, modello):
        super().__init__()
        self.modello = modello

    def forward(self, immagini):
        return self.modello.encode_image(immagini)
# --- esecuzione ---


if __name__ == "__main__":
    ingresso = Path("/kaggle/input")
    uscita = Path("/kaggle/working")
    dati = Path("/tmp/dati")
    generatore = np.random.default_rng(0)
    dispositivo = torch.device("cuda")

    modelli = []
    for nome, pesi, lato in MODELLI:
        modello = open_clip.create_model(nome, pretrained=pesi).eval().half().to(dispositivo)
        rete = SoloImmagini(modello)
        if torch.cuda.device_count() > 1:
            rete = torch.nn.DataParallel(rete)
        modelli.append((rete, lato))
    # prova sulla GPU prima della decodifica: un errore così si vede in pochi minuti
    prova = feature(modelli, np.zeros((2, ALTEZZA, LARGHEZZA, 3), np.uint8), dispositivo)
    print("prova ok:", [p.shape for p in prova], f"{trascorse():.2f} h", flush=True)

    clip_annotate = prepara_video_esperimenti(ingresso, dati)
    percorso_finestre = next(ingresso.glob("**/finestre.npz"))
    with np.load(percorso_finestre) as finestre:
        id_clip = finestre["id_clip"].astype(str)
        inizi = finestre["inizio_finestra"]
        etichette = finestre["etichetta"]
        sorgenti = finestre["sorgente"].astype(str)
    split = {}
    with np.load(percorso_finestre.parent / "split.npz") as file_split:
        for nome_split in file_split.files:
            split[nome_split] = file_split[nome_split]

    # finestre di argo-anomalie in coda, come b1a
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
    tipo_nuovo = np.concatenate([np.full(len(etichette), "", dtype=object), np.array(nuovi_tipi, dtype=object)])
    id_clip = np.concatenate([id_clip, np.array(nuovi_id, dtype=str)])
    inizi = np.concatenate([inizi, np.array(nuovi_inizi, inizi.dtype)])
    etichette = np.concatenate([etichette, np.array(nuove_etichette, etichette.dtype)])
    sorgenti = np.concatenate([sorgenti, np.array(nuove_sorgenti, dtype=str)])
    for nome_split in ("train", "val", "test"):
        split[nome_split] = np.concatenate([split[nome_split], np.array(nuovi_split[nome_split], np.int64)])
    nuova = tipo_nuovo != ""
    print(f"argo-anomalie: {len(nuovi_id)} finestre, {trascorse():.2f} h", flush=True)

    clip_note = []
    for clip in id_clip:
        clip_note.append(clip in clip_annotate)
    utilizzabile = np.array(clip_note)
    armata = etichette == INDICE_CLASSE["violenza_con_arma"]
    train = np.where(np.isin(np.arange(len(etichette)), split["train"]) & utilizzabile)[0]
    vecchie = train[~nuova[train]]
    nuove = train[nuova[train]]
    gruppo_finestra = {"arma": vecchie[armata[vecchie]],
                       "scippo": vecchie[etichette[vecchie] == INDICE_CLASSE["scippo"]],
                       "violenza": vecchie[etichette[vecchie] == INDICE_CLASSE["violenza"]],
                       "normale": vecchie[etichette[vecchie] == INDICE_CLASSE["normale"]],
                       "anomalie_nuove": nuove[tipo_nuovo[nuove] != "normale"],
                       "normali_nuove": nuove[tipo_nuovo[nuove] == "normale"]}
    scelti = []
    for nome, indici in gruppo_finestra.items():
        quante = min(FINESTRE_PER_GRUPPO[nome], len(indici))
        scelti.append(generatore.choice(indici, quante, replace=False))
        print(f"train {nome}: {quante} di {len(indici)}", flush=True)
    righe_per_split = {"train": np.sort(np.concatenate(scelti))}
    for nome_split in ("val", "test"):
        righe_per_split[nome_split] = np.where(np.isin(np.arange(len(etichette)), split[nome_split]) & utilizzabile
                                               & (inizi % 20 == 0))[0]
    byte_per_frame = ALTEZZA * LARGHEZZA * 3

    def frame_del_blocco(clip: str, primo_frame: int, frame_necessari: int) -> np.ndarray:
        """I frame [primo_frame, primo_frame + frame_necessari) della clip, a 25 fps, 640×480."""
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

    for nome_split in ("val", "test", "train"):
        indici = righe_per_split[nome_split]
        # blocchi = (clip, primo frame, righe delle finestre che cominciano nel blocco); il train ha finestre sparse,
        # quindi un blocco per finestra
        blocchi = []
        for clip in np.unique(id_clip[indici]):
            righe = indici[id_clip[indici] == clip]
            righe = righe[np.argsort(inizi[righe])]
            if nome_split == "train":
                for riga in righe:
                    blocchi.append((clip, int(inizi[riga]), np.array([riga]), 40))
                continue
            numero_blocco = inizi[righe] // (FRAME_PER_BLOCCO - 40)
            for valore in np.unique(numero_blocco):
                blocchi.append((clip, int(valore) * (FRAME_PER_BLOCCO - 40), righe[numero_blocco == valore],
                                FRAME_PER_BLOCCO))
        tenute = []
        feature_per_modello = []
        for _ in MODELLI:
            feature_per_modello.append([])
        numero_thread = os.cpu_count() * 2
        with ThreadPoolExecutor(numero_thread) as esecutore:
            for primo_gruppo in range(0, len(blocchi), numero_thread):
                gruppo = blocchi[primo_gruppo: primo_gruppo + numero_thread]
                lavori = []
                for clip, primo_frame, righe, quanti in gruppo:
                    lavori.append(esecutore.submit(frame_del_blocco, clip, primo_frame, quanti))
                frame_gruppo = []
                for numero, (clip, primo_frame, righe, quanti) in enumerate(gruppo):
                    frame = lavori[numero].result()
                    for riga in righe:
                        inizio_nel_blocco = int(inizi[riga]) - primo_frame
                        posizioni = []
                        for scelto in FRAME_SCELTI:
                            posizioni.append(min(inizio_nel_blocco + scelto, len(frame) - 1))
                        if len(frame) == 0 or inizio_nel_blocco + FRAME_SCELTI[1] >= len(frame):
                            continue
                        tenute.append(riga)
                        frame_gruppo.append(frame[posizioni])
                if len(frame_gruppo) == 0:
                    continue
                tutti = np.concatenate(frame_gruppo)
                for numero_modello, valori in enumerate(feature(modelli, tutti, dispositivo)):
                    feature_per_modello[numero_modello].append(valori.reshape(len(frame_gruppo), len(FRAME_SCELTI),
                                                                              3, -1))
                if (primo_gruppo // numero_thread) % 50 == 0:
                    print(f"{nome_split}: blocchi {primo_gruppo + len(gruppo)}/{len(blocchi)}, "
                          f"finestre {len(tenute)}, {trascorse():.2f} h", flush=True)
        righe = np.array(tenute, np.int64)
        np.savez(uscita / f"finestre_{nome_split}.npz", riga=righe, id_clip=id_clip[righe],
                 inizio_finestra=inizi[righe], etichetta=etichette[righe], sorgente=sorgenti[righe],
                 tipo_nuovo=tipo_nuovo[righe].astype(str))
        for numero_modello, (nome, pesi, lato) in enumerate(MODELLI):
            np.save(uscita / f"feature_{nome}_{nome_split}.npy", np.concatenate(feature_per_modello[numero_modello]))
        print(f"{nome_split} fatto: {len(righe)} finestre, {trascorse():.2f} h", flush=True)
    print("fatto", flush=True)
