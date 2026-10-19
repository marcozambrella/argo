"""Sicurezza NN — visione. File unico (richiesta di Marco: massimo 5 file Python)."""

from collections import Counter
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import yaml

from .dati import GOMITI, POLSI, VideoDaEstrarre, FPS_ANALISI, leggi_elenco, leggi_frame, leggi_info_video, nome_file_estratto


# ==============================================================================================================
# estrazione
# Estrazione delle timeline: video a 25 fps → pose tracciate (YOLO26s-pose + BoT-SORT) + armi → un .npz per video.
#
# L'estrazione è la parte costosa e non dipende dalle etichette: si fa una volta sola e i file si riusano per ogni
# esperimento. Riprende da dove si era fermata (salta i .npz già presenti) e si divide in parti, una per GPU.
#
# Uso (una parte per GPU):
#     python -m sicurezza.visione estrai --elenco elenco.jsonl --uscita estratti/ --parte 0 --parti 2 --dispositivo 0
# ==============================================================================================================

MODELLO_POSE_PREDEFINITO = "yolo26s-pose.pt"   # D3
TRACKER_PREDEFINITO = "tracktrack.yaml"        # D21: il vero default di .track in Ultralytics 8.4 (D3)


PERSONE_VUOTE = {"id_traccia": np.empty(0, np.int32), "box": np.empty((0, 4), np.float32),
                 "confidenza": np.empty(0, np.float32), "keypoint": np.empty((0, 17, 3), np.float32)}
ARMI_VUOTE = {"box": np.empty((0, 4), np.float32), "classe": np.empty(0, np.int32),
              "confidenza": np.empty(0, np.float32)}


class Estrattore:
    """Tiene caricati i modelli e produce i dati grezzi di un video.

    D20: pose e armi vengono calcolate a lotti di frame (la GPU lavora meglio), il tracker (TrackTrack, D21) invece
    riceve le detection frame per frame esattamente come fa `.track` di Ultralytics (stesso risultato, test
    in tests/test_estrazione.py). Per i modelli pose TrackTrack non usa le predizioni grezze pre-NMS (Ultralytics
    attiva quell'aggancio solo per detect/obb), quindi riceve esattamente gli stessi dati. La compensazione del movimento della telecamera (GMC) del tracker costa
    ~11 ms a frame: si accende solo per i video con telecamera_mobile=True.
    """

    def __init__(self, percorso_modello_pose: str = MODELLO_POSE_PREDEFINITO,
                 percorso_modello_armi: str | None = None, dispositivo: str | int | None = None,
                 tracker: str = TRACKER_PREDEFINITO, confidenza_minima_persone: float = 0.1,
                 confidenza_minima_armi: float = 0.25, mezza_precisione: bool = False, dimensione_lotto: int = 8,
                 percorso_modello_mani: str | None = None):
        from ultralytics import YOLO  # import qui: i moduli che non estraggono non devono caricare torch

        self.modello_pose = YOLO(percorso_modello_pose)
        self.modello_armi = YOLO(percorso_modello_armi) if percorso_modello_armi else None
        self.modello_mani = YOLO(percorso_modello_mani) if percorso_modello_mani else None   # D44
        self.dispositivo = dispositivo
        self.percorso_tracker = tracker
        # 0,1 è il default di Ultralytics per .track (D3): il tracker usa anche le detection poco sicure nel
        # secondo passaggio. Con 0,25 le tracce in una scena affollata duravano la metà (prova del 2026-10-17).
        self.confidenza_minima_persone = confidenza_minima_persone
        # soglia bassa in estrazione: il filtro vero (≥ 0,5 per il WARNING ARMA) si applica dopo, così si può tarare
        self.confidenza_minima_armi = confidenza_minima_armi
        self.mezza_precisione = mezza_precisione  # D18 rivista: FP16 più lento (limite CPU), spento di default
        self.dimensione_lotto = dimensione_lotto
        self.tracker = None

    def nuovo_video(self, telecamera_mobile: bool = False) -> None:
        """Tracker nuovo per ogni video (come `.track` con persist=False al primo frame)."""
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        configurazione = IterableSimpleNamespace(**YAML.load(check_yaml(self.percorso_tracker)))
        configurazione.device = self.dispositivo  # come .track (serve solo se si attiva la ReID)
        if not telecamera_mobile and hasattr(configurazione, "gmc_method"):
            configurazione.gmc_method = "none"
        self.tracker = TRACKER_MAP[configurazione.tracker_type](args=configurazione)

    def analizza_lotto(self, frame: list[np.ndarray]) -> list[tuple[dict, dict]]:
        """Persone tracciate e armi per ogni frame del lotto, nell'ordine. Richiede nuovo_video() prima."""
        import torch

        if self.tracker is None:
            self.nuovo_video()
        risultati_pose = self.modello_pose.predict(frame, device=self.dispositivo, conf=self.confidenza_minima_persone,
                                                   half=self.mezza_precisione, verbose=False)
        risultati_armi = (self.modello_armi.predict(frame, device=self.dispositivo, conf=self.confidenza_minima_armi,
                                                    half=self.mezza_precisione, verbose=False)
                          if self.modello_armi is not None else [None] * len(frame))
        analisi = []
        for risultato_pose, risultato_armi in zip(risultati_pose, risultati_armi):
            # stessa logica di ultralytics.trackers.track.on_predict_postprocess_end
            tracce = self.tracker.update(risultato_pose.boxes.cpu().numpy(), risultato_pose.orig_img)
            persone = PERSONE_VUOTE
            if len(tracce):
                tracciate = risultato_pose[tracce[:, -1].astype(int)]
                tracciate.update(boxes=torch.as_tensor(tracce[:, :-1], device=risultato_pose.boxes.data.device))
                persone = {"id_traccia": tracciate.boxes.id.int().cpu().numpy(),
                           "box": tracciate.boxes.xyxy.cpu().numpy(),
                           "confidenza": tracciate.boxes.conf.cpu().numpy(),
                           "keypoint": tracciate.keypoints.data.cpu().numpy()}  # (persone, 17, 3)
            armi = ARMI_VUOTE
            if risultato_armi is not None and len(risultato_armi.boxes):
                armi = {"box": risultato_armi.boxes.xyxy.cpu().numpy(),
                        "classe": risultato_armi.boxes.cls.int().cpu().numpy(),
                        "confidenza": risultato_armi.boxes.conf.cpu().numpy()}
            analisi.append((persone, armi))
        if self.modello_mani is not None:   # ingrandimento sulle mani delle persone tracciate (D44), a lotti
            probabilita = probabilita_armi_in_mano_lotto(self.modello_mani, frame, [p["box"] for p, _ in analisi],
                                                         [p["keypoint"] for p, _ in analisi], self.dispositivo)
            analisi = [({**persone, "mani": valori}, armi) for (persone, armi), valori in zip(analisi, probabilita)]
        return analisi

    def analizza_frame(self, frame: np.ndarray, primo_frame_del_video: bool,
                       telecamera_mobile: bool = False) -> tuple[dict, dict]:
        """Un frame alla volta (demo dal vivo): stesso tracker e stessa logica dell'estrazione."""
        if primo_frame_del_video:
            self.nuovo_video(telecamera_mobile)
        return self.analizza_lotto([frame])[0]

    def estrai_video(self, video: VideoDaEstrarre) -> dict[str, np.ndarray]:
        """Tutti i frame del video a 25 fps → array "lunghi" (una riga per persona o arma per frame)."""
        info = leggi_info_video(video.percorso_video)
        self.nuovo_video(video.telecamera_mobile)
        righe_persone, righe_armi, lotto = [], [], []

        def elabora_lotto():
            for persone, armi in self.analizza_lotto(lotto):
                righe_persone.append((len(righe_persone), persone))
                righe_armi.append((len(righe_armi), armi))
            lotto.clear()

        for frame in leggi_frame(video.percorso_video, FPS_ANALISI, video.inizio_secondi, video.fine_secondi):
            lotto.append(frame)
            if len(lotto) == self.dimensione_lotto:
                elabora_lotto()
        if lotto:
            elabora_lotto()
        return impacchetta_estrazione(righe_persone, righe_armi, len(righe_persone), info.larghezza, info.altezza,
                                      video.inizio_secondi or 0.0)


def impacchetta_estrazione(righe_persone, righe_armi, numero_frame: int, larghezza: int, altezza: int,
                           inizio_secondi: float) -> dict[str, np.ndarray]:
    """Converte le liste per frame in array compatti per np.savez_compressed."""
    def concatena(righe, chiave, forma_vuota, tipo):
        blocchi = [dati[chiave] for _, dati in righe if len(dati[chiave])]
        return np.concatenate(blocchi).astype(tipo) if blocchi else np.empty(forma_vuota, tipo)

    def indici_frame(righe, chiave):
        blocchi = [np.full(len(dati[chiave]), indice_frame, np.int32) for indice_frame, dati in righe]
        return np.concatenate(blocchi) if blocchi else np.empty(0, np.int32)

    return {
        "numero_frame": np.int32(numero_frame),
        "larghezza": np.int32(larghezza),
        "altezza": np.int32(altezza),
        "fps": np.int32(FPS_ANALISI),
        "inizio_secondi": np.float32(inizio_secondi),
        "persone_frame": indici_frame(righe_persone, "id_traccia"),
        "persone_id_traccia": concatena(righe_persone, "id_traccia", (0,), np.int32),
        "persone_box": concatena(righe_persone, "box", (0, 4), np.float32),
        "persone_confidenza": concatena(righe_persone, "confidenza", (0,), np.float16),
        "persone_keypoint": concatena(righe_persone, "keypoint", (0, 17, 3), np.float32),
        "armi_frame": indici_frame(righe_armi, "classe"),
        "armi_box": concatena(righe_armi, "box", (0, 4), np.float32),
        "armi_classe": concatena(righe_armi, "classe", (0,), np.int32),
        "armi_confidenza": concatena(righe_armi, "confidenza", (0,), np.float16),
        **({"persone_mani": concatena(righe_persone, "mani", (0, len(CLASSI_ARMA_MANI)), np.float16)}
           if righe_persone and all("mani" in dati for _, dati in righe_persone) else {}),
    }


def estrai_elenco(video_da_estrarre: list[VideoDaEstrarre], cartella_uscita: str | Path, estrattore: Estrattore,
                  parte: int = 0, parti: int = 1, cartelle_gia_estratte: list[str | Path] = (),
                  secondi_massimi: float | None = None) -> None:
    """Estrae la propria parte dell'elenco. Con `secondi_massimi` si ferma (pulito) prima di sforare il tempo:
    le sessioni Kaggle durano al massimo 12 h e quello che manca si riprende nella sessione dopo."""
    cartella_uscita = Path(cartella_uscita)
    cartella_uscita.mkdir(parents=True, exist_ok=True)
    gia_fatti = {file.name for cartella in [cartella_uscita, *map(Path, cartelle_gia_estratte)]
                 if cartella.exists() for file in cartella.glob("*.npz")}
    mia_parte = video_da_estrarre[parte::parti]
    da_fare = [video for video in mia_parte if nome_file_estratto(video.id_clip) not in gia_fatti]
    print(f"[parte {parte}/{parti}] {len(mia_parte)} video, {len(mia_parte) - len(da_fare)} già estratti, "
          f"{len(da_fare)} da fare", flush=True)

    frame_totali, inizio = 0, time.perf_counter()
    for numero, video in enumerate(da_fare, 1):
        if secondi_massimi is not None and time.perf_counter() - inizio > secondi_massimi:
            print(f"[parte {parte}] tempo massimo raggiunto: mi fermo, {len(da_fare) - numero + 1} video restano "
                  f"per la prossima sessione", flush=True)
            break
        file_finale = cartella_uscita / nome_file_estratto(video.id_clip)
        try:
            dati = estrattore.estrai_video(video)
        except Exception as errore:  # un video rotto non deve fermare ore di estrazione
            print(f"  ERRORE {video.id_clip}: {errore}", flush=True)
            with open(cartella_uscita / f"errori_parte{parte}.txt", "a") as file_errori:
                file_errori.write(f"{video.id_clip}\t{errore}\n")
            continue
        file_temporaneo = file_finale.with_suffix(".tmp.npz")
        np.savez_compressed(file_temporaneo, **dati)
        os.replace(file_temporaneo, file_finale)  # mai un .npz a metà, anche se la sessione Kaggle si chiude
        frame_totali += int(dati["numero_frame"])
        secondi = time.perf_counter() - inizio
        print(f"  [{numero}/{len(da_fare)}] {video.id_clip}: {int(dati['numero_frame'])} frame, "
              f"media {frame_totali / secondi:.1f} frame/s", flush=True)


def comando_estrai(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elenco", required=True, help="file .jsonl con un VideoDaEstrarre per riga")
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--parte", type=int, default=0)
    parser.add_argument("--parti", type=int, default=1)
    parser.add_argument("--dispositivo", default=None, help="es. 0 per la prima GPU, cpu per la CPU")
    parser.add_argument("--modello-pose", default=MODELLO_POSE_PREDEFINITO)
    parser.add_argument("--modello-armi", default=None, help="pesi del detector armi fine-tunato (D4)")
    parser.add_argument("--modello-mani", default=None, help="classificatore dell'ingrandimento sulle mani (D44)")
    parser.add_argument("--tracker", default=TRACKER_PREDEFINITO,
                        help="configurazione del tracker di Ultralytics (D47: botsort.yaml per la riestrazione)")
    parser.add_argument("--gia-estratti", nargs="*", default=[], help="cartelle con .npz di sessioni precedenti")
    parser.add_argument("--ore-massime", type=float, default=None, help="fermati dopo queste ore (sessioni Kaggle)")
    parser.add_argument("--mezza-precisione", action="store_true", help="FP16 su GPU (D18)")
    argomenti = parser.parse_args(argomenti)

    estrattore = Estrattore(argomenti.modello_pose, argomenti.modello_armi, argomenti.dispositivo,
                            tracker=argomenti.tracker, mezza_precisione=argomenti.mezza_precisione,
                            percorso_modello_mani=argomenti.modello_mani)
    estrai_elenco(leggi_elenco(argomenti.elenco), argomenti.uscita, estrattore, argomenti.parte, argomenti.parti,
                  argomenti.gia_estratti, argomenti.ore_massime * 3600 if argomenti.ore_massime else None)


# ==============================================================================================================
# aggiungi_armi
# Secondo passaggio OPZIONALE (DA DECIDERE, punto 5 del diario): aggiunge le armi a .npz estratti con la sola pose.
#
# Rilegge ogni video a 25 fps (stesso segmento dell'estrazione), passa solo il detector armi e sostituisce i campi
# armi_* nel .npz. Le persone non vengono toccate.
#
# Uso:
#     python -m sicurezza.visione aggiungi-armi --elenco elenco.jsonl --estratti estratti/ --uscita estratti_con_armi/         --modello-armi best.pt --parte 0 --parti 2 --dispositivo 0
# ==============================================================================================================

def armi_di_un_video(modello_armi, video, dispositivo, confidenza_minima: float = 0.25,
                     dimensione_lotto: int = 8, modello_mani=None, persone: dict | None = None) -> dict[str, np.ndarray]:
    """Il detector armi su tutti i frame (a lotti, come l'estrazione: D20). Con `modello_mani` e le persone già
    estratte (campi persone_* del .npz) anche l'ingrandimento sulle mani (D44): `persone_mani` [righe persone, 3]."""
    frame_armi, box, classi, confidenze, lotto = [], [], [], [], []
    if modello_mani is not None:
        persone_frame = np.asarray(persone["persone_frame"])
        ordine = np.argsort(persone_frame, kind="stable")
        inizi_righe = np.searchsorted(persone_frame[ordine], np.arange(persone_frame.max(initial=-1) + 2))
        persone_mani = np.zeros((len(persone_frame), len(CLASSI_ARMA_MANI)), np.float32)

    def righe_del_frame(indice_frame):
        if indice_frame + 1 >= len(inizi_righe):
            return ordine[:0]
        return ordine[inizi_righe[indice_frame]:inizi_righe[indice_frame + 1]]

    def elabora_lotto(primo_indice):
        for scostamento, risultato in enumerate(modello_armi.predict(lotto, device=dispositivo, conf=confidenza_minima,
                                                                     verbose=False)):
            if len(risultato.boxes):
                frame_armi.append(np.full(len(risultato.boxes), primo_indice + scostamento, np.int32))
                box.append(risultato.boxes.xyxy.cpu().numpy())
                classi.append(risultato.boxes.cls.int().cpu().numpy())
                confidenze.append(risultato.boxes.conf.cpu().numpy())
        if modello_mani is not None:
            righe = [righe_del_frame(primo_indice + scostamento) for scostamento in range(len(lotto))]
            probabilita = probabilita_armi_in_mano_lotto(
                modello_mani, lotto, [np.asarray(persone["persone_box"])[r] for r in righe],
                [np.asarray(persone["persone_keypoint"])[r] for r in righe], dispositivo)
            for righe_frame, valori in zip(righe, probabilita):
                persone_mani[righe_frame] = valori
        lotto.clear()

    indice_frame = -1
    for indice_frame, frame in enumerate(leggi_frame(video.percorso_video, FPS_ANALISI, video.inizio_secondi,
                                                     video.fine_secondi)):
        lotto.append(frame)
        if len(lotto) == dimensione_lotto:
            elabora_lotto(indice_frame - dimensione_lotto + 1)
    if lotto:
        elabora_lotto(indice_frame - len(lotto) + 1)
    concatena = lambda blocchi, forma, tipo: np.concatenate(blocchi).astype(tipo) if blocchi else np.empty(forma, tipo)
    uscita = {"armi_frame": concatena(frame_armi, (0,), np.int32), "armi_box": concatena(box, (0, 4), np.float32),
              "armi_classe": concatena(classi, (0,), np.int32),
              "armi_confidenza": concatena(confidenze, (0,), np.float16)}
    if modello_mani is not None:
        uscita["persone_mani"] = persone_mani.astype(np.float16)
    return uscita


def comando_aggiungi_armi(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elenco", required=True)
    parser.add_argument("--estratti", required=True, help="cartella con i .npz della sola pose")
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--modello-armi", required=True)
    parser.add_argument("--modello-mani", default=None, help="classificatore dell'ingrandimento sulle mani (D44)")
    parser.add_argument("--parte", type=int, default=0)
    parser.add_argument("--parti", type=int, default=1)
    parser.add_argument("--dispositivo", default=None)
    argomenti = parser.parse_args(argomenti)

    from ultralytics import YOLO
    modello_armi = YOLO(argomenti.modello_armi)
    modello_mani = YOLO(argomenti.modello_mani) if argomenti.modello_mani else None
    cartella_uscita = Path(argomenti.uscita)
    cartella_uscita.mkdir(parents=True, exist_ok=True)
    for video in leggi_elenco(argomenti.elenco)[argomenti.parte::argomenti.parti]:
        nome = nome_file_estratto(video.id_clip)
        sorgente, destinazione = Path(argomenti.estratti) / nome, cartella_uscita / nome
        if not sorgente.exists() or destinazione.exists():
            continue
        dati = dict(np.load(sorgente))
        armi = armi_di_un_video(modello_armi, video, argomenti.dispositivo, modello_mani=modello_mani, persone=dati)
        if len(armi["armi_frame"]) and armi["armi_frame"].max() >= int(dati["numero_frame"]):
            print(f"ATTENZIONE {video.id_clip}: numero di frame diverso dall'estrazione, salto")
            continue
        dati.update(armi)
        temporaneo = destinazione.with_suffix(".tmp.npz")
        np.savez_compressed(temporaneo, **dati)
        os.replace(temporaneo, destinazione)
        print(f"{video.id_clip}: {len(armi['armi_frame'])} armi", flush=True)


# ==============================================================================================================
# armi/unisci_dataset
# Unisce più export Roboflow in formato YOLO in un unico dataset con classi pistol / knife / rifle (P2).
#
# Le classi non previste (grenade, helmet, ...) vengono tolte dalle etichette, ma le immagini restano: diventano
# esempi "senza arma" o con meno armi, utili contro i falsi positivi.
#
# Uso:
#     python -m sicurezza.visione unisci-armi --sorgenti export_master/ export_prospace/ --uscita armi_unite/
# ==============================================================================================================

CLASSI_ARMA = ("pistol", "knife", "rifle")
SINONIMI = {"pistol": "pistol", "handgun": "pistol", "gun": "pistol", "knife": "knife", "rifle": "rifle",
            "short_rifle": "rifle"}
CARTELLE_SPLIT = {"train": "train", "valid": "val", "val": "val", "test": "test"}
ESTENSIONI_IMMAGINE = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def nomi_classi_export(cartella_export: Path) -> list[str]:
    configurazione = yaml.safe_load((cartella_export / "data.yaml").read_text())
    nomi = configurazione["names"]
    return [nomi[indice] for indice in sorted(nomi)] if isinstance(nomi, dict) else list(nomi)


def unisci(cartelle_export: list[Path], cartella_uscita: Path, area_massima_train: float | None = None,
           solo_split: tuple[str, ...] = ("train", "val", "test"), collega: bool = False) -> Counter:
    """area_massima_train (D37): in train e val si tolgono le immagini con un'arma il cui riquadro supera questa
    frazione dell'immagine (primi piani, lontani dalla CCTV). Il test resta intero, per confrontare i detector."""
    conteggio = Counter()
    for numero_sorgente, cartella_export in enumerate(cartelle_export):
        nomi = nomi_classi_export(cartella_export)
        nuovo_indice = {indice: CLASSI_ARMA.index(SINONIMI[nome.lower()])
                        for indice, nome in enumerate(nomi) if nome.lower() in SINONIMI}
        for cartella_split_export, split in CARTELLE_SPLIT.items():
            cartella_immagini = cartella_export / cartella_split_export / "images"
            if not cartella_immagini.exists() or split not in solo_split:
                continue
            (cartella_uscita / "images" / split).mkdir(parents=True, exist_ok=True)
            (cartella_uscita / "labels" / split).mkdir(parents=True, exist_ok=True)
            for immagine in sorted(cartella_immagini.iterdir()):
                if immagine.suffix.lower() not in ESTENSIONI_IMMAGINE:
                    continue
                nome_nuovo = f"s{numero_sorgente}_{immagine.name}"   # niente collisioni tra export diversi
                file_etichette = cartella_export / cartella_split_export / "labels" / (immagine.stem + ".txt")
                righe_nuove, classi_tolte = [], []
                if file_etichette.exists():
                    for riga in file_etichette.read_text().splitlines():
                        valori = riga.split()
                        if valori and int(valori[0]) in nuovo_indice:
                            righe_nuove.append(" ".join([str(nuovo_indice[int(valori[0])]), *valori[1:]]))
                        elif valori:
                            classi_tolte.append(nomi[int(valori[0])])
                if area_massima_train is not None and split != "test" and any(
                        float(riga.split()[3]) * float(riga.split()[4]) > area_massima_train for riga in righe_nuove):
                    conteggio[(split, "tolta:primo_piano")] += 1
                    continue
                for riga in righe_nuove:
                    conteggio[(split, CLASSI_ARMA[int(riga.split()[0])])] += 1
                for nome_classe in classi_tolte:
                    conteggio[(split, f"tolta:{nome_classe}")] += 1
                if collega:   # collegamento invece della copia: veloce quando le immagini stanno sul disco USB
                    (cartella_uscita / "images" / split / nome_nuovo).symlink_to(immagine.resolve())
                else:
                    shutil.copy2(immagine, cartella_uscita / "images" / split / nome_nuovo)
                (cartella_uscita / "labels" / split / (Path(nome_nuovo).stem + ".txt")).write_text(
                    "\n".join(righe_nuove) + ("\n" if righe_nuove else ""))
                conteggio[(split, "immagini")] += 1
    percorsi = {split: f"images/{split}" for split in ("train", "val", "test")
                if (cartella_uscita / "images" / split).exists()}
    if "test" in percorsi:   # dataset di sola valutazione: Ultralytics vuole comunque train e val nel data.yaml
        percorsi = {"train": percorsi["test"], "val": percorsi["test"], **percorsi}
    (cartella_uscita / "data.yaml").write_text(yaml.safe_dump({
        "path": str(cartella_uscita.resolve()), **percorsi, "names": dict(enumerate(CLASSI_ARMA))}, sort_keys=False))
    return conteggio


def comando_unisci_armi(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sorgenti", nargs="+", required=True)
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--area-massima-train", type=float, default=None,
                        help="toglie da train/val le immagini con armi più grandi di questa frazione (D37)")
    parser.add_argument("--solo-test", action="store_true", help="converte solo il test (per confrontare detector)")
    parser.add_argument("--collega", action="store_true", help="collegamenti simbolici alle immagini invece di copie")
    argomenti = parser.parse_args(argomenti)
    solo_split = ("test",) if argomenti.solo_test else ("train", "val", "test")
    for (split, voce), numero in sorted(unisci([Path(s) for s in argomenti.sorgenti], Path(argomenti.uscita),
                                               argomenti.area_massima_train, solo_split, argomenti.collega).items()):
        print(f"{split:6s} {voce:20s} {numero}")


# ==============================================================================================================
# US Mock Attack (Univ. Siviglia, Hugging Face, CC BY-NC 4.0) — D30
# 5149 frame 1920×1080 di 20 sequenze CCTV (3 telecamere), annotazioni VOC (Handgun, Short_rifle, Knife), ~70% di
# frame senza armi (utili come negativi). Non ha split: lo facciamo PER SEQUENZA (frame vicini della stessa sequenza
# in train e test falserebbero la misura), 70/20/10 come D25.
# ==============================================================================================================
URL_MOCK_ATTACK = ("https://huggingface.co/datasets/jsalazar/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset"
                   "/resolve/main/weapons_images_2fps.zip")
CLASSI_MOCK_ATTACK = ("Handgun", "Knife", "Short_rifle")
QUOTE_SPLIT_MOCK_ATTACK = {"train": 0.7, "valid": 0.2, "test": 0.1}


def sequenza_del_frame(nome_file: str) -> str:
    """Cam1-From09-23-00To10-03-25_Segment_0_x264_frame_1.jpg → Cam1-From09-23-00To10-03-25_Segment_0_x264"""
    import re
    return re.sub(r"_frame_\d+$", "", Path(nome_file).stem)


def dividi_sequenze(frame_per_sequenza: dict[str, int], seme: int = 0) -> dict[str, str]:
    """Ogni sequenza intera in uno split, cercando di rispettare le quote in numero di frame."""
    import random
    sequenze = sorted(frame_per_sequenza)
    random.Random(seme).shuffle(sequenze)
    totale, cumulati, split_della_sequenza = sum(frame_per_sequenza.values()), 0, {}
    soglie = [("train", QUOTE_SPLIT_MOCK_ATTACK["train"]),
              ("valid", QUOTE_SPLIT_MOCK_ATTACK["train"] + QUOTE_SPLIT_MOCK_ATTACK["valid"]), ("test", 1.0)]
    for sequenza in sequenze:
        centro = (cumulati + frame_per_sequenza[sequenza] / 2) / totale
        split_della_sequenza[sequenza] = next(nome for nome, soglia in soglie if centro <= soglia)
        cumulati += frame_per_sequenza[sequenza]
    return split_della_sequenza


def converti_mock_attack(cartella_voc: Path, cartella_uscita: Path, seme: int = 0) -> Counter:
    """Cartella con coppie .jpg/.xml (VOC) → export in stile Roboflow YOLO (train/valid/test + data.yaml),
    così `unisci` lo tratta come gli altri dataset."""
    import xml.etree.ElementTree as ET

    immagini = sorted(Path(cartella_voc).rglob("*.jpg"))
    frame_per_sequenza = Counter(sequenza_del_frame(immagine.name) for immagine in immagini)
    split_della_sequenza = dividi_sequenze(frame_per_sequenza, seme)
    conteggio = Counter()
    for immagine in immagini:
        split = split_della_sequenza[sequenza_del_frame(immagine.name)]
        (cartella_uscita / split / "images").mkdir(parents=True, exist_ok=True)
        (cartella_uscita / split / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(immagine, cartella_uscita / split / "images" / immagine.name)
        righe = []
        file_xml = immagine.with_suffix(".xml")
        if file_xml.exists():
            radice = ET.parse(file_xml).getroot()
            larghezza = float(radice.findtext("size/width"))
            altezza = float(radice.findtext("size/height"))
            for oggetto in radice.iter("object"):
                nome = oggetto.findtext("name")
                if nome not in CLASSI_MOCK_ATTACK:
                    conteggio[(split, f"sconosciuta:{nome}")] += 1
                    continue
                box = oggetto.find("bndbox")
                x1, y1, x2, y2 = (float(box.findtext(lato)) for lato in ("xmin", "ymin", "xmax", "ymax"))
                righe.append(f"{CLASSI_MOCK_ATTACK.index(nome)} {(x1 + x2) / 2 / larghezza:.6f} "
                             f"{(y1 + y2) / 2 / altezza:.6f} {(x2 - x1) / larghezza:.6f} {(y2 - y1) / altezza:.6f}")
                conteggio[(split, nome)] += 1
        (cartella_uscita / split / "labels" / (immagine.stem + ".txt")).write_text(
            "\n".join(righe) + ("\n" if righe else ""))
        conteggio[(split, "immagini")] += 1
        conteggio[(split, "senza_armi")] += not righe
    (cartella_uscita / "data.yaml").write_text(yaml.safe_dump(
        {"train": "../train/images", "val": "../valid/images", "test": "../test/images",
         "nc": len(CLASSI_MOCK_ATTACK), "names": list(CLASSI_MOCK_ATTACK)}, sort_keys=False))
    return conteggio


def comando_converti_mock_attack(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sorgente", required=True, help="cartella con le coppie .jpg/.xml di weapons_images_2fps.zip")
    parser.add_argument("--uscita", required=True)
    argomenti = parser.parse_args(argomenti)
    for (split, voce), numero in sorted(converti_mock_attack(Path(argomenti.sorgente), Path(argomenti.uscita)).items()):
        print(f"{split:6s} {voce:20s} {numero}")


# ==============================================================================================================
# Sohas (DaSCI, Univ. Granada, CC BY-SA 4.0) — D39
# "Small Objects Handled Similarly to a weapon": pistole e coltelli impugnati + smartphone, borsellini, banconote e
# tessere impugnati come un'arma. Contiene quasi tutto DaSCI Knife_detection (1985 immagini su 2078 con lo stesso
# nome), quindi basta questo. Le 4 classi innocue non sono in SINONIMI: `unisci` le toglie dalle etichette e le
# immagini restano come negativi difficili. Il loro test resta il nostro test; dal loro train si prende il 10% per val.
# Si scarica con: git clone --depth 1 --filter=blob:none --sparse https://github.com/ari-dasci/OD-WeaponDetection
#                 git sparse-checkout set "Weapons and similar handled objects/Sohas_weapon-Detection-YOLOv5"
# ==============================================================================================================
URL_DASCI = "https://github.com/ari-dasci/OD-WeaponDetection"
CARTELLA_SOHAS_NEL_REPO = "Weapons and similar handled objects/Sohas_weapon-Detection-YOLOv5"


def prepara_sohas(cartella_sohas: Path, cartella_uscita: Path, quota_val: float = 0.1, seme: int = 0) -> Counter:
    """cartella_sohas: quella con dataset.yaml e obj_train_data/{images,labels}/{train,test} → export in stile
    Roboflow (train/valid/test + data.yaml)."""
    import random
    nomi = yaml.safe_load((cartella_sohas / "dataset.yaml").read_text())["names"]
    dati = cartella_sohas / "obj_train_data"
    conteggio = Counter()
    for split_sohas in ("train", "test"):
        immagini = sorted(p for p in (dati / "images" / split_sohas).iterdir() if p.suffix.lower() in ESTENSIONI_IMMAGINE)
        random.Random(seme).shuffle(immagini)
        quante_val = round(len(immagini) * quota_val) if split_sohas == "train" else 0
        for posizione, immagine in enumerate(immagini):
            split = "test" if split_sohas == "test" else ("valid" if posizione < quante_val else "train")
            (cartella_uscita / split / "images").mkdir(parents=True, exist_ok=True)
            (cartella_uscita / split / "labels").mkdir(parents=True, exist_ok=True)
            shutil.copy2(immagine, cartella_uscita / split / "images" / immagine.name)
            etichette = dati / "labels" / split_sohas / (immagine.stem + ".txt")
            testo = etichette.read_text() if etichette.exists() else ""
            (cartella_uscita / split / "labels" / (immagine.stem + ".txt")).write_text(testo)
            conteggio[(split, "immagini")] += 1
            for riga in testo.splitlines():
                if riga.strip():
                    conteggio[(split, nomi[int(riga.split()[0])])] += 1
    (cartella_uscita / "data.yaml").write_text(yaml.safe_dump(
        {"train": "../train/images", "val": "../valid/images", "test": "../test/images", "nc": len(nomi),
         "names": list(nomi)}, sort_keys=False))
    return conteggio


def comando_prepara_sohas(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sorgente", required=True, help=f"la cartella '{CARTELLA_SOHAS_NEL_REPO}' del repo DaSCI")
    parser.add_argument("--uscita", required=True)
    argomenti = parser.parse_args(argomenti)
    for (split, voce), numero in sorted(prepara_sohas(Path(argomenti.sorgente), Path(argomenti.uscita)).items()):
        print(f"{split:6s} {voce:20s} {numero}")


# ==============================================================================================================
# armi/addestra_detector
# Fine-tuning di YOLO26s sul dataset armi unito (D4). Iperparametri PROVVISORI (DA DECIDERE).
#
# Uso (su Kaggle, GPU):
#     python -m sicurezza.visione addestra-detector --dati armi_unite/data.yaml --uscita runs/armi
# ==============================================================================================================

def addestra_detector(percorso_data_yaml: str, cartella_uscita: str, modello_base: str = "yolo26s.pt",
                      epoche: int = 100, pazienza: int = 20, dimensione_immagine: int = 640, lotto: int = 16,
                      dispositivo: str | int | None = None):
    """dispositivo "0,1" = due GPU (DDP). Niente `time` di Ultralytics: allungherebbe le epoche fino a riempire tutto
    il tempo (quota GPU sprecata); il tetto delle 5 h di D37 lo mette il kernel fermando il processo."""
    from ultralytics import YOLO

    # percorso assoluto: con uno relativo Ultralytics 8.4 salva sotto runs/<task>/<percorso>
    YOLO(modello_base).train(data=percorso_data_yaml, epochs=epoche, patience=pazienza, imgsz=dimensione_immagine,
                             batch=lotto, device=dispositivo, project=str(Path(cartella_uscita).resolve()),
                             name="yolo26s_armi", exist_ok=True, seed=0)
    return valuta_detector(Path(cartella_uscita) / "yolo26s_armi" / "weights" / "best.pt", percorso_data_yaml,
                           dispositivo)


def valuta_detector(percorso_pesi: Path, percorso_data_yaml: str, dispositivo: str | int | None = None):
    """Con più GPU l'addestramento gira in sottoprocessi: si rilegge il migliore da disco e si valuta su UNA GPU."""
    from ultralytics import YOLO

    prima_gpu = str(dispositivo).split(",")[0] if dispositivo is not None else None
    return YOLO(str(percorso_pesi)).val(data=percorso_data_yaml, device=prima_gpu,
                                        split="test" if _ha_test(percorso_data_yaml) else "val")


def _ha_test(percorso_data_yaml: str) -> bool:
    import yaml
    return "test" in yaml.safe_load(open(percorso_data_yaml))


def comando_addestra_detector(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dati", required=True)
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--epoche", type=int, default=100)
    parser.add_argument("--lotto", type=int, default=16, help="totale: con 2 GPU metà per GPU")
    parser.add_argument("--dispositivo", default=None, help='"0", oppure "0,1" per due GPU')
    parser.add_argument("--solo-valutazione", action="store_true",
                        help="valuta best.pt già in --uscita (es. dopo che il kernel ha fermato l'addestramento)")
    argomenti = parser.parse_args(argomenti)
    if argomenti.solo_valutazione:
        metriche = valuta_detector(Path(argomenti.uscita) / "yolo26s_armi" / "weights" / "best.pt", argomenti.dati,
                                   argomenti.dispositivo)
    else:
        metriche = addestra_detector(argomenti.dati, argomenti.uscita, epoche=argomenti.epoche, lotto=argomenti.lotto,
                                     dispositivo=argomenti.dispositivo)
    print(f"mAP50 {metriche.box.map50:.4f}  mAP50-95 {metriche.box.map:.4f}")


# ==============================================================================================================
# armi/falsi_allarmi
# Verifica del piano: il detector non deve scattare su oggetti innocui (coltelli da cucina posati, telefoni, forbici,
# ombrelli). È il motivo per cui non usiamo il "knife" di COCO (D4). Su immagini SENZA armi impugnate conta quelle in
# cui il detector trova qualcosa sopra soglia, divise per categoria (= nome della sottocartella). Per ogni arma
# trovata: classe, confidenza, area del riquadro in proporzione all'immagine.
# Una sola immagine è il caso peggiore: il WARNING ARMA vuole l'arma in più frame vicino a una persona (D6).
# Le immagini di prova si scaricano con `python -m sicurezza.strumenti scarica-negativi-coco`.
#
# Uso:
#     python -m sicurezza.visione falsi-allarmi --immagini data/negativi_coco --modello-armi best.pt --uscita falsi.json
# ==============================================================================================================

ESTENSIONI_IMMAGINE = {".jpg", ".jpeg", ".png"}


def conta_falsi_allarmi(modello_armi, cartella_immagini: Path, soglia_confidenza: float = 0.5, dispositivo=None,
                        dimensione_lotto: int = 8) -> dict:
    percorsi = sorted(p for p in cartella_immagini.rglob("*") if p.suffix.lower() in ESTENSIONI_IMMAGINE)
    immagini_per_categoria, allarmi_per_categoria = Counter(), Counter()
    classi_per_categoria: dict[str, Counter] = {}
    immagini_con_allarme = []
    for inizio in range(0, len(percorsi), dimensione_lotto):
        lotto = percorsi[inizio:inizio + dimensione_lotto]
        risultati = modello_armi.predict([str(p) for p in lotto], device=dispositivo, conf=soglia_confidenza,
                                         verbose=False)
        for percorso, risultato in zip(lotto, risultati):
            categoria = percorso.parent.name
            immagini_per_categoria[categoria] += 1
            # area del riquadro / area dell'immagine: distingue l'oggetto scambiato per arma dal riquadro su mezza scena
            armi = [(modello_armi.names[int(classe)], round(float(confidenza), 3), round(float(xywhn[2] * xywhn[3]), 3))
                    for classe, confidenza, xywhn in zip(risultato.boxes.cls, risultato.boxes.conf,
                                                          risultato.boxes.xywhn)]
            if armi:
                allarmi_per_categoria[categoria] += 1
                classi_per_categoria.setdefault(categoria, Counter()).update(nome for nome, _, _ in armi)
                immagini_con_allarme.append({"immagine": str(percorso.relative_to(cartella_immagini)), "armi": armi})
    per_categoria = {categoria: {"immagini": numero, "con_allarme": allarmi_per_categoria[categoria],
                                 "frazione": round(allarmi_per_categoria[categoria] / numero, 4),
                                 "classi": dict(classi_per_categoria.get(categoria, {}))}
                     for categoria, numero in sorted(immagini_per_categoria.items())}
    return {"soglia_confidenza": soglia_confidenza, "per_categoria": per_categoria,
            "immagini_con_allarme": immagini_con_allarme}


def comando_falsi_allarmi(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--immagini", required=True, help="cartella con una sottocartella per categoria")
    parser.add_argument("--modello-armi", required=True)
    parser.add_argument("--soglia", type=float, default=0.5, help="come SOGLIA_CONFIDENZA_ARMA della demo (D6)")
    parser.add_argument("--uscita", default=None, help="file .json con il dettaglio (facoltativo)")
    parser.add_argument("--dispositivo", default=None)
    argomenti = parser.parse_args(argomenti)

    from ultralytics import YOLO
    risultato = conta_falsi_allarmi(YOLO(argomenti.modello_armi), Path(argomenti.immagini), argomenti.soglia,
                                    argomenti.dispositivo)
    for categoria, conteggi in risultato["per_categoria"].items():
        print(f"{categoria:12s} {conteggi['con_allarme']:4d} / {conteggi['immagini']:4d} immagini con allarme  "
              f"{conteggi['classi']}")
    if argomenti.uscita:
        Path(argomenti.uscita).write_text(json.dumps(risultato, indent=1, ensure_ascii=False))

# ==============================================================================================================
# armi/ingrandimento_mani (D44)
# A 640 px un coltello ripreso da una CCTV è largo 2-3 pixel: il detector sull'intero frame non lo vede. Dalle pose
# sappiamo dove sono le mani: si ritaglia un quadrato intorno a ogni polso alla risoluzione PIENA del video (lato =
# 0,4 × altezza della persona, centro un po' oltre il polso, dove la mano tiene l'oggetto), lo si ingrandisce a 224 px
# e un classificatore dice niente / pistola / coltello / fucile.
# Addestramento: positivi = ritagli intorno alle armi annotate (l'arma occupa la stessa frazione del ritaglio che
# avrebbe in mano a una persona); negativi = ritagli intorno ai polsi di persone senza armi, anche con telefoni,
# portafogli, tessere e banconote in mano (le immagini di Sohas senza armi) e oggetti di COCO.
#
# Uso:
#     python -m sicurezza.visione ritagli-mani --dati armi_unite/ --uscita ritagli_mani/ --dispositivo 0
#     python -m sicurezza.visione addestra-mani --dati ritagli_mani/ --uscita runs/mani --dispositivo 0
#     python -m sicurezza.visione valuta-mani --modello-mani .../best.pt --dati armi_unite/ --split test
#         [--modello-armi detector.pt] --uscita valutazione_mani.json
# ==============================================================================================================

CLASSI_ARMA_MANI = ("pistol", "knife", "rifle")   # più "niente" per le mani vuote
LATO_RITAGLIO_SU_ALTEZZA = 0.4
SPOSTAMENTO_OLTRE_IL_POLSO = 0.3     # frazione del vettore gomito→polso
LATO_MINIMO_RITAGLIO = 24            # pixel
FATTORI_LATO_ARMA = (2.5, 4.5)       # ritaglio di addestramento = U(2,5; 4,5) × lato lungo dell'arma (≈ 0,4 × altezza)
CONFIDENZA_MINIMA_POLSO = 0.3
DIMENSIONE_RITAGLIO = 224


def ritaglio_quadrato(immagine: np.ndarray, centro_x: float, centro_y: float, lato: float) -> np.ndarray:
    """Quadrato lato × lato centrato in (centro_x, centro_y); le parti fuori dall'immagine restano nere."""
    lato = max(round(lato), 1)
    x0, y0 = round(centro_x - lato / 2), round(centro_y - lato / 2)
    ritaglio = np.zeros((lato, lato, immagine.shape[2]), immagine.dtype)
    dentro_x0, dentro_y0 = max(x0, 0), max(y0, 0)
    dentro_x1, dentro_y1 = min(x0 + lato, immagine.shape[1]), min(y0 + lato, immagine.shape[0])
    if dentro_x1 > dentro_x0 and dentro_y1 > dentro_y0:
        ritaglio[dentro_y0 - y0:dentro_y1 - y0, dentro_x0 - x0:dentro_x1 - x0] = \
            immagine[dentro_y0:dentro_y1, dentro_x0:dentro_x1]
    return ritaglio


def mani_di_una_persona(box_persona: np.ndarray, keypoint_persona: np.ndarray) -> list[tuple[float, float, float]]:
    """(centro_x, centro_y, lato) del ritaglio per ogni polso visibile."""
    lato = max(LATO_RITAGLIO_SU_ALTEZZA * float(box_persona[3] - box_persona[1]), LATO_MINIMO_RITAGLIO)
    mani = []
    for polso, gomito in zip(POLSI, GOMITI):
        if not keypoint_persona[polso, 2] >= CONFIDENZA_MINIMA_POLSO:   # anche NaN = polso non visto
            continue
        x, y = (float(v) for v in keypoint_persona[polso, :2])
        if keypoint_persona[gomito, 2] >= CONFIDENZA_MINIMA_POLSO:
            x += SPOSTAMENTO_OLTRE_IL_POLSO * (x - float(keypoint_persona[gomito, 0]))
            y += SPOSTAMENTO_OLTRE_IL_POLSO * (y - float(keypoint_persona[gomito, 1]))
        mani.append((x, y, lato))
    return mani


def ritagli_delle_mani(frame: np.ndarray, box_persone: np.ndarray, keypoint_persone: np.ndarray
                       ) -> tuple[list[np.ndarray], list[int]]:
    import cv2
    ritagli, persona_del_ritaglio = [], []
    for indice, (box, keypoint) in enumerate(zip(box_persone, keypoint_persone)):
        for centro_x, centro_y, lato in mani_di_una_persona(box, keypoint):
            ritagli.append(cv2.resize(ritaglio_quadrato(frame, centro_x, centro_y, lato),
                                      (DIMENSIONE_RITAGLIO, DIMENSIONE_RITAGLIO)))
            persona_del_ritaglio.append(indice)
    return ritagli, persona_del_ritaglio


def probabilita_armi_in_mano(modello_mani, frame: np.ndarray, box_persone: np.ndarray, keypoint_persone: np.ndarray,
                             dispositivo=None) -> np.ndarray:
    """[persone, 3]: per ogni persona la probabilità più alta (tra le due mani) di pistola, coltello, fucile."""
    return probabilita_armi_in_mano_lotto(modello_mani, [frame], [box_persone], [keypoint_persone], dispositivo)[0]


def probabilita_armi_in_mano_lotto(modello_mani, frame: list[np.ndarray], box_per_frame: list[np.ndarray],
                                   keypoint_per_frame: list[np.ndarray], dispositivo=None,
                                   ritagli_per_chiamata: int = 128) -> list[np.ndarray]:
    """Come probabilita_armi_in_mano ma per più frame, con UNA chiamata al modello ogni `ritagli_per_chiamata`
    ritagli (una chiamata per frame costerebbe ore sull'intero dataset)."""
    probabilita = [np.zeros((len(box), len(CLASSI_ARMA_MANI)), np.float32) for box in box_per_frame]
    ritagli, proprietari = [], []
    for numero_frame, (immagine, box, keypoint) in enumerate(zip(frame, box_per_frame, keypoint_per_frame)):
        ritagli_frame, persone = ritagli_delle_mani(immagine, box, keypoint)
        ritagli += ritagli_frame
        proprietari += [(numero_frame, persona) for persona in persone]
    if not ritagli:
        return probabilita
    indice_classe = {nome: indice for indice, nome in modello_mani.names.items()}
    colonne = [indice_classe[nome] for nome in CLASSI_ARMA_MANI]
    for inizio in range(0, len(ritagli), ritagli_per_chiamata):
        risultati = modello_mani.predict(ritagli[inizio:inizio + ritagli_per_chiamata], imgsz=DIMENSIONE_RITAGLIO,
                                         device=dispositivo, verbose=False)
        for (numero_frame, persona), risultato in zip(proprietari[inizio:inizio + ritagli_per_chiamata], risultati):
            valori = risultato.probs.data.float().cpu().numpy()[colonne]
            probabilita[numero_frame][persona] = np.maximum(probabilita[numero_frame][persona], valori)
    return probabilita


def ritagli_per_addestramento(cartella_unita: Path, cartella_uscita: Path, modello_pose, dispositivo=None,
                              ritagli_per_arma: int = 2, mani_vuote_per_immagine: int = 4, seme: int = 0,
                              split_da_fare: tuple[str, ...] = ("train", "val", "test")) -> Counter:
    """Dataset unito in formato YOLO (images/<split>, labels/<split>) → dataset di classificazione per Ultralytics:
    cartella_uscita/<split>/<niente|pistol|knife|rifle>/*.jpg (ritagli già a 224 px)."""
    import random

    import cv2
    generatore, conteggio = random.Random(seme), Counter()
    for split in split_da_fare:
        cartella_immagini = cartella_unita / "images" / split
        if not cartella_immagini.exists():
            continue
        senza_armi = []
        for percorso in sorted(cartella_immagini.iterdir()):
            if percorso.suffix.lower() not in ESTENSIONI_IMMAGINE:
                continue
            etichette = cartella_unita / "labels" / split / (percorso.stem + ".txt")
            righe = [r.split() for r in etichette.read_text().splitlines() if r.strip()] if etichette.exists() else []
            if not righe:
                senza_armi.append(percorso)
                continue
            immagine = cv2.imread(str(percorso))
            if immagine is None:
                continue
            altezza, larghezza = immagine.shape[:2]
            for numero_arma, (classe, centro_x, centro_y, w, h) in enumerate(r[:5] for r in righe):
                lato_arma = max(float(w) * larghezza, float(h) * altezza)
                for copia in range(ritagli_per_arma):
                    lato = generatore.uniform(*FATTORI_LATO_ARMA) * max(lato_arma, 4)
                    spostamento_x, spostamento_y = (generatore.uniform(-0.25, 0.25) * lato for _ in range(2))
                    ritaglio = ritaglio_quadrato(immagine, float(centro_x) * larghezza + spostamento_x,
                                                 float(centro_y) * altezza + spostamento_y, lato)
                    nome_classe = CLASSI_ARMA[int(classe)]
                    destinazione = cartella_uscita / split / nome_classe / f"{percorso.stem}_{numero_arma}_{copia}.jpg"
                    destinazione.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(destinazione), cv2.resize(ritaglio, (DIMENSIONE_RITAGLIO, DIMENSIONE_RITAGLIO)))
                    conteggio[(split, nome_classe)] += 1
        # mani vuote: pose sulle immagini senza armi, a lotti
        (cartella_uscita / split / "niente").mkdir(parents=True, exist_ok=True)
        for inizio in range(0, len(senza_armi), 16):
            lotto = senza_armi[inizio:inizio + 16]
            immagini = [cv2.imread(str(p)) for p in lotto]
            validi = [(p, i) for p, i in zip(lotto, immagini) if i is not None]
            for (percorso, immagine), risultato in zip(validi, modello_pose.predict(
                    [i for _, i in validi], device=dispositivo, conf=0.25, verbose=False)):
                if not len(risultato.boxes):
                    continue
                box = risultato.boxes.xyxy.cpu().numpy()
                keypoint = np.concatenate([risultato.keypoints.xy.cpu().numpy(),
                                           risultato.keypoints.conf.cpu().numpy()[..., None]], -1)
                ritagli, _ = ritagli_delle_mani(immagine, box, keypoint)
                for numero, ritaglio in enumerate(generatore.sample(ritagli, min(len(ritagli), mani_vuote_per_immagine))):
                    cv2.imwrite(str(cartella_uscita / split / "niente" / f"{percorso.stem}_{numero}.jpg"), ritaglio)
                    conteggio[(split, "niente")] += 1
    return conteggio


def comando_ritagli_mani(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dati", required=True, help="dataset unito da `unisci-armi` (images/, labels/)")
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--modello-pose", default=MODELLO_POSE_PREDEFINITO)
    parser.add_argument("--dispositivo", default=None)
    parser.add_argument("--split", nargs="+", default=["train", "val", "test"], help="per lanciarne più in parallelo")
    argomenti = parser.parse_args(argomenti)
    from ultralytics import YOLO
    conteggio = ritagli_per_addestramento(Path(argomenti.dati), Path(argomenti.uscita), YOLO(argomenti.modello_pose),
                                          argomenti.dispositivo, split_da_fare=tuple(argomenti.split))
    for (split, classe), numero in sorted(conteggio.items()):
        print(f"{split:6s} {classe:8s} {numero}")


def addestra_mani(cartella_ritagli: str, cartella_uscita: str, epoche: int = 30, lotto: int = 64,
                  dispositivo: str | int | None = None):
    from ultralytics import YOLO
    # percorso assoluto: con uno relativo Ultralytics 8.4 salva sotto runs/classify/<percorso>
    YOLO("yolo26s-cls.pt").train(data=cartella_ritagli, epochs=epoche, imgsz=DIMENSIONE_RITAGLIO, batch=lotto,
                                 device=dispositivo, project=str(Path(cartella_uscita).resolve()), name="yolo26s_mani",
                                 exist_ok=True, patience=8, seed=0)
    return Path(cartella_uscita) / "yolo26s_mani" / "weights" / "best.pt"


def comando_addestra_mani(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dati", required=True, help="cartella di `ritagli-mani` (train/ val/ test/)")
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--epoche", type=int, default=30)
    parser.add_argument("--lotto", type=int, default=64)
    parser.add_argument("--dispositivo", default=None)
    argomenti = parser.parse_args(argomenti)
    print("pesi:", addestra_mani(argomenti.dati, argomenti.uscita, argomenti.epoche, argomenti.lotto,
                                 argomenti.dispositivo))


def precisione_media(punteggi: np.ndarray, positivi: np.ndarray) -> float:
    """Average precision (area sotto la curva precisione-recall) di una classe a livello di immagine."""
    if not positivi.any():
        return float("nan")
    ordine = np.argsort(-punteggi, kind="stable")
    veri = positivi[ordine].astype(float)
    precisione = np.cumsum(veri) / np.arange(1, len(veri) + 1)
    return float((precisione * veri).sum() / veri.sum())


def valuta_mani(cartella_unita: Path, split: str, modello_pose, modello_mani=None, modello_armi=None,
                dispositivo=None, soglia: float = 0.5) -> dict:
    """Per immagine: c'è una pistola / un coltello / un fucile? Confronta l'ingrandimento sulle mani con il detector
    sull'intero frame (stesse immagini): average precision e recall/falsi allarmi alla soglia."""
    import cv2
    punteggi = {"mani": [], "detector": []}
    presenti = []
    for percorso in sorted((cartella_unita / "images" / split).iterdir()):
        immagine = cv2.imread(str(percorso))
        if immagine is None:
            continue
        etichette = cartella_unita / "labels" / split / (percorso.stem + ".txt")
        classi = {int(r.split()[0]) for r in etichette.read_text().splitlines() if r.strip()} if etichette.exists() else set()
        presenti.append([CLASSI_ARMA.index(nome) in classi for nome in CLASSI_ARMA_MANI])
        if modello_mani is not None:
            risultato = modello_pose.predict(immagine, device=dispositivo, conf=0.25, verbose=False)[0]
            box = risultato.boxes.xyxy.cpu().numpy()
            keypoint = (np.concatenate([risultato.keypoints.xy.cpu().numpy(),
                                        risultato.keypoints.conf.cpu().numpy()[..., None]], -1)
                        if len(box) else np.empty((0, 17, 3)))
            per_persona = probabilita_armi_in_mano(modello_mani, immagine, box, keypoint, dispositivo)
            punteggi["mani"].append(per_persona.max(0) if len(per_persona) else np.zeros(len(CLASSI_ARMA_MANI)))
        if modello_armi is not None:
            risultato = modello_armi.predict(immagine, device=dispositivo, conf=0.05, verbose=False)[0]
            migliori = np.zeros(len(CLASSI_ARMA_MANI))
            for classe, confidenza in zip(risultato.boxes.cls.int().tolist(), risultato.boxes.conf.tolist()):
                posizione = CLASSI_ARMA_MANI.index(modello_armi.names[classe])
                migliori[posizione] = max(migliori[posizione], confidenza)
            punteggi["detector"].append(migliori)
    presenti = np.array(presenti, bool)
    valutazione = {"immagini": len(presenti), "con_classe": dict(zip(CLASSI_ARMA_MANI, presenti.sum(0).tolist()))}
    for metodo, valori in punteggi.items():
        if not valori:
            continue
        valori = np.array(valori)
        valutazione[metodo] = {nome: {
            "precisione_media": round(precisione_media(valori[:, posizione], presenti[:, posizione]), 4),
            "recall": round(float((valori[presenti[:, posizione], posizione] >= soglia).mean()), 4)
            if presenti[:, posizione].any() else None,
            "falsi_allarmi": round(float((valori[~presenti[:, posizione], posizione] >= soglia).mean()), 4)}
            for posizione, nome in enumerate(CLASSI_ARMA_MANI)}
    return valutazione


def comando_valuta_mani(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dati", required=True, help="dataset unito (images/<split>, labels/<split>)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--modello-mani", default=None)
    parser.add_argument("--modello-armi", default=None, help="detector sull'intero frame, per confronto")
    parser.add_argument("--modello-pose", default=MODELLO_POSE_PREDEFINITO)
    parser.add_argument("--dispositivo", default=None)
    parser.add_argument("--uscita", required=True)
    argomenti = parser.parse_args(argomenti)
    from ultralytics import YOLO
    valutazione = valuta_mani(Path(argomenti.dati), argomenti.split, YOLO(argomenti.modello_pose),
                              YOLO(argomenti.modello_mani) if argomenti.modello_mani else None,
                              YOLO(argomenti.modello_armi) if argomenti.modello_armi else None, argomenti.dispositivo)
    Path(argomenti.uscita).write_text(json.dumps(valutazione, indent=1, ensure_ascii=False))
    print(json.dumps(valutazione, indent=1, ensure_ascii=False))


# ==============================================================================================================
# armi/confronta_detector
# D37: tabella unica per scegliere il detector. Ogni detector su ogni test set (mAP50 per classe, precisione, recall)
# e sulle immagini senza armi di `falsi-allarmi`. I test set devono avere le classi del progetto: per un export con
# altre classi (es. Sohas) prima `unisci-armi --solo-test`.
#
# Uso:
#     python -m sicurezza.visione confronta-detector --pesi v1=runs/.../best.pt v2=... --test roboflow=data.yaml
#         mock_attack=... --negativi data/negativi_coco --uscita runs/confronto_detector.json
# ==============================================================================================================

def confronta_detector(pesi: dict[str, str], test: dict[str, str], cartella_negativi: Path | None = None,
                       dispositivo: str = "cpu") -> dict:
    from ultralytics import YOLO

    cartella_temporanea = tempfile.mkdtemp()   # Ultralytics salva sempre qualcosa: non nel repo
    tabella = {}
    for nome_detector, percorso_pesi in pesi.items():
        riga = {}
        for nome_test, data_yaml in test.items():
            misure = YOLO(percorso_pesi).val(data=data_yaml, split="test", device=dispositivo, batch=4, workers=0,
                                             plots=False, verbose=False, project=cartella_temporanea,
                                             name=f"{nome_detector}_{nome_test}")
            riga[nome_test] = {"map50": round(float(misure.box.map50), 3), "precisione": round(float(misure.box.mp), 3),
                               "recall": round(float(misure.box.mr), 3),
                               **{misure.names[classe]: round(float(misure.box.ap50[posizione]), 3)
                                  for posizione, classe in enumerate(misure.box.ap_class_index)}}
            print(nome_detector, nome_test, riga[nome_test], flush=True)
        if cartella_negativi:
            falsi = conta_falsi_allarmi(YOLO(percorso_pesi), Path(cartella_negativi), dispositivo=dispositivo)
            per_categoria = {categoria: valori["con_allarme"] for categoria, valori in falsi["per_categoria"].items()}
            riga["falsi_allarmi"] = {**per_categoria, "totale": sum(per_categoria.values()),
                                     "immagini": sum(v["immagini"] for v in falsi["per_categoria"].values())}
            print(nome_detector, "falsi allarmi", riga["falsi_allarmi"], flush=True)
        tabella[nome_detector] = riga
    shutil.rmtree(cartella_temporanea, ignore_errors=True)
    return tabella


def comando_confronta_detector(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pesi", nargs="+", required=True, help="nome=percorso/best.pt")
    parser.add_argument("--test", nargs="+", required=True, help="nome=percorso/data.yaml (classi del progetto)")
    parser.add_argument("--negativi", default=None, help="cartella di `strumenti scarica-negativi-coco`")
    parser.add_argument("--dispositivo", default="cpu")
    parser.add_argument("--uscita", required=True)
    argomenti = parser.parse_args(argomenti)
    coppie = lambda valori: dict(valore.split("=", 1) for valore in valori)
    tabella = confronta_detector(coppie(argomenti.pesi), coppie(argomenti.test), argomenti.negativi,
                                 argomenti.dispositivo)
    Path(argomenti.uscita).write_text(json.dumps(tabella, indent=1, ensure_ascii=False))

# ==============================================================================================================
# riga di comando: python -m sicurezza.visione <comando> [opzioni]
# ==============================================================================================================
COMANDI = {"estrai": comando_estrai, "aggiungi-armi": comando_aggiungi_armi, "unisci-armi": comando_unisci_armi, "converti-mock-attack": comando_converti_mock_attack, "prepara-sohas": comando_prepara_sohas, "addestra-detector": comando_addestra_detector, "falsi-allarmi": comando_falsi_allarmi, "confronta-detector": comando_confronta_detector,
           "ritagli-mani": comando_ritagli_mani, "addestra-mani": comando_addestra_mani, "valuta-mani": comando_valuta_mani}


def main(argomenti: list[str] | None = None) -> None:
    argomenti = sys.argv[1:] if argomenti is None else argomenti
    if not argomenti or argomenti[0] not in COMANDI:
        print("uso: python -m sicurezza.visione {" + "|".join(COMANDI) + "} [opzioni]  (--help dopo il comando)")
        raise SystemExit(2)
    COMANDI[argomenti[0]](argomenti[1:])


if __name__ == "__main__":
    main()
