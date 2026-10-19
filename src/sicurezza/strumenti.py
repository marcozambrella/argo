"""Sicurezza NN — strumenti. File unico (richiesta di Marco: massimo 5 file Python)."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote
import argparse
import csv
import io
import json
import re
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import urllib.request
import zipfile




# ==============================================================================================================
# annota_scippi
# Strumento per annotare a mano l'istante dello scippo nei video chain-snatching (D15).
#
# Apre una pagina nel browser con il video e i tasti:
#     Spazio  play / pausa              ← →   indietro / avanti 0,2 s (con Maiusc: 1 s)
#     , .     un frame indietro/avanti  1 2 3 velocità 0,25× / 0,5× / 1×
#     I       inizio scippo (istante corrente)
#     F       fine scippo  → l'evento viene aggiunto alla lista (si possono segnare più scippi)
#     N       nel video NON si vede uno scippo (verrà escluso)
#     Canc    cancella gli eventi di questo video
#     Invio   salva e passa al prossimo video da annotare
#     P S     video precedente / successivo (senza salvare: per rivedere o correggere quelli già fatti)
# Ogni salvataggio scrive subito il file JSON: si può chiudere e riprendere quando si vuole.
#
# Uso:
#     uv run python -m sicurezza.strumenti annota-scippi
#     (poi apri http://localhost:8770 nel browser)
# ==============================================================================================================

CARTELLA_VIDEO_PREDEFINITA = "data/chain-snatching-cctv-dataset/Chain_Snatching_Videos"
FILE_ANNOTAZIONI_PREDEFINITO = "annotations/chain_snatching/tempi_scippo.json"

PAGINA = """<!doctype html>
<html lang="it"><head><meta charset="utf-8"><title>Annota scippi</title>
<style>
 body { font: 15px system-ui, sans-serif; margin: 16px; background: #111; color: #eee; }
 video { width: 100%; max-height: 70vh; background: #000; }
 .riga { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin: 8px 0; }
 .evento { background: #c33; padding: 2px 8px; border-radius: 4px; }
 .tasto { border: 1px solid #666; padding: 0 6px; border-radius: 3px; font-family: monospace; }
 #stato { color: #9c9; } #avviso { color: #fc6; }
</style></head><body>
<div class="riga"><strong id="nome">…</strong><span id="avanzamento"></span><span id="stato"></span></div>
<video id="video" controls preload="auto"></video>
<div class="riga">tempo <strong id="tempo">0.00</strong> s · velocità <strong id="velocita">1</strong>×
 · inizio segnato: <strong id="inizio">—</strong></div>
<div class="riga" id="eventi"></div>
<div class="riga" id="avviso"></div>
<div class="riga"><span><span class="tasto">Spazio</span> play/pausa</span><span><span class="tasto">← →</span> ±0,2 s
 (<span class="tasto">Maiusc</span> ±1 s)</span><span><span class="tasto">, .</span> frame</span>
 <span><span class="tasto">1 2 3</span> velocità</span><span><span class="tasto">I</span> inizio</span>
 <span><span class="tasto">F</span> fine</span><span><span class="tasto">N</span> nessuno scippo</span>
 <span><span class="tasto">Canc</span> cancella</span><span><span class="tasto">Invio</span> salva e prossimo</span>
 <span><span class="tasto">P S</span> video precedente / successivo</span></div>
<script>
const video = document.getElementById("video");
let elenco = [], annotazioni = {}, indice = 0, inizio = null, eventi = [], nessunoScippo = false;
const testo = (id, valore) => document.getElementById(id).textContent = valore;

async function carica() {
  const risposta = await (await fetch("/elenco")).json();
  elenco = risposta.video; annotazioni = risposta.annotazioni;
  if (risposta.problema) { testo("avviso", risposta.problema); return; }
  const primoDaFare = elenco.findIndex((nome) => !(nome in annotazioni));
  apri(primoDaFare >= 0 ? primoDaFare : 0);
}
function apri(nuovoIndice) {
  indice = nuovoIndice;
  const nome = elenco[indice], gia = annotazioni[nome];
  eventi = gia ? gia.eventi.map((e) => [...e]) : []; nessunoScippo = gia ? gia.nessuno_scippo : false; inizio = null;
  video.src = "/video/" + encodeURIComponent(nome); video.playbackRate = 1;
  video.onerror = () => testo("avviso", "Video non caricato: il disco esterno si è scollegato? Rimontalo e ricarica la pagina.");
  testo("nome", nome); testo("velocita", 1); testo("inizio", "—"); testo("avviso", "");
  testo("avanzamento", `${Object.keys(annotazioni).length}/${elenco.length} annotati · video ${indice + 1}`);
  disegnaEventi();
}
function disegnaEventi() {
  const contenitore = document.getElementById("eventi");
  contenitore.replaceChildren(...eventi.map(([a, b]) => Object.assign(document.createElement("span"),
    { className: "evento", textContent: `scippo ${a.toFixed(2)}–${b.toFixed(2)} s` })));
  if (nessunoScippo) contenitore.append("NESSUNO SCIPPO VISIBILE");
}
async function salvaEProssimo() {
  if (!eventi.length && !nessunoScippo) { testo("avviso", "Segna almeno uno scippo (I poi F) oppure premi N."); return; }
  const nome = elenco[indice];
  await fetch("/salva", { method: "POST", body: JSON.stringify({ video: nome, eventi, nessuno_scippo: nessunoScippo }) });
  annotazioni[nome] = { eventi, nessuno_scippo: nessunoScippo };
  testo("stato", "salvato " + nome);
  const prossimo = elenco.findIndex((n, i) => i > indice && !(n in annotazioni));
  const qualsiasi = elenco.findIndex((n) => !(n in annotazioni));
  if (prossimo >= 0) apri(prossimo); else if (qualsiasi >= 0) apri(qualsiasi);
  else testo("avviso", "Finito: tutti i video sono annotati. Puoi chiudere la pagina.");
}
video.addEventListener("timeupdate", () => testo("tempo", video.currentTime.toFixed(2)));
document.addEventListener("keydown", (evento) => {
  const passo = evento.shiftKey ? 1 : 0.2, frame = 1 / 30;
  const azioni = {
    " ": () => (video.paused ? video.play() : video.pause()),
    "ArrowLeft": () => (video.currentTime = Math.max(0, video.currentTime - passo)),
    "ArrowRight": () => (video.currentTime += passo),
    ",": () => { video.pause(); video.currentTime = Math.max(0, video.currentTime - frame); },
    ".": () => { video.pause(); video.currentTime += frame; },
    "1": () => { video.playbackRate = 0.25; testo("velocita", 0.25); },
    "2": () => { video.playbackRate = 0.5; testo("velocita", 0.5); },
    "3": () => { video.playbackRate = 1; testo("velocita", 1); },
    "i": () => { inizio = video.currentTime; testo("inizio", inizio.toFixed(2)); nessunoScippo = false; },
    "f": () => {
      if (inizio === null || video.currentTime <= inizio) { testo("avviso", "Prima premi I su un istante precedente."); return; }
      eventi.push([inizio, video.currentTime]); inizio = null; testo("inizio", "—"); testo("avviso", ""); disegnaEventi();
    },
    "n": () => { nessunoScippo = true; eventi = []; disegnaEventi(); },
    "Delete": () => { eventi = []; nessunoScippo = false; disegnaEventi(); },
    "Enter": salvaEProssimo,
    "p": () => { if (indice > 0) apri(indice - 1); },
    "s": () => { if (indice < elenco.length - 1) apri(indice + 1); },
  };
  const azione = azioni[evento.key] || azioni[evento.key.toLowerCase()];
  if (azione) { evento.preventDefault(); azione(); }
});
carica();
</script></body></html>
"""


def crea_gestore(cartella_video: Path, file_annotazioni: Path):
    blocco_scrittura = threading.Lock()

    def leggi_annotazioni() -> dict:
        return json.loads(file_annotazioni.read_text()) if file_annotazioni.exists() else {}

    class Gestore(BaseHTTPRequestHandler):
        def log_message(self, *argomenti):  # niente log per ogni richiesta di pezzi di video
            pass

        def rispondi(self, codice: int, contenuto: bytes, tipo: str, intestazioni: dict | None = None):
            self.send_response(codice)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(contenuto)))
            for chiave, valore in (intestazioni or {}).items():
                self.send_header(chiave, valore)
            self.end_headers()
            try:
                self.wfile.write(contenuto)
            except (BrokenPipeError, ConnectionResetError):  # il browser interrompe i pezzi di video che non servono
                pass

        def do_GET(self):
            if self.path == "/":
                return self.rispondi(200, PAGINA.encode(), "text/html; charset=utf-8")
            if self.path == "/elenco":
                video = sorted(p.name for p in cartella_video.glob("*.mp4"))
                problema = "" if video else (f"Nessun video in {cartella_video}: il disco esterno è montato? "
                                             "(udisksctl mount -b /dev/sda1, poi ricarica la pagina)")
                corpo = json.dumps({"video": video, "annotazioni": leggi_annotazioni(), "problema": problema}).encode()
                return self.rispondi(200, corpo, "application/json")
            if self.path.startswith("/video/"):
                percorso = cartella_video / Path(unquote(self.path[len("/video/"):])).name
                if not percorso.exists():
                    return self.rispondi(404, b"", "text/plain")
                return self.invia_video(percorso)
            self.rispondi(404, b"", "text/plain")

        def invia_video(self, percorso: Path):
            """Supporta le richieste Range: senza, il browser non permette di spostarsi nel video."""
            dimensione = percorso.stat().st_size
            intervallo = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if not intervallo:
                return self.rispondi(200, percorso.read_bytes(), "video/mp4", {"Accept-Ranges": "bytes"})
            primo = int(intervallo.group(1) or 0)
            ultimo = min(int(intervallo.group(2) or dimensione - 1), dimensione - 1)
            with open(percorso, "rb") as file_video:
                file_video.seek(primo)
                pezzo = file_video.read(ultimo - primo + 1)
            self.rispondi(206, pezzo, "video/mp4", {"Accept-Ranges": "bytes",
                                                     "Content-Range": f"bytes {primo}-{ultimo}/{dimensione}"})

        def do_POST(self):
            if self.path != "/salva":
                return self.rispondi(404, b"", "text/plain")
            dati = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with blocco_scrittura:
                annotazioni = leggi_annotazioni()
                annotazioni[Path(dati["video"]).name] = {
                    "eventi": [[round(float(a), 2), round(float(b), 2)] for a, b in dati["eventi"]],
                    "nessuno_scippo": bool(dati["nessuno_scippo"])}
                file_annotazioni.parent.mkdir(parents=True, exist_ok=True)
                righe = [f"{json.dumps(nome)}: {json.dumps(valore)}" for nome, valore in sorted(annotazioni.items())]
                file_annotazioni.write_text("{\n" + ",\n".join(righe) + "\n}\n")  # una riga per video
            self.rispondi(200, b"{}", "application/json")

    return Gestore


def comando_annota_scippi(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=CARTELLA_VIDEO_PREDEFINITA)
    parser.add_argument("--uscita", default=FILE_ANNOTAZIONI_PREDEFINITO)
    parser.add_argument("--porta", type=int, default=8770)
    argomenti = parser.parse_args(argomenti)
    server = ThreadingHTTPServer(("127.0.0.1", argomenti.porta),
                                 crea_gestore(Path(argomenti.video), Path(argomenti.uscita)))
    print(f"Apri http://localhost:{argomenti.porta} — annotazioni in {argomenti.uscita} (Ctrl+C per chiudere)")
    server.serve_forever()


# ==============================================================================================================
# scarica_campioni
# Scarica pochi video veri dai dataset Kaggle per provare la catena in locale (senza login: API pubblica).
#
# Kaggle a volte consegna i file grandi dentro uno zip con lo stesso nome: qui vengono estratti.
# Uso:  uv run python -m sicurezza.strumenti scarica-campioni [--cartella data/campioni]
#       uv run python -m sicurezza.strumenti scarica-campioni --dataset-intero aniketsahu09/chain-snatching-cctv-dataset
# ==============================================================================================================

API = "https://www.kaggle.com/api/v1/datasets"
CAMPIONI = {  # dataset → file da scaricare (percorsi interni al dataset)
    "vulamnguyen/rwf2000": ["RWF-2000/val/Fight/0Ow4cotKOuw_0.avi", "RWF-2000/val/Fight/0Ow4cotKOuw_1.avi",
                            "RWF-2000/val/NonFight/10uSOcwS_0.avi", "RWF-2000/val/NonFight/1AURh0Wj_0.avi"],
    "mohamedtariki/vsd-cctv-fights": ["groundtruth.json", "original_format-001-200/fight_0001.webm",
                                      "original_format-001-200/fight_0002.mp4"],
    "toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd": [
        "SCVD/SCVD_converted/Test/Normal/t_n001_converted.avi", "SCVD/SCVD_converted/Test/Violence/t_v001_converted.avi",
        "SCVD/SCVD_converted/Test/Weaponized/t_w001_converted.avi"],
}


def scarica_file(riferimento_dataset: str, percorso_interno: str, destinazione: Path) -> None:
    url = f"{API}/download/{riferimento_dataset}/{urllib.parse.quote(percorso_interno, safe='')}"
    contenuto = urllib.request.urlopen(url, timeout=300).read()
    if contenuto[:4] == b"PK\x03\x04" and not percorso_interno.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(contenuto)) as archivio:
            contenuto = archivio.read(archivio.namelist()[0])
    destinazione.parent.mkdir(parents=True, exist_ok=True)
    destinazione.write_bytes(contenuto)


def elenca_file_dataset(riferimento_dataset: str) -> list[str]:
    nomi, segnalibro = [], None
    while True:
        url = f"{API}/list/{riferimento_dataset}?pageSize=200"
        if segnalibro:
            url += f"&pageToken={urllib.parse.quote(segnalibro)}"
        risposta = json.load(urllib.request.urlopen(url, timeout=60))
        nomi += [file["name"] for file in risposta.get("datasetFiles", [])]
        segnalibro = risposta.get("nextPageTokenNullable")
        if not segnalibro:
            return nomi


def comando_scarica_campioni(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartella", default="data/campioni")
    parser.add_argument("--dataset-intero", default=None, help="scarica tutti i file di questo dataset Kaggle")
    argomenti = parser.parse_args(argomenti)
    cartella = Path(argomenti.cartella)
    campioni = CAMPIONI
    if argomenti.dataset_intero:
        campioni = {argomenti.dataset_intero: elenca_file_dataset(argomenti.dataset_intero)}
    for riferimento_dataset, percorsi in campioni.items():
        for percorso_interno in percorsi:
            destinazione = cartella / riferimento_dataset.split("/")[1] / percorso_interno
            if destinazione.exists():
                continue
            scarica_file(riferimento_dataset, percorso_interno, destinazione)
            print(f"{destinazione} ({destinazione.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps({"cartella": str(cartella)}))


# ==============================================================================================================
# scarica_negativi_coco
# Immagini di prova SENZA armi impugnate per `visione falsi-allarmi`: coltelli da cucina, telefoni, forbici, ombrelli
# dal validation set di COCO 2017. Per categoria le prime N in ordine di id (sempre le stesse), e ogni immagine in una
# sola categoria. Servono solo per la verifica in locale: non vanno nel repo (foto Flickr con licenze varie).
#
# Uso:  uv run python -m sicurezza.strumenti scarica-negativi-coco [--cartella data/negativi_coco] [--per-categoria 100]
# ==============================================================================================================

URL_ANNOTAZIONI_COCO = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
URL_IMMAGINI_COCO = "http://images.cocodataset.org/val2017/"
CATEGORIE_NEGATIVE_COCO = ("knife", "cell phone", "scissors", "umbrella")
# D37: negativi difficili per ADDESTRARE il detector v3 (oggetti impugnati o allungati come un'arma). Si prendono
# immagini diverse da quelle della prova (--escludi-prova), che resta così una misura pulita.
CATEGORIE_NEGATIVE_ADDESTRAMENTO = ("knife", "cell phone", "scissors", "umbrella", "remote", "fork",
                                    "baseball bat", "hair drier", "toothbrush")


def scegli_immagini_coco(annotazioni: dict, categorie=CATEGORIE_NEGATIVE_COCO, per_categoria: int = 100,
                         escludi: set[str] = frozenset()) -> dict[str, list[str]]:
    id_categoria = {categoria["name"]: categoria["id"] for categoria in annotazioni["categories"]}
    immagini_con_categoria: dict[int, set[int]] = {}
    for oggetto in annotazioni["annotations"]:
        immagini_con_categoria.setdefault(oggetto["category_id"], set()).add(oggetto["image_id"])
    nome_file = {immagine["id"]: immagine["file_name"] for immagine in annotazioni["images"]}
    gia_prese, scelte = {id_immagine for id_immagine, nome in nome_file.items() if nome in escludi}, {}
    for categoria in categorie:
        candidate = sorted(immagini_con_categoria.get(id_categoria[categoria], set()) - gia_prese)[:per_categoria]
        gia_prese.update(candidate)
        scelte[categoria] = [nome_file[id_immagine] for id_immagine in candidate]
    return scelte


def comando_scarica_negativi_coco(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartella", default="data/negativi_coco")
    parser.add_argument("--per-categoria", type=int, default=100)
    parser.add_argument("--per-addestramento", action="store_true",
                        help="categorie di CATEGORIE_NEGATIVE_ADDESTRAMENTO, senza le immagini della prova, salvate "
                             "come export YOLO (train/images + data.yaml senza classi) per `visione unisci-armi`")
    argomenti = parser.parse_args(argomenti)
    cartella = Path(argomenti.cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    file_annotazioni = cartella / "instances_val2017.json"
    if not file_annotazioni.exists():  # lo zip (241 MB) serve solo per estrarre questo file (25 MB)
        archivio_zip = cartella / "annotations_trainval2017.zip"
        urllib.request.urlretrieve(URL_ANNOTAZIONI_COCO, archivio_zip)
        with zipfile.ZipFile(archivio_zip) as archivio:
            file_annotazioni.write_bytes(archivio.read("annotations/instances_val2017.json"))
        archivio_zip.unlink()
    annotazioni = json.loads(file_annotazioni.read_text())
    if argomenti.per_addestramento:
        prova = {nome for nomi in scegli_immagini_coco(annotazioni).values() for nome in nomi}
        scelte = scegli_immagini_coco(annotazioni, CATEGORIE_NEGATIVE_ADDESTRAMENTO, argomenti.per_categoria, prova)
        (cartella / "data.yaml").write_text("train: ../train/images\nnames: []\n")
    else:
        scelte = scegli_immagini_coco(annotazioni, per_categoria=argomenti.per_categoria)
    for categoria, nomi_file in scelte.items():
        sottocartella = cartella / ("train/images" if argomenti.per_addestramento else categoria.replace(" ", "_"))
        sottocartella.mkdir(parents=True, exist_ok=True)
        for nome in nomi_file:
            if not (sottocartella / nome).exists():
                urllib.request.urlretrieve(URL_IMMAGINI_COCO + nome, sottocartella / nome)
        print(f"{categoria:12s} {len(nomi_file)} immagini in {sottocartella}")


# ==============================================================================================================
# esporta_risultati_sito
# Dai risultati.json dell'addestramento (uno per modello) e dalla taratura del WARNING ARMA (D35) al file che il
# sito legge: site/dati/risultati.json. Solo numeri misurati sul test set: niente da inventare (regola del sito).
#
# Uso:  uv run python -m sicurezza.strumenti esporta-risultati-sito \
#           --risultati runs/addestramento/graph_transformer/risultati.json runs/addestramento/stgcn/risultati.json \
#           [--taratura runs/addestramento/taratura_warning.json] [--uscita site/dati/risultati.json]
# ==============================================================================================================

NOMI_MODELLI_SITO = {"graph_transformer": "Graph-Transformer", "stgcn": "ST-GCN"}


def risultati_per_il_sito(risultati_modelli: list[dict], taratura_warning: dict | None = None) -> dict:
    modelli = []
    for risultati in risultati_modelli:
        test = risultati["test"]
        modelli.append({
            "nome": NOMI_MODELLI_SITO.get(risultati["modello"], risultati["modello"]),
            "macro_f1": round(test["macro_f1"], 4), "accuratezza": round(test["accuratezza"], 4),
            "latenza_cpu_ms": round(risultati["latenza_cpu_ms_per_finestra"], 2),
            "parametri": risultati["parametri"], "epoca_migliore": risultati["epoca_migliore"],
            "recall": test["recall"], "precisione": test["precisione"], "f1": test["f1"],
            "matrice_confusione": test["matrice_confusione"],
            "finestre_test": int(sum(map(sum, test["matrice_confusione"]))),
            "macro_f1_val_per_epoca": [round(epoca["macro_f1_val"], 4) for epoca in risultati["storia"]],
        })
    uscita = {"_nota": "Generato da `strumenti esporta-risultati-sito`: metriche sul test set (split per video, P4).",
              "classi": list(modelli[0]["recall"]) if modelli else [], "modelli": modelli}
    if taratura_warning:
        uscita["warning_arma"] = taratura_warning["migliore"]
    return uscita


def comando_esporta_risultati_sito(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risultati", nargs="+", required=True, help="risultati.json di ogni modello")
    parser.add_argument("--taratura", default=None, help="taratura_warning.json (facoltativo)")
    parser.add_argument("--uscita", default="site/dati/risultati.json")
    argomenti = parser.parse_args(argomenti)
    taratura = json.loads(Path(argomenti.taratura).read_text()) if argomenti.taratura else None
    uscita = risultati_per_il_sito([json.loads(Path(p).read_text()) for p in argomenti.risultati], taratura)
    Path(argomenti.uscita).write_text(json.dumps(uscita, indent=1, ensure_ascii=False))
    for modello in uscita["modelli"]:
        print(f"{modello['nome']:18s} macro-F1 test {modello['macro_f1']:.3f}  {modello['latenza_cpu_ms']:.1f} ms/finestra")


# ==============================================================================================================
# presentation/monta_presentazione
# Monta la video presentation da presentation/scaletta.json con ffmpeg (D12). Riproducibile: quando il modello
# migliora si rigenerano i clip annotati e si rilancia questo script.
#
# Uso:  uv run python -m sicurezza.strumenti monta-presentazione [--scaletta presentation/scaletta.json] [--uscita runs/presentazione.mp4]
#
# SEGNAPOSTO DESIGN: font, colori e transizioni sono provvisori (vedi STILE).
# ==============================================================================================================

STILE = {"sfondo": "0x111111", "colore_testo": "white", "dimensione_titolo": 64, "dimensione_sottotitolo": 30}


def ffmpeg(*argomenti: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *argomenti], check=True)


def testo_sicuro(testo: str) -> str:
    return testo.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")


def scena_titolo(scena, larghezza, altezza, fps, uscita: Path) -> None:
    filtro = (f"drawtext=text='{testo_sicuro(scena['testo'])}':fontcolor={STILE['colore_testo']}:"
              f"fontsize={STILE['dimensione_titolo']}:x=(w-text_w)/2:y=h/2-80")
    if scena.get("sottotitolo"):
        filtro += (f",drawtext=text='{testo_sicuro(scena['sottotitolo'])}':fontcolor={STILE['colore_testo']}:"
                   f"fontsize={STILE['dimensione_sottotitolo']}:x=(w-text_w)/2:y=h/2+20")
    ffmpeg("-f", "lavfi", "-i", f"color=c={STILE['sfondo']}:s={larghezza}x{altezza}:r={fps}:d={scena['secondi']}",
           "-vf", filtro, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(uscita))


def scena_clip(scena, larghezza, altezza, fps, uscita: Path) -> None:
    filtro = (f"scale={larghezza}:{altezza}:force_original_aspect_ratio=decrease,"
              f"pad={larghezza}:{altezza}:(ow-iw)/2:(oh-ih)/2:color={STILE['sfondo']},fps={fps}")
    if scena.get("didascalia"):
        filtro += (f",drawtext=text='{testo_sicuro(scena['didascalia'])}':fontcolor=white:fontsize=26:"
                   f"box=1:boxcolor=black@0.5:x=(w-text_w)/2:y=h-60")
    ffmpeg("-i", scena["percorso"], "-vf", filtro, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(uscita))


def scena_grafico_probabilita(scena, larghezza, altezza, fps, uscita: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(scena["percorso_csv"]) as file_csv:
        righe = list(csv.DictReader(file_csv))
    classi = [nome for nome in righe[0] if nome not in {"inizio_secondi", "fine_secondi", "classe", "warning_arma"}]
    tempi = [float(riga["fine_secondi"]) for riga in righe]
    figura, asse = plt.subplots(figsize=(larghezza / 100, altezza / 100), dpi=100)
    for classe in classi:  # SEGNAPOSTO DESIGN: colori di matplotlib
        asse.plot(tempi, [float(riga[classe]) for riga in righe], label=classe, linewidth=2)
    asse.set_xlabel("secondi")
    asse.set_ylabel("probabilità")
    asse.set_ylim(0, 1)
    asse.legend(loc="upper right")
    figura.tight_layout()
    immagine = uscita.with_suffix(".png")
    figura.savefig(immagine)
    plt.close(figura)
    ffmpeg("-loop", "1", "-i", str(immagine), "-t", str(scena["secondi"]), "-r", str(fps),
           "-vf", f"scale={larghezza}:{altezza}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(uscita))


SCENE = {"titolo": scena_titolo, "clip": scena_clip, "grafico_probabilita": scena_grafico_probabilita}


def monta(percorso_scaletta: Path, percorso_uscita: Path) -> None:
    scaletta = json.loads(percorso_scaletta.read_text())
    larghezza, altezza = scaletta["risoluzione"]
    with tempfile.TemporaryDirectory() as cartella_temporanea:
        pezzi = []
        for numero, scena in enumerate(scaletta["scene"]):
            pezzo = Path(cartella_temporanea) / f"{numero:03d}.mp4"
            percorso_necessario = scena.get("percorso") or scena.get("percorso_csv")
            if percorso_necessario and not Path(percorso_necessario).exists():
                print(f"salto la scena {numero} ({scena['tipo']}): manca {percorso_necessario}")
                continue
            SCENE[scena["tipo"]](scena, larghezza, altezza, scaletta["fps"], pezzo)
            pezzi.append(pezzo)
        elenco = Path(cartella_temporanea) / "elenco.txt"
        elenco.write_text("".join(f"file '{pezzo}'\n" for pezzo in pezzi))
        percorso_uscita.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg("-f", "concat", "-safe", "0", "-i", str(elenco), "-c", "copy", str(percorso_uscita))
    print(f"presentazione: {percorso_uscita} ({len(pezzi)} scene)")

def comando_monta_presentazione(argomenti: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Monta la video presentation da presentation/scaletta.json")
    parser.add_argument("--scaletta", default="presentation/scaletta.json")
    parser.add_argument("--uscita", default="runs/presentazione.mp4")
    argomenti = parser.parse_args(argomenti)
    monta(Path(argomenti.scaletta), Path(argomenti.uscita))


# ==============================================================================================================
# riga di comando: python -m sicurezza.strumenti <comando> [opzioni]
# ==============================================================================================================
COMANDI = {"annota-scippi": comando_annota_scippi, "scarica-campioni": comando_scarica_campioni, "scarica-negativi-coco": comando_scarica_negativi_coco, "esporta-risultati-sito": comando_esporta_risultati_sito, "monta-presentazione": comando_monta_presentazione}


def main(argomenti: list[str] | None = None) -> None:
    argomenti = sys.argv[1:] if argomenti is None else argomenti
    if not argomenti or argomenti[0] not in COMANDI:
        print("uso: python -m sicurezza.strumenti {" + "|".join(COMANDI) + "} [opzioni]  (--help dopo il comando)")
        raise SystemExit(2)
    COMANDI[argomenti[0]](argomenti[1:])


if __name__ == "__main__":
    main()
