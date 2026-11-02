"""Sicurezza NN — strumenti. File unico (richiesta di Marco: massimo 5 file Python)."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote
import argparse
import csv
import io
import json
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
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
#     X       video NON IDONEO (non è una ripresa da telecamera fissa, montaggio, ...): tolto anche dall'estrazione
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
 <span><span class="tasto">X</span> video non idoneo</span>
 <span><span class="tasto">Canc</span> cancella</span><span><span class="tasto">Invio</span> salva e prossimo</span>
 <span><span class="tasto">P S</span> video precedente / successivo</span></div>
<script>
const video = document.getElementById("video");
let elenco = [], annotazioni = {}, indice = 0, inizio = null, eventi = [], nessunoScippo = false, nonIdoneo = false;
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
  nonIdoneo = gia ? Boolean(gia.non_idoneo) : false;
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
  if (nonIdoneo) contenitore.append("VIDEO NON IDONEO");
}
async function salvaEProssimo() {
  if (!eventi.length && !nessunoScippo && !nonIdoneo) {
    testo("avviso", "Segna almeno uno scippo (I poi F), oppure premi N o X."); return;
  }
  const nome = elenco[indice];
  const valore = { eventi, nessuno_scippo: nessunoScippo, non_idoneo: nonIdoneo };
  await fetch("/salva", { method: "POST", body: JSON.stringify({ video: nome, ...valore }) });
  annotazioni[nome] = valore;
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
    "i": () => { inizio = video.currentTime; testo("inizio", inizio.toFixed(2)); nessunoScippo = false; nonIdoneo = false; },
    "f": () => {
      if (inizio === null || video.currentTime <= inizio) { testo("avviso", "Prima premi I su un istante precedente."); return; }
      eventi.push([inizio, video.currentTime]); inizio = null; testo("inizio", "—"); testo("avviso", ""); disegnaEventi();
    },
    "n": () => { nessunoScippo = true; nonIdoneo = false; eventi = []; disegnaEventi(); },
    "x": () => { nonIdoneo = true; nessunoScippo = false; eventi = []; disegnaEventi(); },
    "Delete": () => { eventi = []; nessunoScippo = false; nonIdoneo = false; disegnaEventi(); },
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


class GestoreAnnotazioni(BaseHTTPRequestHandler):
    """Il server della pagina di annotazione. cartella_video e file_annotazioni li assegna `crea_gestore`."""
    cartella_video = None
    file_annotazioni = None
    blocco_scrittura = None   # due salvataggi insieme non devono mescolare il file

    def log_message(self, *argomenti):  # niente log per ogni richiesta di pezzi di video
        pass

    def leggi_annotazioni(self) -> dict:
        if not self.file_annotazioni.exists():
            return {}
        return json.loads(self.file_annotazioni.read_text())

    def rispondi(self, codice: int, contenuto: bytes, tipo: str, intestazioni=None):
        self.send_response(codice)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(contenuto)))
        if intestazioni is not None:
            for chiave, valore in intestazioni.items():
                self.send_header(chiave, valore)
        self.end_headers()
        try:
            self.wfile.write(contenuto)
        except (BrokenPipeError, ConnectionResetError):  # il browser interrompe i pezzi di video che non servono
            pass

    def do_GET(self):
        if self.path == "/":
            self.rispondi(200, PAGINA.encode(), "text/html; charset=utf-8")
        elif self.path == "/elenco":
            video = []
            for percorso in self.cartella_video.glob("*.mp4"):
                video.append(percorso.name)
            video.sort()
            problema = ""
            if len(video) == 0:
                problema = (f"Nessun video in {self.cartella_video}: il disco esterno è montato? "
                            "(udisksctl mount -b /dev/sda1, poi ricarica la pagina)")
            corpo = json.dumps({"video": video, "annotazioni": self.leggi_annotazioni(), "problema": problema})
            self.rispondi(200, corpo.encode(), "application/json")
        elif self.path.startswith("/video/"):
            nome = Path(unquote(self.path[len("/video/"):])).name   # solo il nome: niente percorsi dal browser
            percorso = self.cartella_video / nome
            if percorso.exists():
                self.invia_video(percorso)
            else:
                self.rispondi(404, b"", "text/plain")
        else:
            self.rispondi(404, b"", "text/plain")

    def invia_video(self, percorso: Path):
        """Supporta le richieste Range: senza, il browser non permette di spostarsi nel video."""
        dimensione = percorso.stat().st_size
        intervallo = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if not intervallo:
            self.rispondi(200, percorso.read_bytes(), "video/mp4", {"Accept-Ranges": "bytes"})
            return
        primo = 0
        if intervallo.group(1):
            primo = int(intervallo.group(1))
        ultimo = dimensione - 1
        if intervallo.group(2):
            ultimo = min(int(intervallo.group(2)), dimensione - 1)
        with open(percorso, "rb") as file_video:
            file_video.seek(primo)
            pezzo = file_video.read(ultimo - primo + 1)
        self.rispondi(206, pezzo, "video/mp4", {"Accept-Ranges": "bytes",
                                                 "Content-Range": f"bytes {primo}-{ultimo}/{dimensione}"})

    def do_POST(self):
        if self.path != "/salva":
            self.rispondi(404, b"", "text/plain")
            return
        dati = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        eventi = []
        for inizio, fine in dati["eventi"]:
            eventi.append([round(float(inizio), 2), round(float(fine), 2)])
        with self.blocco_scrittura:
            annotazioni = self.leggi_annotazioni()
            valore = {"eventi": eventi, "nessuno_scippo": bool(dati["nessuno_scippo"])}
            if dati.get("non_idoneo"):   # i file vecchi (chain-snatching) non hanno questo campo
                valore["non_idoneo"] = True
            annotazioni[Path(dati["video"]).name] = valore
            self.file_annotazioni.parent.mkdir(parents=True, exist_ok=True)
            righe = []   # una riga per video: il file si legge bene anche a occhio
            for nome, valore in sorted(annotazioni.items()):
                righe.append(f"{json.dumps(nome)}: {json.dumps(valore)}")
            self.file_annotazioni.write_text("{\n" + ",\n".join(righe) + "\n}\n")
        self.rispondi(200, b"{}", "application/json")


def crea_gestore(cartella_video: Path, file_annotazioni: Path, classe_base=None):
    # una sottoclasse con le sue cartelle: così si possono avere più server con cartelle diverse
    if classe_base is None:
        classe_base = GestoreAnnotazioni

    class GestoreDiQuesteCartelle(classe_base):
        pass
    GestoreDiQuesteCartelle.cartella_video = cartella_video
    GestoreDiQuesteCartelle.file_annotazioni = file_annotazioni
    GestoreDiQuesteCartelle.blocco_scrittura = threading.Lock()
    return GestoreDiQuesteCartelle


def comando_annota_scippi(argomenti=None) -> None:
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
# annota_tipo_arma
# D68: SCVD non dice che arma c'è nelle clip "Weaponized"; per misurare armi da fuoco e coltelli separati il tipo
# lo segna Marco guardando le clip (convertite in mp4 dal kernel experiments_and_future_work/tipo_arma_scvd).
# Stesso server di annota-scippi, pagina più semplice: un tasto per clip, salva e passa alla prossima.
#     F  arma da fuoco    C  coltello    E  entrambi    X  non si capisce (la clip esce dall'addestramento)
#     Spazio play/pausa   ← →  ±1 s   1 2 3 velocità   P S  clip precedente / successiva
# Uso:
#     uv run python -m sicurezza.strumenti annota-tipo-arma --video <cartella video/ del kernel>
#     (poi apri http://localhost:8771)
# ==============================================================================================================

FILE_TIPO_ARMA_PREDEFINITO = "annotations/armi/tipo_arma_scvd.json"

PAGINA_TIPO_ARMA = """<!doctype html>
<html lang="it"><head><meta charset="utf-8"><title>Tipo d'arma</title>
<style>
 body { font: 15px system-ui, sans-serif; margin: 16px; background: #111; color: #eee; }
 video { width: 100%; max-height: 75vh; background: #000; }
 .riga { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin: 8px 0; }
 .tasto { border: 1px solid #666; padding: 0 6px; border-radius: 3px; font-family: monospace; }
 #stato { color: #9c9; } #gia { color: #fc6; }
</style></head><body>
<div class="riga"><strong id="nome">…</strong><span id="avanzamento"></span><span id="gia"></span>
 <span id="stato"></span></div>
<video id="video" controls loop autoplay muted></video>
<div class="riga"><span><span class="tasto">F</span> arma da fuoco</span><span><span class="tasto">C</span> coltello</span>
 <span><span class="tasto">E</span> entrambi</span><span><span class="tasto">X</span> non si capisce (esclusa)</span>
 <span><span class="tasto">Spazio</span> play/pausa</span><span><span class="tasto">← →</span> ±1 s</span>
 <span><span class="tasto">1 2 3</span> velocità</span><span><span class="tasto">P S</span> precedente / successiva</span></div>
<script>
const video = document.getElementById("video");
let elenco = [], annotazioni = {}, indice = 0;
const testo = (id, valore) => document.getElementById(id).textContent = valore;

async function carica() {
  const risposta = await (await fetch("/elenco")).json();
  elenco = risposta.video; annotazioni = risposta.annotazioni;
  if (risposta.problema) { testo("stato", risposta.problema); return; }
  const primoDaFare = elenco.findIndex((nome) => !(nome in annotazioni));
  apri(primoDaFare >= 0 ? primoDaFare : 0);
}
function apri(nuovoIndice) {
  indice = nuovoIndice;
  const nome = elenco[indice];
  video.src = "/video/" + encodeURIComponent(nome); video.playbackRate = 1;
  testo("nome", nome);
  testo("avanzamento", `${Object.keys(annotazioni).length}/${elenco.length} segnate · clip ${indice + 1}`);
  if (nome in annotazioni) testo("gia", "già segnata: " + (annotazioni[nome].join(" + ") || "non si capisce"));
  else testo("gia", "");
}
async function segna(tipi) {
  const nome = elenco[indice];
  await fetch("/salva", { method: "POST", body: JSON.stringify({ video: nome, tipi }) });
  annotazioni[nome] = tipi;
  testo("stato", "salvata " + nome);
  const prossimo = elenco.findIndex((n, i) => i > indice && !(n in annotazioni));
  const qualsiasi = elenco.findIndex((n) => !(n in annotazioni));
  if (prossimo >= 0) apri(prossimo); else if (qualsiasi >= 0) apri(qualsiasi);
  else testo("stato", "Finito: tutte le clip sono segnate. Puoi chiudere la pagina.");
}
document.addEventListener("keydown", (evento) => {
  const azioni = {
    "f": () => segna(["fuoco"]),
    "c": () => segna(["coltello"]),
    "e": () => segna(["coltello", "fuoco"]),
    "x": () => segna([]),
    " ": () => (video.paused ? video.play() : video.pause()),
    "ArrowLeft": () => (video.currentTime = Math.max(0, video.currentTime - 1)),
    "ArrowRight": () => (video.currentTime += 1),
    "1": () => { video.playbackRate = 0.25; },
    "2": () => { video.playbackRate = 0.5; },
    "3": () => { video.playbackRate = 1; },
    "p": () => { if (indice > 0) apri(indice - 1); },
    "s": () => { if (indice < elenco.length - 1) apri(indice + 1); },
  };
  const azione = azioni[evento.key] || azioni[evento.key.toLowerCase()];
  if (azione) { evento.preventDefault(); azione(); }
});
carica();
</script></body></html>
"""


class GestoreTipoArma(GestoreAnnotazioni):
    """Come GestoreAnnotazioni (elenco e video uguali), ma con la sua pagina e il suo salvataggio."""

    def do_GET(self):
        if self.path == "/":
            self.rispondi(200, PAGINA_TIPO_ARMA.encode(), "text/html; charset=utf-8")
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != "/salva":
            self.rispondi(404, b"", "text/plain")
            return
        dati = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tipi = []
        for tipo in dati["tipi"]:
            if tipo in ("fuoco", "coltello"):
                tipi.append(tipo)
        with self.blocco_scrittura:
            annotazioni = self.leggi_annotazioni()
            annotazioni[Path(dati["video"]).name] = sorted(tipi)
            self.file_annotazioni.parent.mkdir(parents=True, exist_ok=True)
            righe = []   # una riga per clip, come i tempi degli scippi
            for nome, valore in sorted(annotazioni.items()):
                righe.append(f"{json.dumps(nome)}: {json.dumps(valore)}")
            self.file_annotazioni.write_text("{\n" + ",\n".join(righe) + "\n}\n")
        self.rispondi(200, b"{}", "application/json")


def comando_annota_tipo_arma(argomenti=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="cartella video/ scaricata dal kernel tipo_arma_scvd")
    parser.add_argument("--uscita", default=FILE_TIPO_ARMA_PREDEFINITO)
    parser.add_argument("--porta", type=int, default=8771)
    argomenti = parser.parse_args(argomenti)
    gestore = crea_gestore(Path(argomenti.video), Path(argomenti.uscita), GestoreTipoArma)
    server = ThreadingHTTPServer(("127.0.0.1", argomenti.porta), gestore)
    print(f"Apri http://localhost:{argomenti.porta} — tipi d'arma in {argomenti.uscita} (Ctrl+C per chiudere)")
    server.serve_forever()


# ==============================================================================================================
# annota_anomalie
# D69: uscita "anomalia" della rete video. NWPU Campus, CHAD e ATMA-V dicono solo quali fotogrammi sono anomali,
# non che cosa succede; molte anomalie sono innocue (corsa, bici, palla) e insegnerebbero falsi allarmi. Marco
# guarda gli spezzoni anomali (ritagliati prima, con il margine) e dà un tasto per spezzone:
#     S scippo   F furto senza strappo   V violenza/lite   R con arma   A altra anomalia rilevante
#     I innocua (diventa normale)   X non idoneo (escluso)
#     Spazio play/pausa   ← →  ±1 s   1 2 3 velocità   P N  spezzone precedente / successivo
# Uso:
#     uv run python -m sicurezza.strumenti annota-anomalie --video <cartella degli spezzoni> --uscita <file .json>
#     (poi apri http://localhost:8774)
# ==============================================================================================================

TIPI_ANOMALIA = ("scippo", "furto", "violenza", "arma", "altra", "innocua", "non_idoneo")

PAGINA_ANOMALIE = """<!doctype html>
<html lang="it"><head><meta charset="utf-8"><title>Anomalie</title>
<style>
 body { font: 15px system-ui, sans-serif; margin: 16px; background: #111; color: #eee; }
 video { width: 100%; max-height: 75vh; background: #000; }
 .riga { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin: 8px 0; }
 .tasto { border: 1px solid #666; padding: 0 6px; border-radius: 3px; font-family: monospace; }
 #stato { color: #9c9; } #gia { color: #fc6; }
</style></head><body>
<div class="riga"><strong id="nome">…</strong><span id="avanzamento"></span><span id="gia"></span>
 <span id="stato"></span></div>
<video id="video" controls loop autoplay muted></video>
<div class="riga"><span><span class="tasto">S</span> scippo</span><span><span class="tasto">F</span> furto senza strappo</span>
 <span><span class="tasto">V</span> violenza / lite</span><span><span class="tasto">R</span> con arma</span>
 <span><span class="tasto">A</span> altra anomalia rilevante</span><span><span class="tasto">I</span> innocua (normale)</span>
 <span><span class="tasto">X</span> non idoneo (escluso)</span></div>
<div class="riga"><span><span class="tasto">Spazio</span> play/pausa</span><span><span class="tasto">← →</span> ±1 s</span>
 <span><span class="tasto">1 2 3</span> velocità</span><span><span class="tasto">P N</span> precedente / successivo</span></div>
<script>
const video = document.getElementById("video");
let elenco = [], annotazioni = {}, indice = 0;
const testo = (id, valore) => document.getElementById(id).textContent = valore;

async function carica() {
  const risposta = await (await fetch("/elenco")).json();
  elenco = risposta.video; annotazioni = risposta.annotazioni;
  if (risposta.problema) { testo("stato", risposta.problema); return; }
  const primoDaFare = elenco.findIndex((nome) => !(nome in annotazioni));
  apri(primoDaFare >= 0 ? primoDaFare : 0);
}
function apri(nuovoIndice) {
  indice = nuovoIndice;
  const nome = elenco[indice];
  video.src = "/video/" + encodeURIComponent(nome); video.playbackRate = 1;
  testo("nome", nome);
  testo("avanzamento", `${Object.keys(annotazioni).length}/${elenco.length} segnati · spezzone ${indice + 1}`);
  if (nome in annotazioni) testo("gia", "già segnato: " + annotazioni[nome]);
  else testo("gia", "");
}
let salvando = false;
async function segna(tipo) {
  if (salvando) return;   // un tasto alla volta: finché salva, gli altri tasti non contano
  salvando = true;
  const nome = elenco[indice];
  try {
    await fetch("/salva", { method: "POST", body: JSON.stringify({ video: nome, tipo }) });
  } finally {
    salvando = false;
  }
  annotazioni[nome] = tipo;
  testo("stato", "salvato " + nome);
  const prossimo = elenco.findIndex((n, i) => i > indice && !(n in annotazioni));
  const qualsiasi = elenco.findIndex((n) => !(n in annotazioni));
  if (prossimo >= 0) apri(prossimo); else if (qualsiasi >= 0) apri(qualsiasi);
  else testo("stato", "Finito: tutti gli spezzoni sono segnati. Puoi chiudere la pagina.");
}
document.addEventListener("keydown", (evento) => {
  if (evento.repeat) return;   // tasto tenuto premuto: la ripetizione segnava gli spezzoni dopo senza vederli
  const azioni = {
    "s": () => segna("scippo"),
    "f": () => segna("furto"),
    "v": () => segna("violenza"),
    "r": () => segna("arma"),
    "a": () => segna("altra"),
    "i": () => segna("innocua"),
    "x": () => segna("non_idoneo"),
    " ": () => (video.paused ? video.play() : video.pause()),
    "ArrowLeft": () => (video.currentTime = Math.max(0, video.currentTime - 1)),
    "ArrowRight": () => (video.currentTime += 1),
    "1": () => { video.playbackRate = 0.25; },
    "2": () => { video.playbackRate = 0.5; },
    "3": () => { video.playbackRate = 1; },
    "p": () => { if (indice > 0) apri(indice - 1); },
    "n": () => { if (indice < elenco.length - 1) apri(indice + 1); },
  };
  const azione = azioni[evento.key] || azioni[evento.key.toLowerCase()];
  if (azione) { evento.preventDefault(); azione(); }
});
carica();
</script></body></html>
"""


class GestoreAnomalie(GestoreAnnotazioni):
    """Come GestoreTipoArma, ma salva un solo tipo d'anomalia per spezzone."""

    def do_GET(self):
        if self.path == "/":
            self.rispondi(200, PAGINA_ANOMALIE.encode(), "text/html; charset=utf-8")
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != "/salva":
            self.rispondi(404, b"", "text/plain")
            return
        dati = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if dati["tipo"] not in TIPI_ANOMALIA:
            self.rispondi(400, b"tipo sconosciuto", "text/plain")
            return
        with self.blocco_scrittura:
            annotazioni = self.leggi_annotazioni()
            annotazioni[Path(dati["video"]).name] = dati["tipo"]
            self.file_annotazioni.parent.mkdir(parents=True, exist_ok=True)
            righe = []   # una riga per spezzone
            for nome, valore in sorted(annotazioni.items()):
                righe.append(f"{json.dumps(nome)}: {json.dumps(valore)}")
            self.file_annotazioni.write_text("{\n" + ",\n".join(righe) + "\n}\n")
        self.rispondi(200, b"{}", "application/json")


def comando_annota_anomalie(argomenti=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="cartella degli spezzoni anomali (.mp4)")
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--porta", type=int, default=8774)
    argomenti = parser.parse_args(argomenti)
    gestore = crea_gestore(Path(argomenti.video), Path(argomenti.uscita), GestoreAnomalie)
    server = ThreadingHTTPServer(("127.0.0.1", argomenti.porta), gestore)
    print(f"Apri http://localhost:{argomenti.porta} — anomalie in {argomenti.uscita} (Ctrl+C per chiudere)")
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


def elenca_file_dataset(riferimento_dataset: str) -> list:
    """Tutti i file di un dataset Kaggle: l'API li dà a pagine da 200, il segnalibro dice dove riprendere."""
    nomi = []
    segnalibro = None
    while True:
        url = f"{API}/list/{riferimento_dataset}?pageSize=200"
        if segnalibro:
            url += f"&pageToken={urllib.parse.quote(segnalibro)}"
        risposta = json.load(urllib.request.urlopen(url, timeout=60))
        for file in risposta.get("datasetFiles", []):
            nomi.append(file["name"])
        segnalibro = risposta.get("nextPageTokenNullable")
        if not segnalibro:
            return nomi


def comando_scarica_campioni(argomenti=None) -> None:
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
                         escludi=()) -> dict:
    """Per ogni categoria i nomi dei file scelti; `escludi` = nomi di file da non prendere."""
    id_categoria = {}
    for categoria in annotazioni["categories"]:
        id_categoria[categoria["name"]] = categoria["id"]
    immagini_con_categoria = {}   # id categoria → insieme degli id delle immagini che la contengono
    for oggetto in annotazioni["annotations"]:
        if oggetto["category_id"] not in immagini_con_categoria:
            immagini_con_categoria[oggetto["category_id"]] = set()
        immagini_con_categoria[oggetto["category_id"]].add(oggetto["image_id"])
    nome_file = {}
    for immagine in annotazioni["images"]:
        nome_file[immagine["id"]] = immagine["file_name"]

    gia_prese = set()   # ogni immagine in una sola categoria
    for id_immagine, nome in nome_file.items():
        if nome in escludi:
            gia_prese.add(id_immagine)
    scelte = {}
    for categoria in categorie:
        con_categoria = immagini_con_categoria.get(id_categoria[categoria], set())
        candidate = sorted(con_categoria - gia_prese)[:per_categoria]   # le prime per id: sempre le stesse
        gia_prese.update(candidate)
        scelte[categoria] = []
        for id_immagine in candidate:
            scelte[categoria].append(nome_file[id_immagine])
    return scelte


def scarica_con_riprova(url: str, destinazione: Path, tentativi: int = 3) -> None:
    """urlretrieve con un timeout sul socket e qualche riprova: una connessione bloccata non deve appendere un kernel."""
    for tentativo in range(tentativi):
        try:
            urllib.request.urlretrieve(url, destinazione)
            return
        except OSError:
            if tentativo == tentativi - 1:
                raise


def comando_scarica_negativi_coco(argomenti=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartella", default="data/negativi_coco")
    parser.add_argument("--per-categoria", type=int, default=100)
    parser.add_argument("--per-addestramento", action="store_true",
                        help="categorie di CATEGORIE_NEGATIVE_ADDESTRAMENTO, senza le immagini della prova, salvate "
                             "come export YOLO (train/images + data.yaml senza classi) per `visione unisci-armi`")
    argomenti = parser.parse_args(argomenti)
    socket.setdefaulttimeout(120)
    cartella = Path(argomenti.cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    file_annotazioni = cartella / "instances_val2017.json"
    if not file_annotazioni.exists():  # lo zip (241 MB) serve solo per estrarre questo file (25 MB)
        archivio_zip = cartella / "annotations_trainval2017.zip"
        scarica_con_riprova(URL_ANNOTAZIONI_COCO, archivio_zip)
        with zipfile.ZipFile(archivio_zip) as archivio:
            file_annotazioni.write_bytes(archivio.read("annotations/instances_val2017.json"))
        archivio_zip.unlink()
    annotazioni = json.loads(file_annotazioni.read_text())
    if argomenti.per_addestramento:
        # le immagini della prova non si usano per addestrare
        prova = set()
        for nomi in scegli_immagini_coco(annotazioni).values():
            for nome in nomi:
                prova.add(nome)
        scelte = scegli_immagini_coco(annotazioni, CATEGORIE_NEGATIVE_ADDESTRAMENTO, argomenti.per_categoria, prova)
        (cartella / "data.yaml").write_text("train: ../train/images\nnames: []\n")
    else:
        scelte = scegli_immagini_coco(annotazioni, per_categoria=argomenti.per_categoria)
    for categoria, nomi_file in scelte.items():
        if argomenti.per_addestramento:
            sottocartella = cartella / "train/images"
        else:
            sottocartella = cartella / categoria.replace(" ", "_")
        sottocartella.mkdir(parents=True, exist_ok=True)
        for nome in nomi_file:
            if not (sottocartella / nome).exists():
                scarica_con_riprova(URL_IMMAGINI_COCO + nome, sottocartella / nome)
        print(f"{categoria:12s} {len(nomi_file)} immagini in {sottocartella}")


# ==============================================================================================================
# scarica_video_cc
# D40: video di scippi di telefono con licenza Creative Commons Attribution da YouTube (fonti trovate dal
# sub-agente, D31), da annotare poi con `annota-scippi`. Per ogni video si legge la licenza con yt-dlp e si scarica
# SOLO se è CC BY; nome, canale, licenza e link finiscono in attribuzioni.csv (la CC BY obbliga a citare l'autore).
# Se YouTube chiede di dimostrare di non essere un robot ci si ferma: niente cookie del browser né aggiramenti.
#
# Uso:  uv run python -m sicurezza.strumenti scarica-video-cc --candidati annotations/phone_snatching/candidati_youtube.txt \
#           --cartella data/phone_snatching
# ==============================================================================================================

PAROLE_FURTO = re.compile(r"snatch|chor\b|thief|theft|steal|stolen|robb|jambret|furto|roub|vol de|vole|ladr",
                          re.IGNORECASE)
PAROLE_TELEFONO = re.compile(r"mobile|phone|cell|handphone|celular|t[ée]l[ée]phone|telefono|smartphone",
                             re.IGNORECASE)
PAROLE_DA_SCARTARE = re.compile(r"prank|funny|vines|idiotz|be like|thug|review|first look|unboxing|animation|"
                                r"explainer|salman|#shorts|hack|software|flashing", re.IGNORECASE)
DURATA_MASSIMA_SECONDI = 360   # le compilation lunghe le annota male anche una persona: meglio clip brevi


def scegli_candidati_scippo_telefono(righe: list) -> list:
    """Righe "id | durata | canale | ... | titolo" (formato del sub-agente) → [(id, titolo)] senza doppioni."""
    scelti = []
    visti = set()
    for riga in righe:
        riga = re.sub(r"^\[\d+\]\s*", "", riga)   # toglie un eventuale "[12] " all'inizio
        campi = []
        for campo in riga.split(" | "):
            campi.append(campo.strip())
        if len(campi) < 2 or campi[0] in visti:
            continue
        titolo = campi[-1]
        parla_di_furto_di_telefono = PAROLE_FURTO.search(titolo) and PAROLE_TELEFONO.search(titolo)
        da_scartare = PAROLE_DA_SCARTARE.search(titolo) or "chain" in titolo.lower()
        if parla_di_furto_di_telefono and not da_scartare:
            visti.add(campi[0])
            scelti.append((campi[0], titolo))
    return scelti


def comando_scarica_video_cc(argomenti=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidati", required=True, help="un video per riga: id | ... | titolo")
    parser.add_argument("--cartella", default="data/phone_snatching")
    parser.add_argument("--pausa-secondi", type=float, default=4.0, help="tra un video e l'altro (evita il 429)")
    argomenti = parser.parse_args(argomenti)
    cartella = Path(argomenti.cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    file_attribuzioni = cartella / "attribuzioni.csv"
    gia_visti = set()
    if file_attribuzioni.exists():
        with file_attribuzioni.open() as file_csv:
            for riga in csv.DictReader(file_csv):
                gia_visti.add(riga["id"])
    else:
        file_attribuzioni.write_text("id,esito,licenza,durata_secondi,canale,titolo,url\n")
    candidati = scegli_candidati_scippo_telefono(Path(argomenti.candidati).read_text().splitlines())
    for id_video, titolo in candidati:
        if id_video in gia_visti:
            continue
        url = f"https://www.youtube.com/watch?v={id_video}"
        lettura = subprocess.run(["yt-dlp", "--skip-download", "--dump-json", url], capture_output=True, text=True,
                                 check=False)
        if "not a bot" in lettura.stderr:
            print("YouTube chiede la verifica anti-robot: mi fermo (vanno scaricati a mano).")
            break
        if lettura.returncode != 0:
            esito = "non_disponibile"
            informazioni = {}
        else:
            informazioni = json.loads(lettura.stdout)
            licenza = informazioni.get("license") or ""    # il campo può mancare o valere None
            durata = informazioni.get("duration") or 0
            if "Creative Commons" not in licenza:
                esito = "licenza_non_cc"
            elif durata > DURATA_MASSIMA_SECONDI:
                esito = "troppo_lungo"
            else:
                scaricamento = subprocess.run(["yt-dlp", "-q", "-f",
                                               "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]",
                                               "--merge-output-format", "mp4", "-o", str(cartella / "%(id)s.%(ext)s"),
                                               url], capture_output=True, text=True, check=False)
                if scaricamento.returncode == 0:
                    esito = "scaricato"
                else:
                    esito = "errore_scaricamento"
        with file_attribuzioni.open("a", newline="") as uscita:
            csv.writer(uscita).writerow([id_video, esito, informazioni.get("license", ""),
                                         informazioni.get("duration", ""), informazioni.get("channel", ""),
                                         informazioni.get("title", ""), url])
        print(f"{id_video}: {esito}", flush=True)
        time.sleep(argomenti.pausa_secondi)


# ==============================================================================================================
# esporta_risultati_sito
# Dai risultati.json dell'addestramento (uno per modello) e dalla taratura del WARNING ARMA (D35) al file che il
# sito legge: site/dati/risultati.json. Solo numeri misurati sul test set: niente da inventare (regola del sito).
#
# Uso:  uv run python -m sicurezza.strumenti esporta-risultati-sito \
#           --risultati runs/addestramento/graph_transformer/risultati.json runs/addestramento/stgcn/risultati.json \
#           [--taratura runs/addestramento/taratura_warning.json] [--uscita site/dati/risultati.json] \
#           [--violenza runs/prova_rgb_clip/fusione_allarme.json --sistema "pixel + add1" --confronto "scheletri add1"]
# --violenza (D48): violenza sì/no del sistema consegnato contro i soli scheletri, per finestra e con la regola
# d'allarme sui pezzi da 10 e 30 s (clip intere mai viste, regola scelta sul validation; file di `fusione_allarme`).
# ==============================================================================================================

NOMI_MODELLI_SITO = {"graph_transformer": "Graph-Transformer", "stgcn": "ST-GCN"}


def violenza_per_il_sito(fusione: dict, sistema: str, confronto: str) -> dict:
    """Dal file della fusione si tengono solo le misure che il sito mostra, per il sistema e per i soli scheletri."""
    tieni = ("accuratezza", "recall", "falsi_allarmi", "pezzi", "ic95", "finestre_di_fila", "media_mobile")
    uscita = {}
    for nome, chiave_fusione in (("sistema", sistema), ("solo_scheletri", confronto)):
        uscita[nome] = {}
        for misura in ("per_finestra", "pezzi_10s", "pezzi_30s"):
            uscita[nome][misura] = {}
            for chiave, valore in fusione[chiave_fusione][misura].items():
                if chiave in tieni:
                    uscita[nome][misura][chiave] = valore
    return uscita


def risultati_per_il_sito(risultati_modelli: list, taratura_warning=None, versione: str = "",
                          violenza=None) -> dict:
    modelli = []
    for risultati in risultati_modelli:
        test = risultati["test"]
        finestre_test = 0   # la somma di tutta la matrice di confusione
        for riga in test["matrice_confusione"]:
            finestre_test += sum(riga)
        macro_f1_per_epoca = []
        for epoca in risultati["storia"]:
            macro_f1_per_epoca.append(round(epoca["macro_f1_val"], 4))
        modelli.append({
            "nome": NOMI_MODELLI_SITO.get(risultati["modello"], risultati["modello"]),
            "macro_f1": round(test["macro_f1"], 4), "accuratezza": round(test["accuratezza"], 4),
            "latenza_cpu_ms": round(risultati["latenza_cpu_ms_per_finestra"], 2),
            "parametri": risultati["parametri"], "epoca_migliore": risultati["epoca_migliore"],
            "recall": test["recall"], "precisione": test["precisione"], "f1": test["f1"],
            "matrice_confusione": test["matrice_confusione"],
            "finestre_test": int(finestre_test),
            "macro_f1_val_per_epoca": macro_f1_per_epoca,
        })
    classi = []
    if len(modelli) > 0:
        classi = list(modelli[0]["recall"])   # i nomi delle classi sono le chiavi di "recall"
    uscita = {"_nota": "Generato da `strumenti esporta-risultati-sito`: metriche sul test set (split per video, P4).",
              "versione": versione, "classi": classi, "modelli": modelli}
    if taratura_warning:
        uscita["warning_arma"] = taratura_warning["migliore"]
    if violenza:
        uscita["violenza"] = violenza
    return uscita


def comando_esporta_risultati_sito(argomenti=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risultati", nargs="+", required=True, help="risultati.json di ogni modello")
    parser.add_argument("--taratura", default=None, help="taratura_warning.json (facoltativo)")
    parser.add_argument("--uscita", default="site/dati/risultati.json")
    parser.add_argument("--versione", default="", help='es. "addestramento 1, armi del detector v1"')
    parser.add_argument("--violenza", default=None, help="fusione_allarme.json (D48), facoltativo")
    parser.add_argument("--sistema", default="pixel + add1", help="chiave del sistema consegnato in --violenza")
    parser.add_argument("--confronto", default="scheletri add1", help="chiave dei soli scheletri in --violenza")
    argomenti = parser.parse_args(argomenti)
    taratura = None
    if argomenti.taratura:
        taratura = json.loads(Path(argomenti.taratura).read_text())
    violenza = None
    if argomenti.violenza:
        fusione = json.loads(Path(argomenti.violenza).read_text())
        violenza = violenza_per_il_sito(fusione, argomenti.sistema, argomenti.confronto)
    risultati_modelli = []
    for percorso in argomenti.risultati:
        risultati_modelli.append(json.loads(Path(percorso).read_text()))
    uscita = risultati_per_il_sito(risultati_modelli, taratura, argomenti.versione, violenza)
    Path(argomenti.uscita).write_text(json.dumps(uscita, indent=1, ensure_ascii=False))
    for modello in uscita["modelli"]:
        print(f"{modello['nome']:18s} macro-F1 test {modello['macro_f1']:.3f}  "
              f"{modello['latenza_cpu_ms']:.1f} ms/finestra")


# ==============================================================================================================
# media/presentation/monta_presentazione
# Monta la video presentation da media/presentation/scaletta.json con ffmpeg (D12). Riproducibile: quando il modello
# migliora si rigenerano i clip annotati e si rilancia questo script.
#
# Uso:  uv run python -m sicurezza.strumenti monta-presentazione [--scaletta media/presentation/scaletta.json] \
#           [--uscita runs/presentazione.mp4]
#
# SEGNAPOSTO DESIGN: font, colori e transizioni sono provvisori (vedi STILE).
# ==============================================================================================================

STILE = {"sfondo": "0x111111", "colore_testo": "white", "dimensione_titolo": 64, "dimensione_sottotitolo": 30}


def ffmpeg(argomenti: list) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y"] + argomenti, check=True)


def testo_sicuro(testo: str) -> str:
    # drawtext: ":" separa le opzioni, "%" apre un'espansione (es. %{pts}) → vanno protetti
    return testo.replace("\\", "\\\\").replace(":", "\\:").replace("%", "\\\\%").replace("'", "’")


def scena_titolo(scena, larghezza, altezza, fps, uscita: Path) -> None:
    filtro = (f"drawtext=text='{testo_sicuro(scena['testo'])}':fontcolor={STILE['colore_testo']}:"
              f"fontsize={STILE['dimensione_titolo']}:x=(w-text_w)/2:y=h/2-80")
    if scena.get("sottotitolo"):
        filtro += (f",drawtext=text='{testo_sicuro(scena['sottotitolo'])}':fontcolor={STILE['colore_testo']}:"
                   f"fontsize={STILE['dimensione_sottotitolo']}:x=(w-text_w)/2:y=h/2+20")
    ffmpeg(["-f", "lavfi", "-i", f"color=c={STILE['sfondo']}:s={larghezza}x{altezza}:r={fps}:d={scena['secondi']}",
            "-vf", filtro, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(uscita)])


def scena_clip(scena, larghezza, altezza, fps, uscita: Path) -> None:
    filtro = (f"scale={larghezza}:{altezza}:force_original_aspect_ratio=decrease,"
              f"pad={larghezza}:{altezza}:(ow-iw)/2:(oh-ih)/2:color={STILE['sfondo']},fps={fps}")
    if scena.get("didascalia"):
        filtro += (f",drawtext=text='{testo_sicuro(scena['didascalia'])}':fontcolor=white:fontsize=26:"
                   f"box=1:boxcolor=black@0.5:x=(w-text_w)/2:y=h-60")
    ffmpeg(["-i", scena["percorso"], "-vf", filtro, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(uscita)])


def scena_grafico_probabilita(scena, larghezza, altezza, fps, uscita: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(scena["percorso_csv"]) as file_csv:
        righe = list(csv.DictReader(file_csv))
    classi = []   # le colonne del CSV che sono probabilità
    for nome in righe[0]:
        if nome not in {"inizio_secondi", "fine_secondi", "classe", "warning_arma"}:
            classi.append(nome)
    tempi = []
    for riga in righe:
        tempi.append(float(riga["fine_secondi"]))
    figura, asse = plt.subplots(figsize=(larghezza / 100, altezza / 100), dpi=100)
    for classe in classi:  # SEGNAPOSTO DESIGN: colori di matplotlib
        valori = []
        for riga in righe:
            valori.append(float(riga[classe]))
        asse.plot(tempi, valori, label=classe, linewidth=2)
    asse.set_xlabel("secondi")
    asse.set_ylabel("probabilità")
    asse.set_ylim(0, 1)
    asse.legend(loc="upper right")
    figura.tight_layout()
    immagine = uscita.with_suffix(".png")
    figura.savefig(immagine)
    plt.close(figura)
    ffmpeg(["-loop", "1", "-i", str(immagine), "-t", str(scena["secondi"]), "-r", str(fps),
            "-vf", f"scale={larghezza}:{altezza}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(uscita)])


SCENE = {"titolo": scena_titolo, "clip": scena_clip, "grafico_probabilita": scena_grafico_probabilita}


def monta(percorso_scaletta: Path, percorso_uscita: Path) -> None:
    scaletta = json.loads(percorso_scaletta.read_text())
    larghezza, altezza = scaletta["risoluzione"]
    with tempfile.TemporaryDirectory() as cartella_temporanea:
        pezzi = []
        for numero, scena in enumerate(scaletta["scene"]):
            pezzo = Path(cartella_temporanea) / f"{numero:03d}.mp4"
            percorso_necessario = scena.get("percorso")   # il video della clip, oppure il CSV del grafico
            if not percorso_necessario:
                percorso_necessario = scena.get("percorso_csv")
            if percorso_necessario and not Path(percorso_necessario).exists():
                print(f"salto la scena {numero} ({scena['tipo']}): manca {percorso_necessario}")
                continue
            SCENE[scena["tipo"]](scena, larghezza, altezza, scaletta["fps"], pezzo)
            pezzi.append(pezzo)
        # l'elenco dei pezzi per il concat di ffmpeg: una riga "file '...'" per pezzo
        elenco = Path(cartella_temporanea) / "elenco.txt"
        testo_elenco = ""
        for pezzo in pezzi:
            testo_elenco += f"file '{pezzo}'\n"
        elenco.write_text(testo_elenco)
        percorso_uscita.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg(["-f", "concat", "-safe", "0", "-i", str(elenco), "-c", "copy", str(percorso_uscita)])
    print(f"presentazione: {percorso_uscita} ({len(pezzi)} scene)")


def comando_monta_presentazione(argomenti=None) -> None:
    parser = argparse.ArgumentParser(description="Monta la video presentation da media/presentation/scaletta.json")
    parser.add_argument("--scaletta", default="media/presentation/scaletta.json")
    parser.add_argument("--uscita", default="runs/presentazione.mp4")
    argomenti = parser.parse_args(argomenti)
    monta(Path(argomenti.scaletta), Path(argomenti.uscita))


# ==============================================================================================================
# riga di comando: python -m sicurezza.strumenti <comando> [opzioni]
# ==============================================================================================================
COMANDI = {"annota-scippi": comando_annota_scippi, "annota-tipo-arma": comando_annota_tipo_arma,
           "annota-anomalie": comando_annota_anomalie,
           "scarica-campioni": comando_scarica_campioni,
           "scarica-negativi-coco": comando_scarica_negativi_coco, "scarica-video-cc": comando_scarica_video_cc,
           "esporta-risultati-sito": comando_esporta_risultati_sito,
           "monta-presentazione": comando_monta_presentazione}


def main(argomenti=None) -> None:
    argomenti = sys.argv[1:] if argomenti is None else argomenti
    if not argomenti or argomenti[0] not in COMANDI:
        print("uso: python -m sicurezza.strumenti {" + "|".join(COMANDI) + "} [opzioni]  (--help dopo il comando)")
        raise SystemExit(2)
    COMANDI[argomenti[0]](argomenti[1:])


if __name__ == "__main__":
    main()
