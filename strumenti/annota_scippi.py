"""Strumento per annotare a mano l'istante dello scippo nei video chain-snatching (D15).

Apre una pagina nel browser con il video e i tasti:
    Spazio  play / pausa              ← →   indietro / avanti 0,2 s (con Maiusc: 1 s)
    , .     un frame indietro/avanti  1 2 3 velocità 0,25× / 0,5× / 1×
    I       inizio scippo (istante corrente)
    F       fine scippo  → l'evento viene aggiunto alla lista (si possono segnare più scippi)
    N       nel video NON si vede uno scippo (verrà escluso)
    Canc    cancella gli eventi di questo video
    Invio   salva e passa al prossimo video da annotare
Ogni salvataggio scrive subito il file JSON: si può chiudere e riprendere quando si vuole.

Uso:
    uv run python strumenti/annota_scippi.py
    (poi apri http://localhost:8770 nel browser)
"""

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

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
 <span><span class="tasto">Canc</span> cancella</span><span><span class="tasto">Invio</span> salva e prossimo</span></div>
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=CARTELLA_VIDEO_PREDEFINITA)
    parser.add_argument("--uscita", default=FILE_ANNOTAZIONI_PREDEFINITO)
    parser.add_argument("--porta", type=int, default=8770)
    argomenti = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", argomenti.porta),
                                 crea_gestore(Path(argomenti.video), Path(argomenti.uscita)))
    print(f"Apri http://localhost:{argomenti.porta} — annotazioni in {argomenti.uscita} (Ctrl+C per chiudere)")
    server.serve_forever()


if __name__ == "__main__":
    main()
