// Avvio del sito. Nessun dato inventato: mappa e risultati leggono i file in dati/.
import { riempiSegnaposto, quandoVisibile, webglDisponibile } from "./js/comune.js";
import { creaElencoPassi } from "./js/passi-modello.js";
import { avviaPulviscolo } from "./js/lampione.js";

riempiSegnaposto();
creaElencoPassi();
avviaPulviscolo(document.querySelector(".pulviscolo"));

// Le 34 voci del Viminale (Prefettura di Roma, delitti commessi a Roma Capitale, 2024): nessuna riguarda le armi
const VOCI_VIMINALE = ["Attentati", "Strage", "Omicidi volontari consumati", "Infanticidi", "Tentati omicidi",
  "Omicidio preterintenzionale", "Omicidi colposi", "Lesioni dolose", "Percosse", "Minacce", "Ingiurie", "Violenze sessuali",
  "Atti sessuali con minorenne", "Corruzione di minorenne", "Furti", "Ricettazione", "Rapine", "Estorsioni", "Usura",
  "Sequestri di persona", "Associazione per delinquere", "Associazione di tipo mafioso", "Riciclaggio e impiego di denaro",
  "Truffe e frodi informatiche", "Incendi", "Danneggiamenti", "Danneggiamento seguito da incendio", "Contrabbando",
  "Stupefacenti", "Sfruttamento della prostituzione e pornografia minorile", "Delitti informatici",
  "Contraffazione di marchi e prodotti industriali", "Violazione alla proprietà intellettuale", "Altri delitti"];
const elencoVoci = document.getElementById("voci-viminale");
for (const voce of VOCI_VIMINALE) elencoVoci.appendChild(Object.assign(document.createElement("li"), { textContent: voce }));

const tre = webglDisponibile();
if (tre) {
  quandoVisibile(document.getElementById("roma"), () => import("./js/mappa-roma.js").then((modulo) => modulo.avviaMappa()));
  quandoVisibile(document.getElementById("come-funziona"), () => import("./js/modello-3d.js").then((modulo) => modulo.avviaModello()));
} else {
  for (const id of ["mappa-3d", "modello-3d"]) {
    const tela = document.getElementById(id);
    tela.replaceWith(Object.assign(document.createElement("p"), { className: "stato-vuoto", textContent: "Questa parte usa WebGL, non disponibile in questo browser." }));
  }
}

// Risultati: solo se l'altra parte del progetto li ha scritti in dati/risultati.json
fetch("dati/risultati.json").then((risposta) => risposta.json()).then((risultati) => {
  if (!risultati.modelli || risultati.modelli.length === 0) return;   // stato vuoto dichiarato nell'HTML
  const righe = risultati.modelli.map((modello) => `<tr><td style="text-align:left">${modello.nome}</td>
    <td>${modello.macro_f1.toFixed(3)}</td><td>${modello.latenza_cpu_ms.toFixed(1)}</td></tr>`).join("");
  const contenitore = document.getElementById("tabella-risultati");
  contenitore.className = "";
  contenitore.innerHTML = `<table><tr><th style="text-align:left">Modello</th><th>Macro-F1 (test)</th><th>ms/finestra (CPU)</th></tr>${righe}</table>`;
  if (!risultati.violenza) return;
  // Violenza sì/no (D48): sistema consegnato (pixel + scheletri) contro i soli scheletri, clip intere mai viste
  const percento = (valore) => `${Math.round(valore * 100)}%`;
  const cella = (misura) => `${percento(misura.accuratezza)}${misura.ic95 ? ` <small>(${percento(misura.ic95[0])}–${percento(misura.ic95[1])})</small>` : ""}`;
  const rigaViolenza = (nome, dati) => `<tr><td style="text-align:left">${nome}</td><td>${cella(dati.per_finestra)}</td>
    <td>${cella(dati.pezzi_10s)}</td><td>${cella(dati.pezzi_30s)}</td><td>${percento(dati.pezzi_30s.falsi_allarmi)}</td></tr>`;
  contenitore.insertAdjacentHTML("beforeend", `<table><tr><th style="text-align:left">Violenza sì/no</th><th>Per finestra (1,6 s)</th>
    <th>Pezzi da 10 s</th><th>Pezzi da 30 s</th><th>Falsi allarmi (30 s)</th></tr>
    ${rigaViolenza("Pixel + scheletri (consegnato)", risultati.violenza.sistema)}${rigaViolenza("Solo scheletri", risultati.violenza.solo_scheletri)}</table>
    <p class="nota">Accuratezza su ${risultati.violenza.sistema.pezzi_30s.pezzi} pezzi da 30 s di clip mai viste; tra parentesi l'intervallo al 95%. Un pezzo dà allarme se ${risultati.violenza.sistema.pezzi_30s.finestre_di_fila} finestre di fila sono violente, con la probabilità mediata sulle ultime ${risultati.violenza.sistema.pezzi_30s.media_mobile} (regola scelta sul validation).</p>`);
}).catch(() => {});
