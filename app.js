// Avvio del sito. Nessun dato inventato: mappa e risultati leggono i file in dati/.
import { riempiSegnaposto, quandoVisibile, webglDisponibile } from "./js/comune.js";
import { creaElencoPassi } from "./js/passi-modello.js";
import { avviaPulviscolo } from "./js/lampione.js";

riempiSegnaposto();
creaElencoPassi();
avviaPulviscolo(document.querySelector(".pulviscolo"));

// Le 34 voci del Viminale (Prefettura di Roma, delitti commessi a Roma Capitale, 2024): nessuna riguarda le armi
const VOCI_VIMINALE = ["Attacks", "Massacre", "Completed murders", "Infanticides", "Attempted murders",   // D64: in inglese
  "Manslaughter", "Negligent homicides", "Intentional injuries", "Beatings", "Threats", "Insults", "Sexual violence",
  "Sexual acts with minors", "Corruption of minors", "Thefts", "Receiving stolen goods", "Robberies", "Extortion", "Usury",
  "Kidnappings", "Criminal association", "Mafia-type association", "Money laundering",
  "Fraud and computer fraud", "Arson", "Criminal damage", "Criminal damage followed by fire", "Smuggling",
  "Drugs", "Exploitation of prostitution and child pornography", "Computer crimes",
  "Counterfeiting of brands and products", "Intellectual property violations", "Other crimes"];
const elencoVoci = document.getElementById("voci-viminale");
for (const voce of VOCI_VIMINALE) elencoVoci.appendChild(Object.assign(document.createElement("li"), { textContent: voce }));

const tre = webglDisponibile();
if (tre) {
  quandoVisibile(document.getElementById("roma"), () => import("./js/mappa-roma.js").then((modulo) => modulo.avviaMappa()));
  quandoVisibile(document.getElementById("come-funziona"), () => import("./js/modello-3d.js").then((modulo) => modulo.avviaModello()));
} else {
  for (const id of ["mappa-3d", "modello-3d"]) {
    const tela = document.getElementById(id);
    tela.replaceWith(Object.assign(document.createElement("p"), { className: "stato-vuoto", textContent: "This part uses WebGL, which is not available in this browser." }));
  }
}

// Risultati: solo se l'altra parte del progetto li ha scritti in dati/risultati.json
fetch("dati/risultati.json").then((risposta) => risposta.json()).then((risultati) => {
  if (!risultati.modelli || risultati.modelli.length === 0) return;   // stato vuoto dichiarato nell'HTML
  const righe = risultati.modelli.map((modello) => `<tr><td style="text-align:left">${modello.nome}</td>
    <td>${modello.macro_f1.toFixed(3)}</td><td>${modello.latenza_cpu_ms.toFixed(1)}</td></tr>`).join("");
  const contenitore = document.getElementById("tabella-risultati");
  contenitore.className = "";
  contenitore.innerHTML = `<table><tr><th style="text-align:left">Model</th><th>Macro-F1 (test)</th><th>ms/window (CPU)</th></tr>${righe}</table>`;
  if (!risultati.violenza) return;
  // Violenza sì/no (D48): sistema consegnato (pixel + scheletri) contro i soli scheletri, clip intere mai viste
  const percento = (valore) => `${Math.round(valore * 100)}%`;
  const cella = (misura) => `${percento(misura.accuratezza)}${misura.ic95 ? ` <small>(${percento(misura.ic95[0])}–${percento(misura.ic95[1])})</small>` : ""}`;
  const rigaViolenza = (nome, dati) => `<tr><td style="text-align:left">${nome}</td><td>${cella(dati.per_finestra)}</td>
    <td>${cella(dati.pezzi_10s)}</td><td>${cella(dati.pezzi_30s)}</td><td>${percento(dati.pezzi_30s.falsi_allarmi)}</td></tr>`;
  contenitore.insertAdjacentHTML("beforeend", `<table><tr><th style="text-align:left">Violence yes/no</th><th>Per window (1.6 s)</th>
    <th>10 s segments</th><th>30 s segments</th><th>False alarms (30 s)</th></tr>
    ${rigaViolenza("Pixels + skeletons (delivered)", risultati.violenza.sistema)}${rigaViolenza("Skeletons only", risultati.violenza.solo_scheletri)}</table>
    <p class="nota">Accuracy on ${risultati.violenza.sistema.pezzi_30s.pezzi} 30 s segments of never-seen clips; 95% interval in brackets. A segment raises an alarm if ${risultati.violenza.sistema.pezzi_30s.finestre_di_fila} windows in a row are violent${risultati.violenza.sistema.pezzi_30s.media_mobile > 1 ? `, with the probability averaged over the last ${risultati.violenza.sistema.pezzi_30s.media_mobile}` : ""} (rule chosen on validation).</p>`);
}).catch(() => {});
