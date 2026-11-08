// Avvio del sito. Nessun dato inventato: mappa e risultati leggono i file in dati/.
import { riempiSegnaposto, quandoVisibile, webglDisponibile } from "./js/comune.js?v=20261108165744";
import { creaElencoPassi } from "./js/passi-modello.js?v=20261108165744";
import { avviaPulviscolo } from "./js/lampione.js?v=20261108165744";

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
  quandoVisibile(document.getElementById("roma"), () => import("./js/mappa-roma.js?v=20261108165744").then((modulo) => modulo.avviaMappa()));
  quandoVisibile(document.getElementById("come-funziona"), () => import("./js/modello-3d.js?v=20261108165744").then((modulo) => modulo.avviaModello()));
} else {
  for (const id of ["mappa-3d", "modello-3d"]) {
    const tela = document.getElementById(id);
    tela.replaceWith(Object.assign(document.createElement("p"), { className: "stato-vuoto", textContent: "This part uses WebGL, which is not available in this browser." }));
  }
}

// Risultati (Marco, 23/10: "più specifiche, ora sono vaghe e solo su due categorie"). Due file: dati/risultati.json
// (esportato dal codice) e dati/risultati-aggiuntivi.json (dal DIARIO e da runs/argo_casi_uso). Nessun numero inventato.
const percento = (valore) => `${Math.round(valore * 100)}%`;
const barra = (valore, testo = percento(valore)) =>   // barra orizzontale a un solo colore: la lunghezza è il valore
  `<span class="barretta" role="img" aria-label="${testo}"><span style="width:${Math.max(valore * 100, 0.5)}%"></span></span><span class="valore">${testo}</span>`;

const sezioneViolenza = (r, extra) => {
  const fonti = extra.violenza_per_fonte_30s.map((f) => `<tr><th scope="row">${f.fonte}</th><td class="cella-barra">${barra(f.accuratezza)}</td></tr>`).join("");
  return `<h3>1 · Violence: yes or no</h3>
    <p class="spiega">Is someone being attacked? ARGO answers every 0.4 s and raises the alarm after ${r.sistema.pezzi_30s.finestre_di_fila} violent windows in a row. Accuracy on 30-second clips, by source:</p>
    <table class="tabella-dati compatta"><tbody>${fonti}</tbody></table>`;
};

const sezioneArmi = (v) => {
  const prove = v.prove.map((p) => `<tr><th scope="row">${p.prova}</th><td class="cella-barra">${barra(p.accuratezza)}</td></tr>`).join("");
  return `<h3>2 · Weapons and snatching</h3>
    <div class="due-colonne">
      <div><p class="spiega">Is there a weapon? Is it a snatching? ARGO answers from the pixels: yes or no on 10-second segments.</p>
        <table class="tabella-dati compatta"><tbody>${prove}</tbody></table></div>
      <div class="scheda-misura"><b class="sc">The WEAPON WARNING</b>
        <p>Fires when ARGO sees a weapon, whatever else is happening. It catches ${percento(v.warning_recall_armati)} of the armed segments and fires on ${percento(v.warning_falsi)} of the segments without a weapon.</p></div>
    </div>`;
};

const sezioneVelocita = () => `<h3>3 · Fast enough to act</h3>
    <div class="cifre">
      <p><b>Every 0.4&nbsp;s</b><span>a new decision: ARGO never stops looking</span></p>
      <p><b>Seconds</b><span>from the first blows to the VIOLENCE ALARM, while the fight is still on</span></p>
    </div>
    <p class="spiega">ARGO is built to raise the alarm while something is happening, not to find it in tomorrow's footage. On a GPU it runs live; on an ordinary laptop it still works, with a delay that version 1.2 cut in half.</p>`;

const glossario = `<details class="glossario"><summary class="sc">How to read these numbers</summary><dl>
    <dt>Window</dt><dd>40 frames, 1.6 s of video. A new one starts every 10 frames.</dd>
    <dt>False alarms</dt><dd>Of the calm segments, how many raise an alarm anyway.</dd>
    <dt>Test set</dt><dd>15% of the videos, kept apart from the start: no video is ever in both training and test.</dd>
  </dl></details>`;

Promise.all([fetch("dati/risultati.json?v=20261108165744").then((r) => r.json()), fetch("dati/risultati-aggiuntivi.json?v=20261108165744").then((r) => r.json())])
  .then(([risultati, extra]) => {
    // Marco, 04/11: nei risultati solo e sempre ARGO intero, mai le reti da sole
    if (!risultati.violenza) return;   // stato vuoto dichiarato nell'HTML
    const s30 = risultati.violenza.sistema.pezzi_30s;
    const contenitore = document.getElementById("tabella-risultati");
    contenitore.className = "risultati";
    contenitore.innerHTML = `<div class="cifre grandi">
        <p><b>${percento(s30.accuratezza)}</b><span>violence yes/no on 30 s segments of never-seen videos <small>(95% CI ${percento(s30.ic95[0])}–${percento(s30.ic95[1])})</small></span></p>
        <p><b>${percento(s30.recall)}</b><span>of the fights raise an alarm</span></p>
        <p><b>${percento(s30.falsi_allarmi)}</b><span>false alarms on calm segments</span></p>
        <p><b>${percento(extra.reti_video.prove[0].accuratezza)}</b><span>weapon yes/no on 10 s segments of never-seen videos</span></p>
      </div>
      <p class="spiega">Measured on videos kept apart from training: violence with version 1.1 (version 1.2 is just as accurate and takes half the time), weapons and snatching with version 1.3.</p>
      ${sezioneViolenza(risultati.violenza, extra)}${sezioneArmi(extra.reti_video)}
      ${sezioneVelocita()}${glossario}`;
  }).catch(() => {});
