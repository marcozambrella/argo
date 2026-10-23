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

// Risultati (Marco, 23/10: "più specifiche, ora sono vaghe e solo su due categorie"). Due file: dati/risultati.json
// (esportato dal codice) e dati/risultati-aggiuntivi.json (dal DIARIO e da runs/argo_casi_uso). Nessun numero inventato.
const NOMI_CLASSI = { normale: "Normal", violenza: "Violence", violenza_con_arma: "Armed violence", scippo: "Snatching" };
const percento = (valore) => `${Math.round(valore * 100)}%`;
const numero = (valore) => valore.toLocaleString("en-GB");
const barra = (valore, testo = percento(valore)) =>   // barra orizzontale a un solo colore: la lunghezza è il valore
  `<span class="barretta" role="img" aria-label="${testo}"><span style="width:${Math.max(valore * 100, 0.5)}%"></span></span><span class="valore">${testo}</span>`;

const sezioneViolenza = (r, extra) => {
  const fonti = extra.violenza_per_fonte_30s.map((f) => `<tr><th scope="row">${f.fonte}</th><td class="cella-barra">${barra(f.accuratezza)}</td></tr>`).join("");
  return `<h3>1 · Violence: yes or no</h3>
    <p class="spiega">Is someone being attacked? ARGO answers every 0.4 s and raises the alarm after ${r.sistema.pezzi_30s.finestre_di_fila} violent windows in a row. Accuracy on 30-second clips, by source:</p>
    <table class="tabella-dati compatta"><tbody>${fonti}</tbody></table>`;
};

const sezioneClassi = (st) => {
  const totali = st.matrice_confusione.map((riga) => riga.reduce((a, b) => a + b, 0));
  const righe = st.classi.map((classe) => `<tr><th scope="row">${NOMI_CLASSI[classe]}</th>
    <td class="cella-barra">${barra(st.recall[classe])}</td><td class="cella-barra">${barra(st.f1[classe], st.f1[classe].toFixed(2))}</td></tr>`).join("");
  return `<h3>2 · All four classes</h3>
    <p class="spiega">For each window the skeleton network picks one of four classes.</p>
    <div class="scorre"><table class="tabella-dati"><thead><tr><th scope="col">Class</th>
      <th scope="col">Recall: how many it finds</th><th scope="col">F1</th></tr></thead><tbody>${righe}</tbody></table></div>
    <p class="nota">Armed violence and snatching are still weak: public footage of them is scarce (${numero(totali[3])} snatching windows in the test, against ${numero(totali[0])} normal ones).</p>`;
};

const sezioneArmi = (w, a) => {
  const prove = a.prove.map((p) => `<tr><th scope="row">${p.prova}</th><td class="cella-barra">${barra(p.map50, p.map50.toFixed(2))}</td></tr>`).join("");
  return `<h3>3 · Weapons</h3>
    <div class="due-colonne">
      <div><p class="spiega">A dedicated detector looks for pistols, knives and rifles (mAP50, 1 = perfect).</p>
        <table class="tabella-dati compatta"><tbody>${prove}</tbody></table></div>
      <div class="scheda-misura"><b class="sc">The WEAPON WARNING</b>
        <p>Fires when a weapon is near a wrist in at least ${w.frame_minimi} of 40 frames. Tuned to avoid false alarms: when it fires it is right ${percento(w.precisione)} of the time, but it still misses most weapons.</p></div>
    </div>`;
};

const sezioneReali = (prove) => {
  const normali = prove.filter((p) => p.vero === "normal"), incidenti = prove.filter((p) => p.vero !== "normal");
  return `<h3>4 · Real videos it never saw</h3>
    <div class="cifre">
      <p><b>${normali.filter((p) => !p.allarme).length}/${normali.length}</b><span>normal scenes (metro, streets of Rome) with no false alarm</span></p>
      <p><b>${incidenti.filter((p) => p.allarme).length}/${incidenti.length}</b><span>real incidents that raised an alarm: a shop robbery and a street fight</span></p>
    </div>
    <p class="spiega">ARGO is strongest on clear fights; quick snatchings and weapons seen from far away are the next step.</p>`;
};

const sezioneVelocita = () => `<h3>5 · Fast enough to act</h3>
    <div class="cifre">
      <p><b>Every 0.4 s</b><span>a new decision: ARGO never stops looking</span></p>
      <p><b>Seconds</b><span>from the first blows to the VIOLENCE ALARM, while the fight is still on</span></p>
    </div>
    <p class="spiega">ARGO is built to raise the alarm while something is happening, not to find it in tomorrow's footage. On a GPU it runs live; on an ordinary laptop it still works, just a little behind real time.</p>`;

const glossario = `<details class="glossario"><summary class="sc">How to read these numbers</summary><dl>
    <dt>Window</dt><dd>40 frames, 1.6 s of video. A new one starts every 10 frames.</dd>
    <dt>Recall</dt><dd>Of the real cases, how many ARGO finds.</dd>
    <dt>Precision</dt><dd>Of the times ARGO says "this class", how many are right.</dd>
    <dt>F1</dt><dd>One number that balances recall and precision (0 to 1).</dd>
    <dt>False alarms</dt><dd>Of the calm segments, how many raise an alarm anyway.</dd>
    <dt>Test set</dt><dd>15% of the videos, kept apart from the start: no video is ever in both training and test.</dd>
  </dl></details>`;

Promise.all([fetch("dati/risultati.json").then((r) => r.json()), fetch("dati/risultati-aggiuntivi.json").then((r) => r.json())])
  .then(([risultati, extra]) => {
    const st = risultati.modelli?.find((m) => m.nome === "ST-GCN");
    if (!st || !risultati.violenza) return;   // stato vuoto dichiarato nell'HTML
    for (const m of risultati.modelli) m.classi = risultati.classi;
    const s30 = risultati.violenza.sistema.pezzi_30s;
    const contenitore = document.getElementById("tabella-risultati");
    contenitore.className = "risultati";
    contenitore.innerHTML = `<div class="cifre grandi">
        <p><b>${percento(s30.accuratezza)}</b><span>violence yes/no on 30 s segments of never-seen videos <small>(95% CI ${percento(s30.ic95[0])}–${percento(s30.ic95[1])})</small></span></p>
        <p><b>${percento(s30.recall)}</b><span>of the fights raise an alarm</span></p>
        <p><b>${percento(s30.falsi_allarmi)}</b><span>false alarms on calm segments</span></p>
        <p><b>${st.macro_f1.toFixed(2)}</b><span>macro-F1 on the four classes: armed violence and snatching are still weak</span></p>
      </div>
      <p class="spiega">Measured on videos kept apart from training.</p>
      ${sezioneViolenza(risultati.violenza, extra)}${sezioneClassi(st)}${sezioneArmi(risultati.warning_arma, extra.rilevatore_armi)}
      ${sezioneReali(extra.prove_reali)}${sezioneVelocita()}${glossario}`;
  }).catch(() => {});
