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
  const cella = (m) => `${percento(m.accuratezza)}${m.ic95 ? ` <small>(${percento(m.ic95[0])}–${percento(m.ic95[1])})</small>` : ""}`;
  const riga = (nome, d) => `<tr><th scope="row">${nome}</th><td>${cella(d.per_finestra)}</td><td>${cella(d.pezzi_10s)}</td>
    <td>${cella(d.pezzi_30s)}</td><td>${percento(d.pezzi_30s.recall)}</td><td>${percento(d.pezzi_30s.falsi_allarmi)}</td></tr>`;
  const fonti = extra.violenza_per_fonte_30s.map((f) => `<tr><th scope="row">${f.fonte}</th><td class="cella-barra">${barra(f.accuratezza)}</td></tr>`).join("");
  const rami = extra.violenza_rami_30s;
  return `<h3>1 · Violence: yes or no</h3>
    <p class="spiega">The question a control room asks first: is someone being attacked? ARGO answers it every 1.6 s; an alarm needs ${r.sistema.pezzi_30s.finestre_di_fila} violent windows in a row (about 4 s), a rule chosen on the validation set.</p>
    <div class="scorre"><table class="tabella-dati"><thead><tr><th scope="col"></th><th scope="col">Per window (1.6 s)</th><th scope="col">${r.sistema.pezzi_10s.pezzi} segments of 10 s</th>
      <th scope="col">${r.sistema.pezzi_30s.pezzi} segments of 30 s</th><th scope="col">Fights caught (30 s)</th><th scope="col">False alarms (30 s)</th></tr></thead>
      <tbody>${riga("Delivered system: skeletons + video network", r.sistema)}${riga("Skeletons only", r.solo_scheletri)}</tbody></table></div>
    <p class="nota">Accuracy, with the 95% confidence interval in brackets. "Fights caught" is the recall: of the 30 s segments that contain a fight, how many raise an alarm. "False alarms": of the calm segments, how many raise one anyway.</p>
    <div class="due-colonne">
      <div><h4>By source (30 s segments)</h4><table class="tabella-dati compatta"><tbody>${fonti}</tbody></table></div>
      <div><h4>Why two networks</h4><table class="tabella-dati compatta"><tbody>
        <tr><th scope="row">Skeletons only</th><td class="cella-barra">${barra(rami.solo_scheletri)}</td></tr>
        <tr><th scope="row">Video network only</th><td class="cella-barra">${barra(rami.solo_rete_video)}</td></tr>
        <tr><th scope="row">Together (delivered)</th><td class="cella-barra">${barra(rami.insieme)}</td></tr></tbody></table>
        <p class="nota">Accuracy on 30 s segments. The skeletons see how people move; the video network sees the pixels. Averaged, they make fewer mistakes than either alone.</p></div>
    </div>`;
};

const sezioneClassi = (st, gt) => {
  const totali = st.matrice_confusione.map((riga) => riga.reduce((a, b) => a + b, 0));
  const righe = st.classi.map((classe, i) => `<tr><th scope="row">${NOMI_CLASSI[classe]}</th><td>${numero(totali[i])}</td>
    <td class="cella-barra">${barra(st.recall[classe])}</td><td class="cella-barra">${barra(st.precisione[classe])}</td>
    <td class="cella-barra">${barra(st.f1[classe], st.f1[classe].toFixed(2))}</td></tr>`).join("");
  // matrice di confusione normalizzata per riga: "di 100 finestre vere di questa classe, quante finiscono in ciascuna"
  const intestazione = st.classi.map((c) => `<th scope="col">${NOMI_CLASSI[c]}</th>`).join("");
  const matrice = st.matrice_confusione.map((riga, i) => `<tr><th scope="row">${NOMI_CLASSI[st.classi[i]]}</th>${riga.map((n, j) => {
    const quota = n / totali[i];
    return `<td class="cella-calore${i === j ? " diagonale" : ""}${quota > 0.5 ? " scura" : ""}" style="--quota:${quota.toFixed(3)}"
      title="${NOMI_CLASSI[st.classi[i]]} → ${NOMI_CLASSI[st.classi[j]]}: ${numero(n)} of ${numero(totali[i])} windows">${percento(quota)}</td>`;
  }).join("")}</tr>`).join("");
  const scippo = st.matrice_confusione[3];
  return `<h3>2 · All four classes</h3>
    <p class="spiega">For each 1.6 s window the skeleton network (ST-GCN) picks one of four classes. Measured on ${numero(st.finestre_test)} test windows from videos it never saw.</p>
    <div class="scorre"><table class="tabella-dati"><thead><tr><th scope="col">Class</th><th scope="col">Test windows</th>
      <th scope="col">Recall: how many it finds</th><th scope="col">Precision: how often it is right when it says so</th><th scope="col">F1</th></tr></thead>
      <tbody>${righe}</tbody></table></div>
    <p class="nota">Macro-F1 ${st.macro_f1.toFixed(2)} (the average of the four F1), accuracy ${percento(st.accuratezza)}. Armed violence and snatching are weak: in the test there are only ${numero(totali[2])} and ${numero(totali[3])} windows of them, against ${numero(totali[0])} normal ones, and the same scarcity held in training.</p>
    <h4>Where the mistakes go</h4>
    <p class="spiega">Each row is the true class; each column is what ARGO said. Read a row as "of 100 real windows of this class…". The diagonal is the right answer.</p>
    <div class="scorre"><table class="tabella-dati matrice"><thead><tr><th scope="col">True ↓ · ARGO says →</th>${intestazione}</tr></thead><tbody>${matrice}</tbody></table></div>
    <p class="nota">Of ${numero(totali[3])} real snatching windows, ${scippo[3]} are recognised as snatching and ${scippo[0]} are taken for normal: a snatch lasts a second or two and often looks like walking past. Armed violence is mostly confused with normal and violence, because the weapon is too small for the detector in many CCTV frames.</p>
    <h4>Two models compared</h4>
    <div class="scorre"><table class="tabella-dati"><thead><tr><th scope="col">Model</th><th scope="col">Macro-F1</th><th scope="col">Accuracy</th>
      ${st.classi.map((c) => `<th scope="col">F1 ${NOMI_CLASSI[c].toLowerCase()}</th>`).join("")}<th scope="col">ms per window (CPU)</th><th scope="col">Parameters</th></tr></thead>
      <tbody>${[st, gt].map((m) => `<tr><th scope="row">${m.nome}${m === st ? " <small>(chosen)</small>" : ""}</th><td>${m.macro_f1.toFixed(2)}</td><td>${percento(m.accuratezza)}</td>
        ${m.classi.map((c) => `<td>${m.f1[c].toFixed(2)}</td>`).join("")}<td>${m.latenza_cpu_ms.toFixed(0)}</td><td>${(m.parametri / 1e6).toFixed(1)} M</td></tr>`).join("")}</tbody></table></div>`;
};

const sezioneArmi = (w, a) => {
  const prove = a.prove.map((p) => `<tr><th scope="row">${p.prova}</th><td class="cella-barra">${barra(p.map50, p.map50.toFixed(2))}</td></tr>`).join("");
  return `<h3>3 · Weapons</h3>
    <p class="spiega">A separate detector (${a.modello}) looks for pistols, knives and rifles. mAP50 measures how well it finds and boxes them: 1 is perfect.</p>
    <div class="due-colonne">
      <div><table class="tabella-dati compatta"><thead><tr><th scope="col">Test</th><th scope="col">mAP50</th></tr></thead><tbody>${prove}</tbody></table>
        <p class="nota">False alarms: ${a.falsi_allarmi.allarmi} on ${a.falsi_allarmi.foto} ${a.falsi_allarmi.cosa}.</p></div>
      <div class="scheda-misura"><b class="sc">The WEAPON WARNING</b>
        <p>It fires when a weapon with confidence ≥ ${w.soglia_confidenza} is near a wrist in at least ${w.frame_minimi} of 40 frames. Tuned to avoid false alarms, so it is cautious: when it fires it is right ${percento(w.precisione)} of the time, but it catches only ${percento(w.recall)} of the windows with a weapon.</p></div>
    </div>`;
};

const sezioneReali = (prove, folle) => {
  const segno = { giusto: "✓ right", "in parte": "◐ partly", mancato: "✗ missed" };
  const righe = prove.map((p) => `<tr class="esito-${p.esito.replace(" ", "-")}"><th scope="row">${p.scena}</th><td>${p.vero}</td><td>${p.risposta}</td>
    <td>${p.allarme ? "ALARM" : "—"}</td><td>${segno[p.esito]}</td></tr>`).join("");
  const normali = prove.filter((p) => p.vero === "normal"), incidenti = prove.filter((p) => p.vero !== "normal");
  return `<h3>4 · Real videos it never saw</h3>
    <p class="spiega">The test set comes from the same datasets used in training. To see how ARGO behaves outside them, we ran the delivered system on ${prove.length} public videos (CC BY on YouTube, Pexels).</p>
    <div class="cifre">
      <p><b>${normali.filter((p) => !p.allarme).length}/${normali.length}</b><span>normal scenes with no false alarm</span></p>
      <p><b>${incidenti.filter((p) => p.allarme).length}/${incidenti.length}</b><span>incidents that raised an alarm</span></p>
      <p><b>0/${prove.filter((p) => p.vero.includes("armed")).length}</b><span>videos with a real weapon that got a WEAPON WARNING</span></p>
    </div>
    <div class="scorre"><table class="tabella-dati prove"><thead><tr><th scope="col">Video</th><th scope="col">What happens</th><th scope="col">What ARGO says</th>
      <th scope="col">Alarm</th><th scope="col">Verdict</th></tr></thead><tbody>${righe}</tbody></table></div>
    <p class="nota">In crowded scenes never seen in training (stations, campuses, streets) the skeleton network gives the right class in ${percento(folle.minimo)}–${percento(folle.massimo)} of the windows. On real incidents it is honest to say it: ARGO works best on clear fights filmed from above, and misses quick snatchings and weapons seen from far away.</p>`;
};

const sezioneVelocita = (st, v) => `<h3>5 · Speed</h3>
    <div class="cifre">
      <p><b>${st.latenza_cpu_ms.toFixed(0)} ms</b><span>to classify one 1.6 s window, on a CPU</span></p>
      <p><b>${v.estrazione_gpu_fps} fps</b><span>${v.estrazione_gpu_dove}: the heavy part (pose and tracking), fast enough for 3 cameras at 25 fps</span></p>
      <p><b>${v.pannello_cpu_fps} fps</b><span>${v.pannello_cpu_dove}</span></p>
    </div>
    <p class="nota">The heavy part is pose and tracking: on a GPU it runs faster than real time. On a laptop the panel records and analyses with a delay, which it shows on screen.</p>`;

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
    const st = risultati.modelli?.find((m) => m.nome === "ST-GCN"), gt = risultati.modelli?.find((m) => m.nome !== "ST-GCN");
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
      <p class="spiega">Test set: ${percento(extra.test.quota_video)} of the videos (${extra.test.fonti.join(", ")}), split by video before training. ${numero(st.finestre_test)} windows of 1.6 s.</p>
      ${sezioneViolenza(risultati.violenza, extra)}${sezioneClassi(st, gt)}${sezioneArmi(risultati.warning_arma, extra.rilevatore_armi)}
      ${sezioneReali(extra.prove_reali, extra.folle_mai_viste)}${sezioneVelocita(st, extra.velocita)}${glossario}`;
  }).catch(() => {});
