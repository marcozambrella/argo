// Parti interattive del sito. Nessun dato inventato: grafici e risultati leggono i file in dati/.
const NS = "http://www.w3.org/2000/svg";
const crea = (nome, attributi = {}, padre) => {
  const elemento = document.createElementNS(NS, nome);
  for (const [chiave, valore] of Object.entries(attributi)) elemento.setAttribute(chiave, valore);
  if (padre) padre.appendChild(elemento);
  return elemento;
};
const colore = (variabile) => getComputedStyle(document.documentElement).getPropertyValue(variabile).trim();

// --- 1. Finestre scorrevoli: 40 frame, passo 10 -------------------------------------------------------------
(function finestreScorrevoli() {
  const svg = document.getElementById("svg-finestre");
  const cursore = document.getElementById("cursore-finestre");
  const numero = document.getElementById("numero-finestra");
  const FRAME_TOTALI = 160, FRAME_PER_FINESTRA = 40, PASSO = 10, LARGHEZZA_FRAME = 780 / FRAME_TOTALI;
  cursore.max = (FRAME_TOTALI - FRAME_PER_FINESTRA) / PASSO;
  function disegna() {
    svg.replaceChildren();
    const finestraAttiva = Number(cursore.value);
    for (let frame = 0; frame < FRAME_TOTALI; frame++) {
      const dentro = frame >= finestraAttiva * PASSO && frame < finestraAttiva * PASSO + FRAME_PER_FINESTRA;
      crea("rect", { x: 10 + frame * LARGHEZZA_FRAME, y: 20, width: LARGHEZZA_FRAME - 1, height: 40,
        fill: dentro ? colore("--accento") : colore("--bordo") }, svg);
    }
    for (let indice = 0; indice <= cursore.max; indice++) {  // tutte le finestre, sovrapposte
      const inizio = indice * PASSO;
      crea("rect", { x: 10 + inizio * LARGHEZZA_FRAME, y: 70 + (indice % 4) * 14, height: 10,
        width: FRAME_PER_FINESTRA * LARGHEZZA_FRAME, rx: 3,
        fill: indice === finestraAttiva ? colore("--accento") : colore("--finestra-attiva") }, svg);
    }
    const inizio = finestraAttiva * PASSO;
    const testo = crea("text", { x: 10, y: 14, "font-size": 13, fill: colore("--testo") }, svg);
    testo.textContent = `frame ${inizio}–${inizio + FRAME_PER_FINESTRA - 1}  (secondi ${(inizio / 25).toFixed(1)}–${((inizio + FRAME_PER_FINESTRA) / 25).toFixed(1)})`;
    numero.textContent = finestraAttiva + 1;
  }
  cursore.addEventListener("input", disegna);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", disegna);
  disegna();
})();

// --- 2. Grafo: due scheletri e un'arma, archi per tipo -----------------------------------------------------------
(function grafo() {
  const svg = document.getElementById("svg-grafo");
  const contenitore = document.getElementById("interruttori-archi");
  const OSSA = [[15,13],[13,11],[16,14],[14,12],[11,12],[5,11],[6,12],[5,6],[5,7],[6,8],[7,9],[8,10],[1,2],[0,1],[0,2],[1,3],[2,4],[3,5],[4,6]];
  const POSA = [[0,.07],[-.02,.05],[.02,.05],[-.04,.06],[.04,.06],[-.1,.2],[.1,.2],[-.13,.38],[.16,.3],[-.14,.52],[.3,.25],[-.07,.53],[.07,.53],[-.07,.76],[.07,.76],[-.07,1],[.07,1]];
  const persona = (centroX, specchio) => POSA.map(([x, y]) => [centroX + (specchio ? -x : x) * 280, 20 + y * 300]);
  const persone = [persona(200, false), persona(420, true)];
  const arma = [persone[0][10][0] + 18, persone[0][10][1] - 6];
  const TIPI = {
    osso: { etichetta: "Ossa dello scheletro", colore: "--testo", attivo: true },
    persone: { etichetta: "Persone diverse", colore: "--violenza", attivo: true },
    arma: { etichetta: "Arma – polso", colore: "--arma", attivo: true },
  };
  for (const [chiave, tipo] of Object.entries(TIPI)) {
    const etichetta = document.createElement("label");
    const casella = Object.assign(document.createElement("input"), { type: "checkbox", checked: tipo.attivo });
    casella.addEventListener("change", () => { tipo.attivo = casella.checked; disegna(); });
    etichetta.append(casella, " " + tipo.etichetta);
    contenitore.appendChild(etichetta);
  }
  function linea(a, b, tipo, opacita = 1) {
    crea("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: colore(TIPI[tipo].colore),
      "stroke-width": tipo === "osso" ? 2 : 1, opacity: opacita }, svg);
  }
  function disegna() {
    svg.replaceChildren();
    if (TIPI.persone.attivo)  // solo alcuni archi tra persone, per leggibilità
      for (const nodo of [0, 9, 10, 5, 6]) for (const altro of [0, 9, 10]) linea(persone[0][nodo], persone[1][altro], "persone", 0.35);
    if (TIPI.osso.attivo) for (const p of persone) for (const [a, b] of OSSA) linea(p[a], p[b], "osso");
    if (TIPI.arma.attivo) for (const p of persone) for (const polso of [9, 10]) linea(arma, p[polso], "arma", 0.8);
    for (const p of persone) for (const [x, y] of p) crea("circle", { cx: x, cy: y, r: 4, fill: colore("--accento") }, svg);
    crea("rect", { x: arma[0] - 8, y: arma[1] - 8, width: 16, height: 16, fill: colore("--arma") }, svg);
  }
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", disegna);
  disegna();
})();

// --- 3. Dati su Roma e risultati: solo se i file esistono e contengono dati --------------------------------------
fetch("dati/roma.json").then((r) => r.json()).then((roma) => {
  const contenitore = document.getElementById("grafici-roma");
  if (!roma.serie || roma.serie.length === 0) return;  // fonti non ancora decise: resta il segnaposto
  // SEGNAPOSTO: il tipo di grafico si decide quando ci sono le fonti
  contenitore.textContent = `${roma.serie.length} serie di dati (fonte: ${roma.fonte})`;
}).catch(() => {});

fetch("dati/risultati.json").then((r) => r.json()).then((risultati) => {
  if (!risultati.modelli || risultati.modelli.length === 0) return;
  const righe = risultati.modelli.map((m) => `<tr><td style="text-align:left">${m.nome}</td><td>${m.macro_f1.toFixed(3)}</td><td>${m.latenza_cpu_ms.toFixed(1)}</td></tr>`).join("");
  document.getElementById("tabella-risultati").innerHTML =
    `<table><tr><th style="text-align:left">Modello</th><th>Macro-F1 (test)</th><th>ms/finestra (CPU)</th></tr>${righe}</table>`;
}).catch(() => {});
