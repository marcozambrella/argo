import { FINESTRE, NUMERO_FINESTRE, PASSI_PIPELINE, formattaSecondi, riempiSegnaposto, statoFinestra } from "../condivisi/progetto.js";

riempiSegnaposto();

// Algoritmo 1: i passi della pipeline come pseudocodice
const elenco = document.getElementById("passi-algoritmo");
for (const passo of PASSI_PIPELINE) {
  const voce = document.createElement("li");
  voce.innerHTML = `<span class="chiave">${passo.nome}</span>: ${passo.dettaglio}. <span class="tecnica">${passo.tecnica}</span>`;
  elenco.appendChild(voce);
}

// Figura 2: grafico con asse dei frame e una riga per ogni finestra
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("grafico-finestre");
const cursore = document.getElementById("cursore-finestre");
cursore.max = NUMERO_FINESTRE - 1;
const SINISTRA = 46, LARGHEZZA = 400, ALTO = 12, ALTEZZA_RIGA = 12;
const x = (frame) => SINISTRA + (frame / FINESTRE.frameTotaliMostrati) * LARGHEZZA;
const crea = (nome, attributi, testo) => {
  const elemento = document.createElementNS(NS, nome);
  for (const [chiave, valore] of Object.entries(attributi)) elemento.setAttribute(chiave, valore);
  if (testo !== undefined) elemento.textContent = testo;
  svg.appendChild(elemento);
  return elemento;
};
const coloreInchiostro = () => getComputedStyle(document.documentElement).getPropertyValue("--inchiostro").trim();

function disegna() {
  svg.replaceChildren();
  const stato = statoFinestra(Number(cursore.value));
  const inchiostro = coloreInchiostro();
  const fondo = ALTO + NUMERO_FINESTRE * ALTEZZA_RIGA + 6;
  // asse x con tacche ogni 40 frame e secondi
  crea("line", { x1: SINISTRA, y1: fondo, x2: SINISTRA + LARGHEZZA, y2: fondo, stroke: inchiostro });
  for (let frame = 0; frame <= FINESTRE.frameTotaliMostrati; frame += 20) {
    crea("line", { x1: x(frame), y1: fondo, x2: x(frame), y2: fondo + 4, stroke: inchiostro });
    crea("text", { x: x(frame), y: fondo + 16, "font-size": 10, "text-anchor": "middle" }, frame);
  }
  crea("text", { x: SINISTRA + LARGHEZZA / 2, y: fondo + 32, "font-size": 11, "text-anchor": "middle", "font-style": "italic" }, "frame");
  crea("text", { x: 12, y: ALTO + (NUMERO_FINESTRE * ALTEZZA_RIGA) / 2, "font-size": 11, "font-style": "italic",
    transform: `rotate(-90 12 ${ALTO + (NUMERO_FINESTRE * ALTEZZA_RIGA) / 2})`, "text-anchor": "middle" }, "finestra");
  for (let indice = 0; indice < NUMERO_FINESTRE; indice++) {
    const finestra = statoFinestra(indice);
    const selezionata = indice === stato.indice;
    crea("rect", { x: x(finestra.primoFrame), y: ALTO + indice * ALTEZZA_RIGA, width: x(FINESTRE.framePerFinestra) - SINISTRA,
      height: ALTEZZA_RIGA - 4, fill: selezionata ? inchiostro : "none", stroke: inchiostro, "stroke-width": selezionata ? 1 : 0.6,
      opacity: selezionata ? 1 : 0.55 });
    crea("text", { x: SINISTRA - 6, y: ALTO + indice * ALTEZZA_RIGA + 7, "font-size": 8, "text-anchor": "end" }, indice + 1);
  }
  // fascia verticale: i frame della finestra selezionata, coperti da più finestre
  crea("rect", { x: x(stato.primoFrame), y: 4, width: x(FINESTRE.framePerFinestra) - SINISTRA, height: fondo - 4,
    fill: inchiostro, opacity: 0.06 });
  document.getElementById("numero-finestra").textContent = stato.indice + 1;
  document.getElementById("descrizione-finestra").textContent =
    `frame ${stato.primoFrame}–${stato.ultimoFrame}, ${formattaSecondi(stato.secondoInizio)}–${formattaSecondi(stato.secondoFine)} s`;
}
cursore.addEventListener("input", disegna);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", disegna);
disegna();
