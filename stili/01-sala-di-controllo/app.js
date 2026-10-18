import { FINESTRE, NUMERO_FINESTRE, PASSI_PIPELINE, formattaSecondi, riempiSegnaposto, statoFinestra } from "../condivisi/progetto.js";

riempiSegnaposto();

// Contatore di sistema nella barra in alto (solo decorativo: frame trascorsi a 25 fps dall'apertura della pagina)
const contatore = document.getElementById("contatore-sistema");
const inizio = performance.now();
const aggiornaContatore = () => {
  contatore.textContent = String(Math.floor(((performance.now() - inizio) / 1000) * FINESTRE.fps)).padStart(6, "0");
  requestAnimationFrame(aggiornaContatore);
};
if (!matchMedia("(prefers-reduced-motion: reduce)").matches) aggiornaContatore();

// Pipeline: un modulo per passo, che si accende quando entra nello schermo
const catena = document.getElementById("catena-pipeline");
for (const passo of PASSI_PIPELINE) {
  const elemento = document.createElement("li");
  elemento.style.setProperty("--passo", passo.numero);
  if (passo.nome === "WARNING ARMA") elemento.classList.add("allarme");
  elemento.innerHTML = `<span class="num">NODO ${String(passo.numero).padStart(2, "0")}</span>
    <span class="nome">${passo.nome}</span><span class="dettaglio">${passo.dettaglio}</span>
    <span class="tecnica">${passo.tecnica}</span>`;
  catena.appendChild(elemento);
}
const osservatore = new IntersectionObserver((voci) => {
  for (const voce of voci) {
    if (!voce.isIntersecting) continue;
    [...catena.children].forEach((figlio, indice) => setTimeout(() => figlio.classList.add("visibile"), indice * 110));
    osservatore.disconnect();
  }
}, { threshold: 0.2 });
osservatore.observe(catena);

// Finestre scorrevoli: nastro di 160 frame e corsie con tutte le finestre sovrapposte
const nastro = document.getElementById("nastro-frame");
const corsie = document.getElementById("corsie-finestre");
const cursore = document.getElementById("cursore-finestre");
cursore.max = NUMERO_FINESTRE - 1;
const celle = Array.from({ length: FINESTRE.frameTotaliMostrati }, (_, frame) => {
  const cella = document.createElement("span");
  if (frame % FINESTRE.passo === 0) cella.classList.add("decina");
  nastro.appendChild(cella);
  return cella;
});
const barre = Array.from({ length: NUMERO_FINESTRE }, (_, indice) => {
  const barra = document.createElement("span");
  const stato = statoFinestra(indice);
  barra.style.left = `${(stato.primoFrame / FINESTRE.frameTotaliMostrati) * 100}%`;
  barra.style.width = `${(FINESTRE.framePerFinestra / FINESTRE.frameTotaliMostrati) * 100}%`;
  barra.style.top = `${(indice % stato.finestrePerFrame) * 18}px`;
  corsie.appendChild(barra);
  return barra;
});

function disegnaFinestra() {
  const stato = statoFinestra(Number(cursore.value));
  celle.forEach((cella, frame) => cella.classList.toggle("dentro", frame >= stato.primoFrame && frame <= stato.ultimoFrame));
  barre.forEach((barra, indice) => barra.classList.toggle("attiva", indice === stato.indice));
  document.getElementById("numero-finestra").textContent = String(stato.indice + 1).padStart(2, "0");
  document.getElementById("frame-finestra").textContent = `${stato.primoFrame}–${stato.ultimoFrame}`;
  document.getElementById("secondi-finestra").textContent = `${formattaSecondi(stato.secondoInizio)}–${formattaSecondi(stato.secondoFine)} s`;
}
cursore.addEventListener("input", disegnaFinestra);
disegnaFinestra();
