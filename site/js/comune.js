// Cose condivise dal sito: segnaposto dei testi di Marco, lettura dei colori del tema, movimento ridotto.

// Testi da scrivere: ogni riquadro dice cosa va scritto e quanto è lungo. Nessuna parola è di Claude.
export const SEGNAPOSTO = {
  titoloHero: "[MARCO'S TEXT — title, ~6 words]",
  sottotitoloHero: "[MARCO'S TEXT — subtitle, ~20 words]",
  cosaHoVisto: "[MARCO'S TEXT — what you saw in your neighbourhood, who you talked to, ~120 words]",
  introMappa: "[MARCO'S TEXT — introduction to the Rome data, ~40 words]",
  problemaArmi: "[MARCO'S TEXT — the weapons problem the statistics do not see, ~80 words]",
  introPipeline: "[MARCO'S TEXT — introduction to how it works, ~40 words]",
  chiLoUserebbe: "[MARCO'S TEXT — who would use it and how the alarm would reach them, ~80 words]",
  limiti: "[MARCO'S TEXT — what it does not do yet and what you would do next, ~80 words]",
  strumenti: "[MARCO'S TEXT — how you used the tools and AI, ~40 words]",
  team: "[MARCO'S TEXT — team and school]",
};

export function riempiSegnaposto(radice = document) {
  for (const elemento of radice.querySelectorAll("[data-segnaposto]")) {
    elemento.textContent = SEGNAPOSTO[elemento.dataset.segnaposto] ?? "[MARCO'S TEXT]";
  }
}

export const coloreCss = (nome) => getComputedStyle(document.documentElement).getPropertyValue(nome).trim();
export const movimentoRidotto = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
export const formattaNumero = (numero) => Math.round(numero).toLocaleString("en-GB");

// Generatore pseudo-casuale con seme (mulberry32): stesse posizioni a ogni caricamento
export function casualeConSeme(seme) {
  let stato = 0;
  for (const carattere of String(seme)) stato = (stato * 31 + carattere.charCodeAt(0)) >>> 0;
  return () => {
    stato = (stato + 0x6d2b79f5) >>> 0;
    let t = stato;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Avvia una funzione solo quando l'elemento si avvicina allo schermo (le scene 3D sono pesanti)
export function quandoVisibile(elemento, funzione) {
  const osservatore = new IntersectionObserver((voci) => {
    if (voci.some((voce) => voce.isIntersecting)) { osservatore.disconnect(); funzione(); }
  }, { rootMargin: "300px" });
  osservatore.observe(elemento);
}

export function webglDisponibile() {
  try {
    const tela = document.createElement("canvas");
    return Boolean(tela.getContext("webgl2") || tela.getContext("webgl"));
  } catch { return false; }
}
