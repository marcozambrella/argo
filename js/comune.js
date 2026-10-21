// Cose condivise dal sito: segnaposto dei testi di Marco, lettura dei colori del tema, movimento ridotto.

// Testi da scrivere: ogni riquadro dice cosa va scritto e quanto è lungo. Nessuna parola è di Claude.
export const SEGNAPOSTO = {
  titoloHero: "[TESTO DI MARCO — titolo, ~6 parole]",
  sottotitoloHero: "[TESTO DI MARCO — sottotitolo, ~20 parole]",
  cosaHoVisto: "[TESTO DI MARCO — cosa hai visto nel tuo quartiere, con chi hai parlato, ~120 parole]",
  introMappa: "[TESTO DI MARCO — introduzione ai dati su Roma, ~40 parole]",
  problemaArmi: "[TESTO DI MARCO — il problema delle armi che le statistiche non vedono, ~80 parole]",
  introPipeline: "[TESTO DI MARCO — introduzione a come funziona, ~40 parole]",
  chiLoUserebbe: "[TESTO DI MARCO — chi lo userebbe e come arriverebbe l'allarme, ~80 parole]",
  limiti: "[TESTO DI MARCO — cosa non fa ancora e cosa faresti dopo, ~80 parole]",
  strumenti: "[TESTO DI MARCO — come avete usato gli strumenti e l'AI, ~40 parole]",
  team: "[TESTO DI MARCO — team e scuola]",
};

export function riempiSegnaposto(radice = document) {
  for (const elemento of radice.querySelectorAll("[data-segnaposto]")) {
    elemento.textContent = SEGNAPOSTO[elemento.dataset.segnaposto] ?? "[TESTO DI MARCO]";
  }
}

export const coloreCss = (nome) => getComputedStyle(document.documentElement).getPropertyValue(nome).trim();
export const movimentoRidotto = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
export const formattaNumero = (numero) => Math.round(numero).toLocaleString("it-IT");

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
