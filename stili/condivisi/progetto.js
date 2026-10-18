// Dati del progetto condivisi dalle demo di stile. SOLO fatti presi da .claude/PANORAMICA.md e src/sicurezza/:
// niente testi descrittivi (li scrive Marco) e niente numeri inventati.

export const PASSI_PIPELINE = [
  { numero: 1, nome: "Video", dettaglio: "25 fotogrammi al secondo dalla telecamera", tecnica: "ffmpeg · 25 fps" },
  { numero: 2, nome: "Persone e corpo", dettaglio: "17 punti del corpo per persona, ID stabile nel tempo", tecnica: "YOLO26s-pose · TrackTrack" },
  { numero: 3, nome: "Armi", dettaglio: "pistola, coltello, fucile — non i coltelli da cucina", tecnica: "YOLO26s addestrato apposta" },
  { numero: 4, nome: "Finestre", dettaglio: "gli ultimi 40 frame (1,6 s), una nuova ogni 10 frame", tecnica: "40 frame · passo 10" },
  { numero: 5, nome: "Grafo", dettaglio: "4 persone × 17 punti + 2 nodi arma; archi: ossa, persone diverse, arma–polso", tecnica: "70 nodi" },
  { numero: 6, nome: "Rete neurale a grafo", dettaglio: "normale · violenza · violenza con arma · scippo", tecnica: "Graph-Transformer vs ST-GCN" },
  { numero: 7, nome: "WARNING ARMA", dettaglio: "arma vicino a una persona in almeno 5 frame su 40, indipendente dal classificatore", tecnica: "regola · ≥ 5/40" },
];

export const CLASSI = ["normale", "violenza", "violenza con arma", "scippo"];

export const FINESTRE = { fps: 25, framePerFinestra: 40, passo: 10, frameTotaliMostrati: 160 };

// Stato della finestra n. `indice` (da 0): primo e ultimo frame, secondi, finestre che si sovrappongono
export function statoFinestra(indice) {
  const { fps, framePerFinestra, passo } = FINESTRE;
  const primoFrame = indice * passo;
  const ultimoFrame = primoFrame + framePerFinestra - 1;
  return {
    indice,
    primoFrame,
    ultimoFrame,
    secondoInizio: primoFrame / fps,
    secondoFine: (ultimoFrame + 1) / fps,
    // quante finestre contengono ciascun frame (al massimo 4: 40 / 10)
    finestrePerFrame: framePerFinestra / passo,
  };
}

export const NUMERO_FINESTRE = (FINESTRE.frameTotaliMostrati - FINESTRE.framePerFinestra) / FINESTRE.passo + 1;

export const formattaSecondi = (secondi) => secondi.toFixed(1).replace(".", ",");

// Segnaposto dei testi di Marco: ogni demo li mostra come riquadri dichiarati, con la lunghezza prevista
export const SEGNAPOSTO = {
  titoloHero: "[TESTO DI MARCO — titolo, ~6 parole]",
  sottotitoloHero: "[TESTO DI MARCO — sottotitolo, ~20 parole]",
  introPipeline: "[TESTO DI MARCO — introduzione a come funziona, ~40 parole]",
  introFinestre: "[TESTO DI MARCO — perché guardare 40 frame insieme, ~40 parole]",
};

// Riempie ogni elemento con data-segnaposto="chiave" con il riquadro dichiarato corrispondente
export function riempiSegnaposto(radice = document) {
  for (const elemento of radice.querySelectorAll("[data-segnaposto]")) {
    elemento.textContent = SEGNAPOSTO[elemento.dataset.segnaposto] ?? "[TESTO DI MARCO]";
  }
}
