// Tema parametrico dell'interfaccia (HUD) del video hero.
// SEGNAPOSTO DESIGN: valori della prima proposta "sala di controllo"; quando Marco sceglie lo stile del sito si
// renderizza una variante cambiando solo questi valori (colori e font), senza toccare le scene.
import { loadFont as caricaShareTechMono } from "@remotion/google-fonts/ShareTechMono";
import { loadFont as caricaRajdhani } from "@remotion/google-fonts/Rajdhani";

const { fontFamily: fontMonospazio } = caricaShareTechMono("normal", { weights: ["400"], subsets: ["latin"] });
const { fontFamily: fontTitoli } = caricaRajdhani("normal", { weights: ["500", "700"], subsets: ["latin"] });

export type Tema = {
  readonly coloreLinee: string;      // box, scheletri, cornici
  readonly coloreAccento: string;    // ID, finestra attiva
  readonly coloreAllarme: string;    // armi, avvisi
  readonly coloreTesto: string;
  readonly sfondoPannelli: string;
  readonly fontMonospazio: string;
  readonly fontTitoli: string;
};

export const TEMA_SALA_DI_CONTROLLO: Tema = {
  coloreLinee: "#e8f4f8",
  coloreAccento: "#38d6f5",
  coloreAllarme: "#ff3b3b",
  coloreTesto: "#e8f4f8",
  sfondoPannelli: "rgba(4, 10, 14, 0.72)",
  fontMonospazio,
  fontTitoli,
};

// Scheletro COCO a 17 punti: stesse ossa di src/sicurezza/dati.py (OSSA)
export const OSSA: ReadonlyArray<readonly [number, number]> = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
  [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
];

// Parametri veri della pipeline (src/sicurezza/dati.py)
export const FRAME_PER_FINESTRA = 40;
export const PASSO_FINESTRE = 10;
export const SOGLIA_VISIBILITA_KEYPOINT = 0.3;  // come STILE_OVERLAY in src/sicurezza/demo.py
export const SOGLIA_CONFIDENZA_ARMA = 0.5;      // come SOGLIA_CONFIDENZA_ARMA in src/sicurezza/demo.py
