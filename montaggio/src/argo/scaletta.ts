// Scaletta del pezzo di apertura di Argo (D61): ~45 s, apertura brutalista con le città e i casi d'uso.
// I numeri e le fonti sono in presentation/fonti_apertura.md (da approvare da Marco).
// I casi d'uso sono le uscite VERE di Argo (runs/argo_casi_uso/<nome>: video_annotato.mp4 + eventi.json), copiate in
// public/argo/ da strumenti/prepara_argo.sh. Il pannello a destra si disegna dagli eventi, non è inventato.

export type Cartello = {
  readonly citta: string;
  readonly numero: string;   // come appare: "71.867" (tutti numeri assoluti di un anno, D61)
  readonly testo: string;
  readonly fonte: string;
};

export const CARTELLI: readonly Cartello[] = [
  { citta: "LONDON", numero: "71,867", testo: "phones stolen or snatched in 2025",
    fonte: "Metropolitan Police, via London City Hall (October 2026)" },
  { citta: "PARIS", numero: "11,095", testo: "victims of physical violence outside the home in 2025",
    fonte: "SSMSI, French Ministry of the Interior: 2025 department data, Paris" },
  { citta: "NEW YORK", numero: "688", testo: "shooting incidents in 2025",
    fonte: "NYPD, January 2026" },
  { citta: "MILAN", numero: "2,213", testo: "snatch thefts reported in 2025",
    fonte: "Italian Ministry of the Interior, via Il Sole 24 Ore Crime Index 2026" },
  { citta: "ROME", numero: "4,659", testo: "reports of intentional assault injuries in 2025",
    fonte: "Italian Ministry of the Interior, via Il Sole 24 Ore Crime Index 2026" },
];

export type CasoDUso = {
  readonly nome: string;          // public/argo/<nome>.mp4 e .json
  readonly inizio: number;        // secondo della clip da cui partire
  readonly capitolo: string;      // es. "METRO"
  readonly titolo: string;        // es. "FOLLA"
  readonly credito: string;
  readonly adatta: "cover" | "contain";
  readonly durata?: number;       // secondi sullo schermo (vetrina; nell'apertura decide la musica)
  // Scheletri piccoli: il punto (cx, cy), in frazioni del riquadro, va verso il centro e si ingrandisce di "scala"
  readonly zoom?: { readonly scala: number; readonly cx: number; readonly cy: number };
};

export const CASI: readonly CasoDUso[] = [
  { nome: "metro_folla", inizio: 3, capitolo: "METRO", titolo: "CROWD", adatta: "cover",
    credito: "VIDEO: Footage 4, CC BY · youtube.com/watch?v=tCYc33IfUWs" },
  { nome: "arma_rapina_milton", inizio: 1.5, capitolo: "ROBBERY", titolo: "ARMED", adatta: "contain",
    credito: "VIDEO: Queensland Police, CC BY · youtube.com/watch?v=A0eEbRnCfSA" },
  { nome: "scippo_strada_delhi", inizio: 12.5, capitolo: "STREET", titolo: "SNATCH AND FIGHT", adatta: "contain",
    credito: "VIDEO: ETV Andhra Pradesh, CC BY · youtube.com/watch?v=O2Ex648qVdk" },
  // Video del dataset (RWF-2000, solo uso accademico): si mostrano solo gli scheletri su fondo nero (D58)
  { nome: "dataset_scheletri_violenza_fight_0978", inizio: 2, capitolo: "FIGHT", titolo: "SKELETONS ONLY", adatta: "contain",
    credito: "RWF-2000 (academic use only): skeletons shown without the video" },
  { nome: "strada_roma_vicolo", inizio: 4, capitolo: "ROME", titolo: "ALLEY", adatta: "cover",
    credito: "VIDEO: Pexels #31524454" },
];

// Vetrina per l'hero del sito (Marco, 23/10): tutti i casi d'uso, senza musica e senza città, i normali per ultimi.
// Muta e in loop: dopo ROME si ricomincia dalla rapina.
export const CASI_VETRINA: readonly CasoDUso[] = [
  { nome: "arma_rapina_milton", inizio: 1.5, capitolo: "ROBBERY", titolo: "ARMED", adatta: "contain",
    credito: "VIDEO: Queensland Police, CC BY · youtube.com/watch?v=A0eEbRnCfSA" },
  { nome: "dataset_scheletri_violenza_fight_0978", inizio: 2, capitolo: "FIGHT", titolo: "SKELETONS ONLY", adatta: "contain",
    credito: "RWF-2000 (academic use only): skeletons shown without the video" },
  { nome: "dataset_scheletri_arma_w001", inizio: 0, durata: 4.8, capitolo: "WEAPON", titolo: "SKELETONS ONLY", adatta: "cover",
    credito: "SCVD dataset (research use): skeletons shown without the video", zoom: { scala: 1.8, cx: 0.523, cy: 0.318 } },
  { nome: "dataset_scheletri_scippo_Chain_Snatching66", inizio: 2, capitolo: "SNATCHING", titolo: "SKELETONS ONLY", adatta: "contain",
    credito: "Chain-snatching CCTV dataset (Kaggle): skeletons shown without the video", zoom: { scala: 1.8, cx: 0.8, cy: 0.7 } },
  { nome: "scippo_strada_delhi", inizio: 12.5, capitolo: "STREET", titolo: "SNATCH AND FIGHT", adatta: "contain",
    credito: "VIDEO: ETV Andhra Pradesh, CC BY · youtube.com/watch?v=O2Ex648qVdk" },
  { nome: "metro_folla", inizio: 3, capitolo: "METRO", titolo: "CROWD", adatta: "cover",
    credito: "VIDEO: Footage 4, CC BY · youtube.com/watch?v=tCYc33IfUWs" },
  { nome: "strada_roma_vicolo", inizio: 4, capitolo: "ROME", titolo: "ALLEY", adatta: "cover",
    credito: "VIDEO: Pexels #31524454" },
];
export const SECONDI_VETRINA = 5;
