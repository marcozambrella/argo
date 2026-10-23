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
};

export const CASI: readonly CasoDUso[] = [
  { nome: "metro_folla", inizio: 3, capitolo: "METRO", titolo: "CROWD", adatta: "cover",
    credito: "VIDEO: Footage 4, CC BY · youtube.com/watch?v=tCYc33IfUWs" },
  { nome: "arma_rapina_milton", inizio: 1.5, capitolo: "ROBBERY", titolo: "ARMED", adatta: "contain",
    credito: "VIDEO: Queensland Police, CC BY · youtube.com/watch?v=A0eEbRnCfSA" },
  { nome: "scippo_strada_delhi", inizio: 12.5, capitolo: "STREET", titolo: "SNATCH AND FIGHT", adatta: "contain",
    credito: "VIDEO: ETV Andhra Pradesh, CC BY · youtube.com/watch?v=O2Ex648qVdk" },
  { nome: "strada_roma_vicolo", inizio: 4, capitolo: "ROME", titolo: "ALLEY", adatta: "cover",
    credito: "VIDEO: Pexels #31524454" },
];

// Da rimettere fra i casi quando il disco esterno con data/demo è collegato (runs/argo_casi_uso/lancia.sh):
export const CASO_RISSA_SCHELETRI: CasoDUso = {
  nome: "dataset_scheletri_violenza_fight_0978", inizio: 4, capitolo: "FIGHT", titolo: "SKELETONS ONLY", adatta: "contain",
  credito: "RWF-2000 (academic use only): skeletons shown without the video",
};
