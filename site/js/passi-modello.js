// I 9 passi della spiegazione "Come funziona": testo tecnico fattuale (dal codice) + segnaposto per i testi di Marco.
// Separati dalla scena 3D così l'elenco esiste anche senza WebGL.
export const PASSI = [
  { chiave: "fotogramma", titolo: "Il fotogramma vero", tecnica: "Clip Pexels #31524454 · 1280×720 · letta a 25 fps",
    didascalia: "Un fotogramma reale: i punti colorati sono l'uscita vera di YOLO26s-pose." },
  { chiave: "punti", titolo: "17 punti, 4 persone", tecnica: "YOLO26s-pose: 17 keypoint COCO · TrackTrack: ID 1–4 stabili",
    didascalia: "I punti si staccano dall'immagine: 4 persone × 17 punti = 68 nodi. Il colore è l'ID del tracker." },
  { chiave: "manichino", titolo: "Dal punto al manichino", tecnica: "Ossa dello scheletro COCO (19 segmenti per persona)",
    didascalia: "Manichini 3D costruiti sui punti veri. La profondità è illustrativa: la rete vede solo x, y e la confidenza.",
    onesta: "La profondità del manichino non esiste nei dati: la rete lavora in 2D." },
  { chiave: "armi", titolo: "Due posti per le armi", tecnica: "YOLO26s addestrato apposta: pistola, coltello, fucile",
    didascalia: "Il grafo ha sempre 2 nodi arma. In questa clip non c'è un'arma: restano vuoti (mascherati).",
    onesta: "Clip senza armi: i nodi arma sono vuoti." },
  { chiave: "grafo", titolo: "Il grafo", tecnica: "70 nodi per frame · 7 tipi di arco", comandi: "archi",
    didascalia: "Ogni nodo vede tutti gli altri; il tipo di arco cambia il peso. Accendi e spegni i tipi." },
  { chiave: "pila", titolo: "40 frame insieme", tecnica: "Finestra di 40 frame (1,6 s), una nuova ogni 10", comandi: "finestra",
    didascalia: "160 frame veri messi in fila nel tempo. La finestra illuminata è quella che la rete guarda in una volta." },
  { chiave: "tempo", titolo: "Attenzione nel tempo", tecnica: "Ogni nodo guarda se stesso nei 40 frame",
    didascalia: "Le scie sono i movimenti veri di ogni punto nei 40 frame della finestra." },
  { chiave: "classi", titolo: "La decisione", tecnica: "ST-GCN (scelto: macro-F1 0,40 contro 0,29 del Graph-Transformer) + rete video R(2+1)D-18 sui pixel per la violenza", comandi: "classi",
    didascalia: "Quattro classi, con le probabilità vere del sistema per la finestra scelta: in questo vicolo è normale (circa 80%) in tutte le 13 finestre." },
  { chiave: "warning", titolo: "WARNING ARMA", tecnica: "Arma con confidenza ≥ 0,7 vicino a un polso in ≥ 5 frame su 40 (soglie tarate sul validation)", comandi: "warning",
    didascalia: "Regola indipendente dal classificatore. Simulazione: sposta il cursore per vedere quando scatta.",
    onesta: "Simulazione della regola, non un'uscita della rete." },
];

export function creaElencoPassi() {
  const elenco = document.getElementById("passi-modello");
  PASSI.forEach((passo, indice) => {
    const voce = document.createElement("li");
    voce.className = "passo-modello";
    voce.dataset.chiave = passo.chiave;
    voce.innerHTML = `<div><span class="numero">${String(indice + 1).padStart(2, "0")} / ${PASSI.length}</span><h3>${passo.titolo}</h3>
      <span class="tecnica">${passo.tecnica}</span>${passo.onesta ? `<p class="onesta">${passo.onesta}</p>` : ""}</div>`;
    elenco.appendChild(voce);
  });
}
