// I 9 passi della spiegazione "How it works": testo tecnico fattuale (dal codice) + segnaposto per i testi di Marco.
// Separati dalla scena 3D così l'elenco esiste anche senza WebGL.
export const PASSI = [
  { chiave: "fotogramma", titolo: "The real frame", tecnica: "Clip Pexels #31524454 · 1280×720 · read at 25 fps",
    didascalia: "A real frame: the coloured points are the real output of YOLO26s-pose." },
  { chiave: "punti", titolo: "17 points, 4 people", tecnica: "YOLO26s-pose: 17 COCO keypoints · TrackTrack: stable IDs 1–4",
    didascalia: "The points lift off the image: 4 people × 17 points = 68 nodes. The colour is the tracker ID." },
  { chiave: "manichino", titolo: "From points to mannequins", tecnica: "Bones of the COCO skeleton (19 segments per person)",
    didascalia: "3D mannequins built on the real points. The depth is illustrative: the network only sees x, y and confidence.",
    onesta: "The mannequin depth does not exist in the data: the network works in 2D." },
  { chiave: "armi", titolo: "Two slots for weapons", tecnica: "YOLO26s trained for this: pistol, knife, rifle",
    didascalia: "The graph always has 2 weapon nodes. There is no weapon in this clip: they stay empty (masked).",
    onesta: "Clip without weapons: the weapon nodes are empty." },
  { chiave: "grafo", titolo: "The graph", tecnica: "70 nodes per frame · 7 edge types", comandi: "archi",
    didascalia: "Every node sees all the others; the edge type changes the weight. Switch the types on and off." },
  { chiave: "pila", titolo: "40 frames together", tecnica: "40-frame window (1.6 s), a new one every 10", comandi: "finestra",
    didascalia: "160 real frames lined up in time. The lit window is what the network looks at in one go." },
  { chiave: "tempo", titolo: "Attention over time", tecnica: "Each node looks at itself across the 40 frames",
    didascalia: "The trails are the real movements of each point over the 40 frames of the window." },
  { chiave: "classi", titolo: "The decision", tecnica: "ST-GCN (chosen: macro-F1 0.40 against 0.29 for the Graph-Transformer) + R(2+1)D-18 video network on the pixels for violence", comandi: "classi",
    didascalia: "Four classes, with the real probabilities of the system for the chosen window: in this alley it is normal (about 80%) in all 13 windows." },
  { chiave: "warning", titolo: "WEAPON WARNING", tecnica: "Weapon with confidence ≥ 0.7 near a wrist in ≥ 5 of 40 frames (thresholds tuned on validation)", comandi: "warning",
    didascalia: "A rule independent of the classifier. Simulation: move the slider to see when it fires.",
    onesta: "A simulation of the rule, not an output of the network." },
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
