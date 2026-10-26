// I 9 passi della spiegazione "How it works": testo tecnico fattuale (dal codice) + il testo di ogni passo (scritto con Marco, 23/10).
// Separati dalla scena 3D così l'elenco esiste anche senza WebGL.
export const PASSI = [
  { chiave: "fotogramma", testo: "Everything starts from an ordinary CCTV frame, read at 25 frames per second. No special camera, no extra hardware: ARGO plugs into the streams a city already has.", titolo: "The real frame", tecnica: "Clip Pexels #31524454 · 1280×720 · read at 25 fps",
    didascalia: "A real frame: the coloured points are the real output of YOLO26s-pose." },
  { chiave: "punti", testo: "YOLO26s-pose finds every person and 17 body points. A tracker keeps the same ID on each person across frames, so ARGO follows individuals, not isolated detections.", titolo: "17 points, 4 people", tecnica: "YOLO26s-pose: 17 COCO keypoints · TrackTrack: stable IDs 1–4",
    didascalia: "The points lift off the image: 4 people × 17 points = 68 nodes. The colour is the tracker ID." },
  { chiave: "manichino", testo: "Points become skeletons. Working on skeletons makes ARGO lighter, less sensitive to light and clothing, and privacy-friendly: the analysis never needs a face.", titolo: "From points to mannequins", tecnica: "Bones of the COCO skeleton (19 segments per person)",
    didascalia: "3D mannequins built on the real points. The depth is illustrative: the network only sees x, y and confidence.",
    onesta: "The mannequin depth does not exist in the data: the network works in 2D." },
  { chiave: "armi", testo: "In parallel, a YOLO26s detector trained on CCTV images looks for pistols, knives and rifles. Up to two weapons per frame enter the model through their own branch.", titolo: "Two slots for weapons", tecnica: "YOLO26s trained for this: pistol, knife, rifle",
    didascalia: "What the detector does: it finds the weapon and draws a box around it. Up to 2 weapons per frame go into a small separate branch that joins the skeletons just before the decision.",
    onesta: "Illustration: this clip has no weapon, the pistol is a 3D model placed in a hand." },
  { chiave: "grafo", testo: "Each skeleton is a graph: joints are nodes, bones are links. The network reads every joint together with the joints it is attached to, the way a body actually moves.", titolo: "One graph per skeleton", tecnica: "17 joints linked by 19 bones · every person goes through the same network", comandi: "archi",
    didascalia: "The ST-GCN works on the graph of the bones: each joint gathers information from the joints it is connected to. Switch on the all-to-all links to see the Graph-Transformer we tested and discarded (macro-F1 0.29 against 0.40)." },
  { chiave: "pila", testo: "A single photo says little. ARGO judges 40 frames, 1.6 seconds, at a time, and opens a new window every 0.4 seconds: more than two decisions per second.", titolo: "40 frames together", tecnica: "40-frame window (1.6 s), a new one every 10", comandi: "finestra",
    didascalia: "160 real frames lined up in time. The lit window is what the network looks at in one go." },
  { chiave: "tempo", testo: "For every joint, the network follows the movement across nearby frames. Ten stacked blocks combine space and time until a raised arm becomes a punch, or a grab.", titolo: "Convolution over time", tecnica: "Each joint looks at itself in 9 nearby frames · 10 blocks, each one bones then time",
    didascalia: "The trails are the real movements of each joint over the 40 frames. The network slides a 9-frame filter along each trail; stacking 10 blocks, it sees the whole 1.6 s." },
  { chiave: "classi", testo: "Skeletons and a second network reading the pixels vote on violence; the skeletons then name the class. Seven violent windows in a row, about four seconds, raise the VIOLENCE ALARM.", titolo: "The decision", tecnica: "ST-GCN on the skeletons: 4 classes · R(2+1)D-18 on the pixels: violence yes/no · violence = the average of the two", comandi: "classi",
    didascalia: "The most suspicious person in the window counts. Skeletons and pixels vote on violence; the skeletons then pick the class. These are the real probabilities for the chosen window: in this alley it is normal (about 80%) in all 13 windows." },
  { chiave: "warning", testo: "A separate, deliberately cautious rule: a weapon near a wrist in at least 5 of 40 frames raises the WEAPON WARNING on its own, so a visible gun is never averaged away.", titolo: "WEAPON WARNING", tecnica: "Weapon with confidence ≥ 0.7 near a wrist in ≥ 5 of 40 frames (thresholds tuned on validation)", comandi: "warning",
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
      <p class="testo-passo">${passo.testo}</p>
      <span class="tecnica">${passo.tecnica}</span>${passo.onesta ? `<p class="onesta">${passo.onesta}</p>` : ""}</div>`;
    elenco.appendChild(voce);
  });
}
