// I 9 passi della spiegazione "How it works": testo tecnico fattuale (dal codice) + il testo di ogni passo (scritto con
// Marco, 23/10; aggiornati alla v1.2 il 04/11: niente detector, le reti video b1 e b2).
// Separati dalla scena 3D così l'elenco esiste anche senza WebGL.
export const PASSI = [
  { chiave: "fotogramma", testo: "Everything starts from an ordinary CCTV frame, read at 25 frames per second. No special camera, no extra hardware: ARGO plugs into the streams a city already has.", titolo: "The real frame", tecnica: "Clip Pexels #31524454 · 1280×720 · read at 25 fps",
    didascalia: "A real frame: the coloured points are the real output of YOLO26s-pose." },
  { chiave: "punti", testo: "YOLO26s-pose finds every person and 17 body points. A tracker keeps the same ID on each person across frames, so ARGO follows individuals, not isolated detections.", titolo: "17 points, 4 people", tecnica: "YOLO26s-pose at 640 px: 17 COCO keypoints · TrackTrack: stable IDs 1–4",
    didascalia: "The points lift off the image: 4 people × 17 points = 68 nodes. The colour is the tracker ID." },
  { chiave: "manichino", testo: "Points become skeletons. Working on skeletons makes ARGO lighter, less sensitive to light and clothing, and privacy-friendly: the analysis never needs a face.", titolo: "From points to mannequins", tecnica: "Bones of the COCO skeleton (19 segments per person)",
    didascalia: "3D mannequins built on the real points. The depth is illustrative: the network only sees x, y and confidence.",
    onesta: "The mannequin depth does not exist in the data: the network works in 2D." },
  { chiave: "pixel", testo: "Skeletons do not see what a hand holds. So two video networks look at the pixels of the same 1.6 seconds: a light one often, a sharper one every few seconds.", titolo: "The pixels too", tecnica: "R(2+1)D-18 · b1: 112 px, every 0.8 s · b2: 224 px, every 4 s · 16 of the 40 frames",
    didascalia: "What the two video networks see: 16 frames of the window, cut to the centre of the image. b1 is small enough to run every 0.8 s; b2 sees four times the pixels, enough for a weapon." },
  { chiave: "grafo", testo: "Each skeleton is a graph: joints are nodes, bones are links. The network reads every joint together with the joints it is attached to, the way a body actually moves.", titolo: "One graph per skeleton", tecnica: "17 joints linked by 19 bones · every person goes through the same network", comandi: "archi",
    didascalia: "The ST-GCN works on the graph of the bones: each joint gathers information from the joints it is connected to. Switch on the all-to-all links to see the Graph-Transformer we tested and discarded (macro-F1 0.29 against 0.40)." },
  { chiave: "pila", testo: "A single photo says little. ARGO judges 40 frames, 1.6 seconds, at a time, and opens a new window every 0.4 seconds: more than two decisions per second.", titolo: "40 frames together", tecnica: "40-frame window (1.6 s), a new one every 10", comandi: "finestra",
    didascalia: "160 real frames lined up in time. The lit window is what the network looks at in one go." },
  { chiave: "tempo", testo: "For every joint, the network follows the movement across nearby frames. Ten stacked blocks combine space and time until a raised arm becomes a punch, or a grab.", titolo: "Convolution over time", tecnica: "Each joint looks at itself in 9 nearby frames · 10 blocks, each one bones then time",
    didascalia: "The trails are the real movements of each joint over the 40 frames. The network slides a 9-frame filter along each trail; stacking 10 blocks, it sees the whole 1.6 s." },
  { chiave: "classi", testo: "The skeletons and b1 vote on violence: their average decides. b1 also says whether someone is snatching. Seven violent windows in a row, about four seconds, raise the VIOLENCE ALARM.", titolo: "The decision", tecnica: "violence = average of ST-GCN and b1 · snatching from b1 · armed violence = violence + WEAPON WARNING", comandi: "classi",
    didascalia: "The most suspicious person in the window counts. These are the real outputs of ARGO for the chosen window: in this alley it stays normal in all 13 windows." },
  { chiave: "warning", testo: "Weapons are b2's job. Every 4 seconds it says whether it sees one, and above a cautious threshold it raises the WEAPON WARNING on its own, so a visible gun is never averaged away.", titolo: "WEAPON WARNING", tecnica: "b2 weapon probability ≥ 0.35 (threshold chosen on validation), every 4 s", comandi: "warning",
    didascalia: "Independent of the class. The bar starts at b2's real answer on this alley; move it to see when the warning fires.",
    onesta: "Moving the bar is a simulation, not an output of the network." },
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
