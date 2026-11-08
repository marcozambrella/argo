// I 9 passi della spiegazione "How it works": testo tecnico fattuale (dal codice) + il testo di ogni passo (scritto con
// Marco, 23/10; aggiornati alla v1.2 il 04/11: niente detector; alla v1.3 l'08/11: CLIP al posto di b2 per armi e scippo,
// e la scena è la rapina armata di Sunnybank, perché sul vicolo di Roma il WEAPON WARNING non aveva nessuna arma da mostrare).
// Separati dalla scena 3D così l'elenco esiste anche senza WebGL.
export const PASSI = [
  { chiave: "fotogramma", testo: "Everything starts from an ordinary CCTV frame, read at 25 frames per second. No special camera, no extra hardware: ARGO plugs into the streams a city already has.", titolo: "The real frame", tecnica: "Real CCTV of an armed robbery, Sunnybank (Queensland Police, CC BY) · 640×480 · read at 25 fps",
    didascalia: "A real frame, 1.6 s into the robbery: the masked man points a gun at a customer. The coloured points are the real output of YOLO26s-pose; the ring on the hand holding the gun is marked by us, ARGO does not draw it.",
    onesta: "The ring is ours: ARGO says whether there is a weapon, not where it is." },
  { chiave: "punti", testo: "YOLO26s-pose finds every person and 17 body points. A tracker keeps the same ID on each person across frames, so ARGO follows individuals, not isolated detections.", titolo: "17 points, 2 people", tecnica: "YOLO26s-pose at 640 px: 17 COCO keypoints · TrackTrack: stable IDs",
    didascalia: "The points lift off the image: 2 people × 17 points = 34 nodes, the robber and the customer. The colour is the tracker ID." },
  { chiave: "manichino", testo: "Points become skeletons. Working on skeletons makes ARGO lighter, less sensitive to light and clothing, and privacy-friendly: the analysis never needs a face.", titolo: "From points to mannequins", tecnica: "Bones of the COCO skeleton (19 segments per person)",
    didascalia: "3D mannequins built on the real points. The depth is illustrative: the network only sees x, y and confidence.",
    onesta: "The mannequin depth does not exist in the data: the network works in 2D." },
  { chiave: "pixel", testo: "Skeletons do not see what a hand holds. So ARGO reads the pixels too: a light video network watches the movement, and CLIP, a vision model that already knows what a pistol or a knife looks like, looks at one frame every 0.8 seconds.", titolo: "The pixels too", tecnica: "b1 R(2+1)D-18: 112 px, 16 of the 40 frames, every 0.8 s · CLIP ViT-B/16 (frozen): 224 px, 1 frame every 0.8 s, 3 views",
    didascalia: "On the frame: the centre cut of b1 and the three views of CLIP (the whole image, the left square and the right square, which see people at twice the resolution). In front: b1 looks at 16 frames at 112 px; CLIP at two frames 0.8 s apart." },
  { chiave: "grafo", testo: "Each skeleton is a graph: joints are nodes, bones are links. The network reads every joint together with the joints it is attached to, the way a body actually moves.", titolo: "One graph per skeleton", tecnica: "17 joints linked by 19 bones · every person goes through the same network", comandi: "archi",
    didascalia: "The ST-GCN works on the graph of the bones: each joint gathers information from the joints it is connected to. Switch on the all-to-all links to see the Graph-Transformer we tested and discarded (macro-F1 0.29 against 0.40)." },
  { chiave: "pila", testo: "A single photo says little. ARGO judges 40 frames, 1.6 seconds, at a time, and opens a new window every 0.4 seconds: more than two decisions per second.", titolo: "40 frames together", tecnica: "40-frame window (1.6 s), a new one every 10", comandi: "finestra",
    didascalia: "160 real frames of the robbery lined up in time. The lit window is what the network looks at in one go." },
  { chiave: "tempo", testo: "For every joint, the network follows the movement across nearby frames. Ten stacked blocks combine space and time until a raised arm becomes a punch, or a grab.", titolo: "Convolution over time", tecnica: "Each joint looks at itself in 9 nearby frames · 10 blocks, each one bones then time",
    didascalia: "The trails are the real movements of each joint over the 40 frames. The network slides a 9-frame filter along each trail; stacking 10 blocks, it sees the whole 1.6 s." },
  { chiave: "classi", testo: "The skeletons and b1 vote on violence: their average decides. CLIP says whether someone is snatching, comparing two frames 0.8 s apart. Seven violent windows in a row, about four seconds, raise the VIOLENCE ALARM.", titolo: "The decision", tecnica: "violence = average of ST-GCN and b1 · snatching from CLIP · armed violence = violence + WEAPON WARNING", comandi: "classi",
    didascalia: "The most suspicious person in the window counts. These are the real outputs of ARGO for the chosen window: in this robbery it says armed violence in 12 of the 13 windows (violence from the skeletons and b1, plus the WEAPON WARNING). Move the window in step 6 to see the others." },
  { chiave: "warning", testo: "Weapons are CLIP's job. Every 4 seconds it says whether it sees one, and above a cautious threshold it raises the WEAPON WARNING on its own, so a visible gun is never averaged away.", titolo: "WEAPON WARNING", tecnica: "CLIP weapon probability ≥ 0.16 (threshold chosen on validation), every 4 s", comandi: "warning",
    didascalia: "Independent of the class. The bar starts at CLIP's real answer on this robbery, above the threshold: the WEAPON WARNING is on. Move it to see when it would switch off.",
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
