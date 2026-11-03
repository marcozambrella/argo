// "Come funziona" in 3D: dal fotogramma vero al manichino, alle reti video b1 e b2 (due cubi di 16 fette), al grafo
// delle ossa (ST-GCN), alla pila di 40 frame, alla convoluzione nel tempo e al WARNING ARMA di b2 (v1.2, 04/11: niente
// detector, quindi niente pistola). Gli archi tutti-con-tutti del Graph-Transformer scartato restano come confronto,
// spenti all'inizio. I punti sono REALI (dati/esempio-finestra.json: YOLO26s-pose + TrackTrack su clip Pexels), come le
// decisioni (dati/esempio-decisione.json); la profondità del manichino è ILLUSTRATIVA e lo dice la didascalia.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { coloreCss, movimentoRidotto } from "./comune.js?v=20261103232018";
import { PASSI } from "./passi-modello.js?v=20261103232018";

const OSSA = [[15,13],[13,11],[16,14],[14,12],[11,12],[5,11],[6,12],[5,6],[5,7],[6,8],[7,9],[8,10],[1,2],[0,1],[0,2],[1,3],[2,4],[3,5],[4,6]];
const SOGLIA_VISIBILITA = 0.3;
const LARGO = 38.4, ALTO = 21.6;                // piano del fotogramma (1280×720 a scala di scena)
const FRAME_PER_FINESTRA = 40, PASSO = 10, FPS = 25;
const PASSO_PILA = 0.12, PASSO_TEMPO = 0.45, RAGGIO_PUNTO = 0.13;
// reti video (src/sicurezza/demo.py, ReteVideoArmiScippi): 16 dei 40 frame, schiacciati a 4:3 alti lato·8/7 (b1 170×128),
// poi il quadrato centrale di lato×lato: sul fotogramma originale è il centro largo 0,66 e alto 0,875.
const FRAME_RETE_VIDEO = 16, PASSO_FETTE = 0.22;
const ORIGINE = new THREE.Vector3();
const RETI_VIDEO = {
  b1: { lato: 112, secondi: 0.8, misura: 3.2, centro: [-6.5, -1, 6] },
  b2: { lato: 224, secondi: 4, misura: 6.4, centro: [5.5, 0, 6] },
};
const RITAGLIO = { larghezza: 112 / 170, altezza: 112 / 128 };
const SOGLIA_ARMA_B2 = 0.35;   // SOGLIA_ARMA_B2 = 0.3467 in demo.py, scelta sul val (D70)
const SMORZAMENTO = 4.4;       // al secondo: le opacità e la camera fanno il 63% della strada in ~0,23 s

const STATI = {   // opacità dei livelli; camera = spostamento dal fuoco (centro delle persone), "intero" = tutta l'immagine
  fotogramma: { immagine: 1, punti2d: 1, intero: true, camera: [0, -2, 36] },
  punti: { immagine: .25, punti3d: 1, ossa: .5, camera: [4, -4, 12] },
  manichino: { immagine: .08, manichini: 1, punti3d: .6, camera: [11, -2.5, 8] },
  pixel: { immagine: .3, ritaglio: 1, reti: 1, intero: true, camera: [12, -9, 25], guarda: [0, 0, 4] },
  grafo: { punti3d: 1, archi: 1, camera: [0, -8, 11] },
  pila: { pila: 1, camera: [26, 4, 8], guarda: [0, 0, 0] },
  tempo: { scie: 1, punti3d: .4, camera: [20, -3, 6], guarda: [0, 0, 0] },
  classi: { punti3d: .5, ossa: .5, camera: [0, -3, 14] },
  warning: { immagine: .2, reti: 1, intero: true, camera: [13, -5, 17], guarda: [5.5, 0, 5] },
};

export async function avviaModello() {
  const dati = await fetch("dati/esempio-finestra.json?v=20261103232018").then((r) => r.json());
  // probabilità VERE del sistema consegnato per le finestre di questa clip (argo analizza-video, D49 + D53)
  const decisioni = await fetch("dati/esempio-decisione.json?v=20261103232018").then((r) => r.json()).catch(() => null);
  const tela = document.getElementById("modello-3d");
  const renderer = new THREE.WebGLRenderer({ canvas: tela, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scena = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
  const controlli = new OrbitControls(camera, tela);
  controlli.enableDamping = true; controlli.enableZoom = false;
  scena.add(new THREE.HemisphereLight(0xfff4e0, 0x26231f, 1.1));
  const luce = new THREE.DirectionalLight(0xffe8c4, 1.2); luce.position.set(-6, -8, 12); scena.add(luce);

  const colore = (nome) => new THREE.Color(coloreCss(nome));
  const COLORI_ID = ["--furti", "--aggressioni", "--rapine", "--metro-c"].map(colore);
  const ultimo = dati.frame.length - 1;                       // il fotogramma mostrato è l'ultimo dei 160
  const persone = dati.frame[ultimo].persone.slice(0, 4);
  const posizione = ([x, y], z = 0) => new THREE.Vector3((x - 0.5) * LARGO, (0.5 - y) * ALTO, z);
  // profondità illustrativa del manichino: chi ha il box più basso nell'immagine è più vicino
  const profondita = Object.fromEntries(persone.map((p) => [p.id, (p.box[3] - 0.7) * 30]));
  // fuoco della camera: il centro delle persone (le persone sono piccole nel fotogramma)
  const fuoco = new THREE.Vector3();
  let contati = 0;
  for (const p of persone) for (const k of p.keypoint) if (k[2] >= SOGLIA_VISIBILITA) { fuoco.add(posizione(k)); contati++; }
  fuoco.divideScalar(Math.max(1, contati));
  const livelli = {};
  // etichetta sotto un cubo: canvas grande (nitida anche su schermi ad alta densità), sprite largo 3,6 unità
  const etichetta = (testo, mezzaMisura) => {
    const tela2d = Object.assign(document.createElement("canvas"), { width: 1024, height: 128 });
    const pennello = tela2d.getContext("2d");
    pennello.fillStyle = coloreCss("--testo");
    pennello.font = "500 64px 'IBM Plex Mono', monospace";
    pennello.textAlign = "center";
    pennello.textBaseline = "middle";
    pennello.fillText(testo, 512, 64);
    const trama = new THREE.CanvasTexture(tela2d);
    trama.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: trama, transparent: true, depthTest: false }));
    sprite.scale.set(4.8, 0.6, 1);
    sprite.position.set(0, -mezzaMisura - 0.6, 0);
    sprite.renderOrder = 10;   // sopra le fette trasparenti
    return sprite;
  };
  const nuovoLivello = (nome) => { const gruppo = new THREE.Group(); gruppo.userData.opacita = 0; livelli[nome] = gruppo; scena.add(gruppo); return gruppo; };
  const materialiDi = (gruppo) => { const materiali = new Set(); gruppo.traverse((o) => o.material && materiali.add(o.material)); return [...materiali]; };

  // 1. immagine del fotogramma
  const immagine = nuovoLivello("immagine");
  const trama = new THREE.TextureLoader().load("immagini/fotogramma-vicolo.jpg");
  trama.colorSpace = THREE.SRGBColorSpace;
  immagine.add(new THREE.Mesh(new THREE.PlaneGeometry(LARGO, ALTO), new THREE.MeshBasicMaterial({ map: trama, transparent: true })));

  // punti 2D sull'immagine e punti 3D staccati
  const sfera = new THREE.SphereGeometry(RAGGIO_PUNTO, 12, 12);
  const punti2d = nuovoLivello("punti2d"), punti3d = nuovoLivello("punti3d"), ossa = nuovoLivello("ossa"), manichini = nuovoLivello("manichini");
  const posizioniNodi = [];   // per gli archi: [persona, keypoint, Vector3]
  persone.forEach((persona, indicePersona) => {
    const coloreId = COLORI_ID[indicePersona];
    const materialePunto = new THREE.MeshBasicMaterial({ color: coloreId, transparent: true });
    persona.keypoint.forEach((punto, indiceKeypoint) => {
      if (punto[2] < SOGLIA_VISIBILITA) return;
      const piatto = new THREE.Mesh(sfera, materialePunto); piatto.position.copy(posizione(punto, 0.02)); punti2d.add(piatto);
      const staccato = new THREE.Mesh(sfera, materialePunto); staccato.position.copy(posizione(punto, 0.9 + profondita[persona.id])); punti3d.add(staccato);
      posizioniNodi.push({ persona: indicePersona, keypoint: indiceKeypoint, punto: staccato.position });
    });
    const materialeOsso = new THREE.LineBasicMaterial({ color: coloreId, transparent: true });
    const materialeManichino = new THREE.MeshStandardMaterial({ color: colore("--gesso"), roughness: .8, transparent: true });
    for (const [a, b] of OSSA) {
      const pa = persona.keypoint[a], pb = persona.keypoint[b];
      if (pa[2] < SOGLIA_VISIBILITA || pb[2] < SOGLIA_VISIBILITA) continue;
      const va = posizione(pa, 0.9 + profondita[persona.id]), vb = posizione(pb, 0.9 + profondita[persona.id]);
      ossa.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([va, vb]), materialeOsso));
      // manichino: una capsula per osso
      const lunghezza = va.distanceTo(vb);
      const capsula = new THREE.Mesh(new THREE.CapsuleGeometry(0.16, Math.max(0.01, lunghezza - 0.24), 4, 10), materialeManichino);
      capsula.position.copy(va).add(vb).multiplyScalar(0.5);
      capsula.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), vb.clone().sub(va).normalize());
      manichini.add(capsula);
    }
    const testa = persona.keypoint[0];
    if (testa[2] >= SOGLIA_VISIBILITA) {
      const sferaTesta = new THREE.Mesh(new THREE.SphereGeometry(0.34, 16, 16), materialeManichino);
      sferaTesta.position.copy(posizione(testa, 0.9 + profondita[persona.id])).add(new THREE.Vector3(0, 0.2, 0));
      manichini.add(sferaTesta);
    }
  });

  // le reti video: il ritaglio centrale sul fotogramma e, davanti, un cubo di 16 fette per rete. Le fette usano il
  // fotogramma mostrato (l'unico che abbiamo come immagine) alla risoluzione vera della rete: b1 si vede a quadretti.
  const ritaglio = nuovoLivello("ritaglio");
  const bordoRitaglio = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.PlaneGeometry(RITAGLIO.larghezza * LARGO, RITAGLIO.altezza * ALTO)),
    new THREE.LineBasicMaterial({ color: colore("--luce"), transparent: true }));
  bordoRitaglio.position.z = 0.05;
  ritaglio.add(bordoRitaglio);
  const reti = nuovoLivello("reti");
  const cornici = {};
  const immagineFotogramma = new Image();
  immagineFotogramma.src = "immagini/fotogramma-vicolo.jpg";
  await immagineFotogramma.decode().catch(() => null);
  for (const [nome, rete] of Object.entries(RETI_VIDEO)) {
    const tela = Object.assign(document.createElement("canvas"), { width: rete.lato, height: rete.lato });
    const larghezzaRitaglio = RITAGLIO.larghezza * immagineFotogramma.naturalWidth;
    const altezzaRitaglio = RITAGLIO.altezza * immagineFotogramma.naturalHeight;
    tela.getContext("2d").drawImage(immagineFotogramma, (immagineFotogramma.naturalWidth - larghezzaRitaglio) / 2,
      (immagineFotogramma.naturalHeight - altezzaRitaglio) / 2, larghezzaRitaglio, altezzaRitaglio, 0, 0, rete.lato, rete.lato);
    const trama = new THREE.CanvasTexture(tela);
    trama.colorSpace = THREE.SRGBColorSpace;
    trama.magFilter = THREE.NearestFilter;
    const cubo = new THREE.Group();
    cubo.position.set(...rete.centro);
    const fetta = new THREE.PlaneGeometry(rete.misura, rete.misura);
    for (let i = 0; i < FRAME_RETE_VIDEO; i++) {
      const materiale = new THREE.MeshBasicMaterial({ map: trama, transparent: true, depthWrite: false });
      materiale.userData.opacitaBase = i === FRAME_RETE_VIDEO - 1 ? 1 : 0.1 + 0.25 * i / FRAME_RETE_VIDEO;
      const piano = new THREE.Mesh(fetta, materiale);
      piano.position.z = (i - FRAME_RETE_VIDEO + 1) * PASSO_FETTE;
      cubo.add(piano);
    }
    const profonditaCubo = FRAME_RETE_VIDEO * PASSO_FETTE;
    const cornice = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(rete.misura, rete.misura, profonditaCubo)),
      new THREE.LineBasicMaterial({ color: colore("--gesso"), transparent: true }));
    cornice.position.z = -profonditaCubo / 2 + PASSO_FETTE / 2;
    cubo.add(cornice);
    cubo.add(etichetta(`${nome} · ${rete.lato} px · every ${rete.secondi} s`, rete.misura / 2));
    cornici[nome] = cornice;
    reti.add(cubo);
  }

  // archi per tipo (stesso frame)
  const archi = nuovoLivello("archi");
  const TIPI_ARCO = {
    ossa: { nome: "Bones", colore: colore("--gesso"), opacita: .9 },
    stessa: { nome: "Same person", colore: colore("--tenue"), opacita: .12 },
    diverse: { nome: "Different people", colore: colore("--aggressioni"), opacita: .07 },
  };
  const segmenti = Object.fromEntries(Object.keys(TIPI_ARCO).map((tipo) => [tipo, []]));
  const ossaInsieme = new Set(OSSA.map(([a, b]) => `${Math.min(a, b)}-${Math.max(a, b)}`));
  for (let i = 0; i < posizioniNodi.length; i++) {
    for (let j = i + 1; j < posizioniNodi.length; j++) {
      const a = posizioniNodi[i], b = posizioniNodi[j];
      let tipo = "diverse";
      if (a.persona === b.persona) tipo = ossaInsieme.has(`${Math.min(a.keypoint, b.keypoint)}-${Math.max(a.keypoint, b.keypoint)}`) ? "ossa" : "stessa";
      segmenti[tipo].push(a.punto, b.punto);
    }
  }
  const lineeArco = {};
  for (const [tipo, punti] of Object.entries(segmenti)) {
    const materiale = new THREE.LineBasicMaterial({ color: TIPI_ARCO[tipo].colore, transparent: true });
    materiale.userData.opacitaBase = TIPI_ARCO[tipo].opacita;
    lineeArco[tipo] = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(punti), materiale);
    archi.add(lineeArco[tipo]);
  }

  // pila di 160 frame veri in profondità (tempo)
  const pila = nuovoLivello("pila");
  const puntiPila = [], coloriPila = [], framePunto = [];
  dati.frame.forEach((frame, indiceFrame) => {
    for (const persona of frame.persone) for (const punto of persona.keypoint) {
      if (punto[2] < SOGLIA_VISIBILITA) continue;
      const v = posizione(punto, -indiceFrame * PASSO_PILA + 9);
      puntiPila.push(v.x, v.y, v.z); coloriPila.push(1, 1, 1); framePunto.push(indiceFrame);
    }
  });
  const geometriaPila = new THREE.BufferGeometry();
  geometriaPila.setAttribute("position", new THREE.Float32BufferAttribute(puntiPila, 3));
  geometriaPila.setAttribute("color", new THREE.Float32BufferAttribute(coloriPila, 3));
  pila.add(new THREE.Points(geometriaPila, new THREE.PointsMaterial({ size: .12, vertexColors: true, transparent: true })));
  const cornice = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(9, 7, FRAME_PER_FINESTRA * PASSO_PILA)),
    new THREE.LineBasicMaterial({ color: colore("--rosso"), transparent: true }));
  cornice.position.x = fuoco.x; cornice.position.y = fuoco.y;
  pila.add(cornice);
  let finestraScelta = 12;
  function aggiornaFinestra() {
    const primo = finestraScelta * PASSO, ultimoFrame = primo + FRAME_PER_FINESTRA - 1;
    const acceso = colore("--luce"), spento = colore("--linea");
    const attributo = geometriaPila.getAttribute("color");
    framePunto.forEach((frame, indice) => {
      const c = frame >= primo && frame <= ultimoFrame ? acceso : spento;
      attributo.setXYZ(indice, c.r, c.g, c.b);
    });
    attributo.needsUpdate = true;
    cornice.position.z = 9 - (primo + FRAME_PER_FINESTRA / 2 - .5) * PASSO_PILA;
    scriviClassi();
  }
  // pannello "classi": la decisione vera per la finestra scelta (se il file delle decisioni c'è)
  const righeClassi = new Map();
  function scriviClassi() {
    const decisione = decisioni?.finestre?.[finestraScelta];
    for (const [classe, uscita] of righeClassi) {
      const valore = decisione?.probabilita?.[classe.replaceAll(" ", "_")];
      uscita.textContent = valore === undefined ? "— waiting for results" : `${Math.round(valore * 100)}%`;
      uscita.parentElement.classList.toggle("scelta", decisione?.classe === classe.replaceAll(" ", "_"));
    }
  }
  aggiornaFinestra();

  // scie: il movimento vero di ogni punto nei 40 frame dell'ultima finestra
  const scie = nuovoLivello("scie");
  const primoScie = dati.frame.length - FRAME_PER_FINESTRA;
  persone.forEach((persona, indicePersona) => {
    const materiale = new THREE.LineBasicMaterial({ color: COLORI_ID[indicePersona], transparent: true });
    for (let k = 0; k < 17; k++) {
      const punti = [];
      for (let f = primoScie; f < dati.frame.length; f++) {
        const p = dati.frame[f].persone.find((q) => q.id === persona.id);
        if (p && p.keypoint[k][2] >= SOGLIA_VISIBILITA) punti.push(posizione(p.keypoint[k], -(f - primoScie) * PASSO_TEMPO + 9));
      }
      if (punti.length > 1) scie.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(punti), materiale));
    }
  });

  // --- comandi della scena (archi, finestra, classi, warning) ---------------------------------------------------
  const comandi = document.getElementById("comandi-scena");
  const didascalia = document.getElementById("didascalia-scena");
  // due gruppi di archi: le ossa (il grafo della ST-GCN) e i collegamenti tutti-con-tutti del Graph-Transformer scartato
  const GRUPPI_ARCO = { ossa: { nome: "Bones: the ST-GCN graph", tipi: ["ossa"] },
    scartato: { nome: "All-to-all links: Graph-Transformer (discarded)", tipi: ["stessa", "diverse"] } };
  const gruppoDi = Object.fromEntries(Object.entries(GRUPPI_ARCO).flatMap(([gruppo, { tipi }]) => tipi.map((tipo) => [tipo, gruppo])));
  const visibilitaArchi = { ossa: true, scartato: false };
  // WARNING ARMA: si parte dalla risposta vera di b2 nella finestra scelta, il cursore la simula
  let armaB2 = decisioni?.finestre?.[finestraScelta]?.probabilita_arma_video ?? 0;
  function mostraComandi(tipo) {
    comandi.replaceChildren();
    comandi.hidden = !tipo;
    if (tipo === "archi") {
      for (const [chiave, arco] of Object.entries(GRUPPI_ARCO)) {
        const etichetta = document.createElement("label");
        const casella = Object.assign(document.createElement("input"), { type: "checkbox", checked: visibilitaArchi[chiave] });
        casella.addEventListener("change", () => { visibilitaArchi[chiave] = casella.checked; });
        etichetta.append(casella, arco.nome);
        comandi.appendChild(etichetta);
      }
    } else if (tipo === "finestra") {
      const etichetta = document.createElement("label");
      const cursore = Object.assign(document.createElement("input"), { type: "range", min: 0, max: (dati.frame.length - FRAME_PER_FINESTRA) / PASSO, value: finestraScelta });
      const uscita = document.createElement("output");
      const scrivi = () => {
        const primo = finestraScelta * PASSO;
        uscita.textContent = `window ${finestraScelta + 1} · frames ${primo}–${primo + FRAME_PER_FINESTRA - 1} · ${(primo / FPS).toFixed(1)}–${((primo + FRAME_PER_FINESTRA) / FPS).toFixed(1)} s`;
      };
      cursore.setAttribute("aria-label", "40-frame window");
      cursore.addEventListener("input", () => { finestraScelta = Number(cursore.value); aggiornaFinestra(); scrivi(); });
      scrivi();
      etichetta.append("Window", cursore, uscita);
      comandi.appendChild(etichetta);
    } else if (tipo === "classi") {
      for (const [classe, testoClasse] of [["normale", "normal"], ["violenza", "violence"], ["violenza con arma", "armed violence"], ["scippo", "snatching"]]) {
        const riga = document.createElement("span");
        riga.className = "riga";
        const uscita = document.createElement("output");
        riga.append(testoClasse, " ", uscita);
        righeClassi.set(classe, uscita);
        comandi.appendChild(riga);
      }
      scriviClassi();
    } else if (tipo === "warning") {
      armaB2 = decisioni?.finestre?.[finestraScelta]?.probabilita_arma_video ?? 0;
      const etichetta = document.createElement("label");
      const cursore = Object.assign(document.createElement("input"), { type: "range", min: 0, max: 100, value: Math.round(armaB2 * 100) });
      cursore.setAttribute("aria-label", "b2 weapon probability (simulation)");
      const uscita = document.createElement("output");
      const scrivi = () => {
        const scatta = armaB2 >= SOGLIA_ARMA_B2;
        uscita.textContent = `b2 weapon ${Math.round(armaB2 * 100)}% · threshold 35% → ${scatta ? "WEAPON WARNING" : "no warning"}`;
        uscita.classList.toggle("allarme", scatta);
      };
      cursore.addEventListener("input", () => { armaB2 = Number(cursore.value) / 100; scrivi(); });
      scrivi();
      etichetta.append("b2", cursore, uscita);
      comandi.appendChild(etichetta);
    }
  }

  // --- passi a scorrimento -----------------------------------------------------------------------------------
  let statoCorrente = STATI.fotogramma, chiaveCorrente = null;
  function vaiA(chiave) {
    if (chiave === chiaveCorrente) return;
    chiaveCorrente = chiave;
    statoCorrente = STATI[chiave];
    const passo = PASSI.find((p) => p.chiave === chiave);
    didascalia.textContent = passo.didascalia;
    mostraComandi(passo.comandi);
    document.querySelectorAll(".passo-modello").forEach((voce) => voce.classList.toggle("attivo", voce.dataset.chiave === chiave));
  }
  // il passo attivo è quello a metà schermo; sul telefono la scena occupa la metà alta, quindi quello che passa sotto
  let fascia = "-45% 0px -45% 0px";
  if (matchMedia("(max-width: 960px)").matches) fascia = "-62% 0px -18% 0px";
  const osservatore = new IntersectionObserver((voci) => {
    for (const voce of voci) if (voce.isIntersecting) vaiA(voce.target.dataset.chiave);
  }, { rootMargin: fascia });
  document.querySelectorAll(".passo-modello").forEach((voce) => osservatore.observe(voce));
  vaiA("fotogramma");

  function ridimensiona() {
    const { width, height } = tela.getBoundingClientRect();
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.fov = width < 600 ? 58 : 40;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(ridimensiona).observe(tela);
  ridimensiona();

  const posizioneCamera = new THREE.Vector3(0, -2, 36), bersaglio = new THREE.Vector3();
  const versoCamera = new THREE.Vector3(), versoBersaglio = new THREE.Vector3();
  camera.position.copy(posizioneCamera);
  let interazioneUtente = false;
  controlli.addEventListener("start", () => { interazioneUtente = true; });
  // i materiali di ogni livello (fissi dopo la costruzione) e i colori: calcolati una volta, non a ogni frame
  const materialiDeiLivelli = {};
  for (const [nome, gruppo] of Object.entries(livelli)) {
    const materiali = new Set();
    gruppo.traverse((oggetto) => { if (oggetto.material) materiali.add(oggetto.material); });
    materialiDeiLivelli[nome] = [...materiali];
  }
  const COLORE_CORNICE = colore("--gesso"), COLORE_CHIAMATA = colore("--luce"), COLORE_ALLARME = colore("--rosso");
  const orologio = new THREE.Clock();
  let inVista = false, inCorso = false;
  function anima() {
    if (!inVista) { inCorso = false; return; }
    const intervallo = Math.min(orologio.getDelta(), 0.1);
    const t = orologio.elapsedTime;
    // smorzamento legato al tempo: la stessa velocità a 30, 60 o 120 Hz
    const passo = movimentoRidotto() ? 1 : 1 - Math.exp(-SMORZAMENTO * intervallo);
    for (const [nome, gruppo] of Object.entries(livelli)) {
      const obiettivo = statoCorrente[nome] ?? 0;
      gruppo.userData.opacita += (obiettivo - gruppo.userData.opacita) * passo;
      gruppo.visible = gruppo.userData.opacita > 0.01;
      if (!gruppo.visible) continue;
      for (const materiale of materialiDeiLivelli[nome]) materiale.opacity = gruppo.userData.opacita * (materiale.userData.opacitaBase ?? 1);
    }
    for (const [tipo, linee] of Object.entries(lineeArco)) linee.visible = visibilitaArchi[gruppoDi[tipo]];
    // le reti video "chiamate" alla loro cadenza: la cornice si accende e si spegne in 0,4 s
    for (const [nome, rete] of Object.entries(RETI_VIDEO)) {
      const materiale = cornici[nome].material;
      if (chiaveCorrente === "warning" && nome === "b2") {
        materiale.color.copy(armaB2 >= SOGLIA_ARMA_B2 ? COLORE_ALLARME : COLORE_CORNICE);
        continue;
      }
      const accensione = movimentoRidotto() ? 0 : Math.max(0, 1 - (t % rete.secondi) / 0.4);
      materiale.color.copy(COLORE_CORNICE).lerp(COLORE_CHIAMATA, accensione);
    }
    if (!interazioneUtente) {
      const centro = statoCorrente.intero ? ORIGINE : fuoco;
      versoCamera.set(...statoCorrente.camera).add(centro);
      posizioneCamera.lerp(versoCamera, passo);
      camera.position.copy(posizioneCamera);
      versoBersaglio.set(...(statoCorrente.guarda ?? [0, 0, 0])).add(centro);
      bersaglio.lerp(versoBersaglio, passo);
      controlli.target.copy(bersaglio);
    }
    controlli.update();
    renderer.render(scena, camera);
    requestAnimationFrame(anima);
  }
  // quando si cambia passo la camera torna a seguire il racconto
  new MutationObserver(() => { interazioneUtente = false; }).observe(didascalia, { childList: true, characterData: true, subtree: true });
  // si disegna solo quando la scena si vede (prima girava per tutta la vita della pagina)
  new IntersectionObserver((voci) => {
    inVista = voci[0].isIntersecting;
    if (inVista && !inCorso) { inCorso = true; orologio.getDelta(); anima(); }
  }).observe(tela);
}
