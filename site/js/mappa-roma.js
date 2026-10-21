// Mappa 3D di Roma: municipi come plastico in gesso, cubetti colorati per categoria di reato.
// I TOTALI sono ufficiali (dati/roma.json); la RIPARTIZIONE per municipio è una stima dichiarata (popolazione,
// +50% nei municipi con zone rosse, i cubetti in più posati sulle vie delle zone rosse). Scelte di Marco, 18/10/2026.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { casualeConSeme, coloreCss, formattaNumero, movimentoRidotto } from "./comune.js";

const LON0 = 12.49, LAT0 = 41.89, SCALA = 100, COS = Math.cos(LAT0 * Math.PI / 180);
const proietta = ([lon, lat]) => new THREE.Vector2((lon - LON0) * COS * SCALA, (lat - LAT0) * SCALA);
const ALTEZZA_PLASTICO = 0.5, LATO_CUBETTO = 0.42, CUBETTI_PER_PILA = 3;
const COLORE_CATEGORIA = { furti_persone: "--furti", aggressioni: "--aggressioni", rapine_negozi: "--rapine" };
const COLORE_METRO = { A: "--metro-a", B: "--metro-b", B1: "--metro-b", C: "--metro-c" };

function puntoNelPoligono(punto, anello) {
  let dentro = false;
  for (let i = 0, j = anello.length - 1; i < anello.length; j = i++) {
    const [xi, yi] = [anello[i].x, anello[i].y], [xj, yj] = [anello[j].x, anello[j].y];
    if ((yi > punto.y) !== (yj > punto.y) && punto.x < ((xj - xi) * (punto.y - yi)) / (yj - yi) + xi) dentro = !dentro;
  }
  return dentro;
}

export async function avviaMappa() {
  const [dati, geo] = await Promise.all([fetch("dati/roma.json").then((r) => r.json()), fetch("dati/mappa-roma.geojson").then((r) => r.json())]);
  riempiLegenda(dati);

  const tela = document.getElementById("mappa-3d");
  const area = tela.parentElement;
  const renderer = new THREE.WebGLRenderer({ canvas: tela, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const etichette = new CSS2DRenderer();
  Object.assign(etichette.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none" });
  area.appendChild(etichette.domElement);

  const scena = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 600);
  camera.up.set(0, 0, 1);
  camera.position.set(38, -36, 15);   // radente da sud-est, alzata del 15% (scelta di Marco)
  const controlli = new OrbitControls(camera, tela);
  controlli.enableDamping = true;
  controlli.maxPolarAngle = Math.PI / 2.3;
  controlli.minDistance = 12; controlli.maxDistance = 140;
  controlli.target.set(0, 2, 0);

  // luce bassa e radente: ombre lunghe sul plastico
  scena.add(new THREE.HemisphereLight(0xffffff, 0x6f685d, 1.05));
  const sole = new THREE.DirectionalLight(0xfff3e0, 1.35);
  sole.position.set(-40, -18, 22);
  sole.castShadow = true;
  sole.shadow.mapSize.set(2048, 2048);
  Object.assign(sole.shadow.camera, { left: -45, right: 45, top: 45, bottom: -45, near: 1, far: 140 });
  scena.add(sole);
  const terreno = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.ShadowMaterial({ opacity: 0.35 }));
  terreno.receiveShadow = true;
  scena.add(terreno);

  // --- municipi in gesso -------------------------------------------------------------------------------------
  const municipi = [];
  const materialeGesso = () => new THREE.MeshStandardMaterial({ color: new THREE.Color(coloreCss("--gesso")), roughness: 0.95 });
  for (const elemento of geo.features.filter((f) => f.properties.tipo === "municipio")) {
    const materiale = materialeGesso();
    const anelli = elemento.geometry.coordinates.map(([anello]) => anello.map(proietta));
    const gruppo = new THREE.Group();
    for (const anello of anelli) {
      const geometria = new THREE.ExtrudeGeometry(new THREE.Shape(anello), { depth: ALTEZZA_PLASTICO, bevelEnabled: false });
      const solido = new THREE.Mesh(geometria, materiale);
      solido.castShadow = true; solido.receiveShadow = true;
      gruppo.add(solido);
      gruppo.add(new THREE.LineSegments(new THREE.EdgesGeometry(geometria, 25),
        new THREE.LineBasicMaterial({ color: 0x5a544b, transparent: true, opacity: 0.6 })));
    }
    scena.add(gruppo);
    const principale = anelli.reduce((a, b) => (a.length > b.length ? a : b));
    const centro = principale.reduce((somma, p) => somma.add(p), new THREE.Vector2()).divideScalar(principale.length);
    const sigla = elemento.properties.sigla;
    const etichetta = Object.assign(document.createElement("span"), { className: "etichetta-municipio-3d", textContent: sigla });
    Object.assign(etichetta.style, { font: "800 13px 'Alegreya Sans SC', sans-serif", color: "#3d3933", letterSpacing: ".06em" });
    const oggettoEtichetta = new CSS2DObject(etichetta);
    oggettoEtichetta.position.set(centro.x, centro.y, ALTEZZA_PLASTICO + 0.05);
    scena.add(oggettoEtichetta);
    municipi.push({ sigla, gruppo, materiale, anelli, principale, centro });
  }

  // --- zone rosse, metro, stazioni ----------------------------------------------------------------------------
  const tubo = (punti, raggio, colore) => {
    const curva = new THREE.CatmullRomCurve3(punti.map((p) => new THREE.Vector3(p.x, p.y, ALTEZZA_PLASTICO + 0.06)), false, "catmullrom", 0);
    return new THREE.Mesh(new THREE.TubeGeometry(curva, Math.max(2, punti.length * 2), raggio, 6, false), new THREE.MeshBasicMaterial({ color: colore }));
  };
  const puntiZoneRosse = { I: [], VII: [] };
  for (const elemento of geo.features) {
    const { tipo } = elemento.properties;
    if (tipo === "zona_rossa") {
      const punti = elemento.geometry.coordinates.map(proietta);
      if (punti.length > 1) scena.add(tubo(punti, 0.07, new THREE.Color(coloreCss("--rosso"))));
      puntiZoneRosse[elemento.properties.municipio].push(...punti);
    } else if (tipo === "metro") {
      for (const tratto of elemento.geometry.coordinates) {
        if (tratto.length > 1) scena.add(tubo(tratto.map(proietta), 0.06, new THREE.Color(coloreCss(COLORE_METRO[elemento.properties.linea]))));
      }
    } else if (tipo === "stazione") {
      const posizione = proietta(elemento.geometry.coordinates);
      const pilastro = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.35, 2.2), new THREE.MeshStandardMaterial({ color: 0xf2eee4, roughness: .6 }));
      pilastro.position.set(posizione.x, posizione.y, ALTEZZA_PLASTICO + 1.1);
      pilastro.castShadow = true;
      scena.add(pilastro);
      const etichetta = Object.assign(document.createElement("span"), { textContent: elemento.properties.nome.replace("Roma ", "") });
      Object.assign(etichetta.style, { font: "700 12px 'Alegreya Sans SC', sans-serif", color: "#f2eee4", background: "rgba(27,26,24,.8)", padding: "1px 6px", letterSpacing: ".05em" });
      const oggetto = new CSS2DObject(etichetta);
      oggetto.position.set(posizione.x, posizione.y, ALTEZZA_PLASTICO + 2.6);
      scena.add(oggetto);
    }
  }

  // --- fermate della metro: pallini come sui cartelli di linea (bianco con anello del colore della linea;
  //     le fermate di scambio bianche con anello scuro e più grandi) -----------------------------------------------
  const fermate = [];
  const disco = (raggio, altezza) => new THREE.CylinderGeometry(raggio, raggio, altezza, 24).rotateX(Math.PI / 2);
  const geometrie = { anello: disco(0.2, 0.08), centro: disco(0.12, 0.1), anelloScambio: disco(0.3, 0.1), centroScambio: disco(0.19, 0.12), presa: disco(0.5, 0.3) };
  const materialeBianco = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const materialePresa = new THREE.MeshBasicMaterial({ visible: false });
  for (const elemento of geo.features.filter((f) => f.properties.tipo === "fermata")) {
    const { nome, linee } = elemento.properties;
    const scambio = linee.length > 1;
    const posizione = proietta(elemento.geometry.coordinates);
    const gruppo = new THREE.Group();
    gruppo.position.set(posizione.x, posizione.y, ALTEZZA_PLASTICO + 0.1);
    const coloreAnello = scambio ? new THREE.Color(0x1b1a18) : new THREE.Color(coloreCss(COLORE_METRO[linee[0]]));
    gruppo.add(new THREE.Mesh(scambio ? geometrie.anelloScambio : geometrie.anello, new THREE.MeshBasicMaterial({ color: coloreAnello })));
    gruppo.add(new THREE.Mesh(scambio ? geometrie.centroScambio : geometrie.centro, materialeBianco));
    const presa = new THREE.Mesh(geometrie.presa, materialePresa);   // area invisibile più larga, per puntarla col mouse
    gruppo.add(presa);
    scena.add(gruppo);
    fermate.push({ nome, linee, presa, gruppo });
  }
  // cartello della fermata sotto il puntatore, come i cartelli di stazione: lettere di linea nei cerchi colorati
  const cartello = Object.assign(document.createElement("span"), { className: "cartello-fermata" });
  const oggettoCartello = new CSS2DObject(cartello);
  oggettoCartello.center.set(0.5, 1.4);
  oggettoCartello.visible = false;
  scena.add(oggettoCartello);

  // --- cubetti: stima per municipio, una InstancedMesh per categoria ----------------------------------------------
  const geometriaCubetto = new THREE.BoxGeometry(LATO_CUBETTO, LATO_CUBETTO, LATO_CUBETTO);
  const categorie = {};
  for (const [indiceCategoria, categoria] of dati.categorie.entries()) {
    const posizioni = [];
    const aggiungiPile = (numero, scegliPunto) => {
      for (let messi = 0; messi < numero;) {
        const base = scegliPunto();
        for (let piano = 0; piano < CUBETTI_PER_PILA && messi < numero; piano++, messi++) {
          posizioni.push(new THREE.Vector3(base.x, base.y, ALTEZZA_PLASTICO + LATO_CUBETTO / 2 + piano * LATO_CUBETTO * 1.04));
        }
      }
    };
    for (const municipio of municipi) {
      const stima = dati.stime_per_municipio[municipio.sigla][categoria.chiave];
      const casuale = casualeConSeme(`${municipio.sigla}-${categoria.chiave}`);
      const [minimo, massimo] = municipio.principale.reduce(([a, b], p) => [a.clone().min(p), b.clone().max(p)],
        [new THREE.Vector2(Infinity, Infinity), new THREE.Vector2(-Infinity, -Infinity)]);
      aggiungiPile(Math.round(stima.base / categoria.unita_cubetto), () => {
        for (let tentativo = 0; tentativo < 500; tentativo++) {
          const punto = new THREE.Vector2(minimo.x + casuale() * (massimo.x - minimo.x), minimo.y + casuale() * (massimo.y - minimo.y));
          if (puntoNelPoligono(punto, municipio.principale)) return punto;
        }
        return municipio.centro;
      });
      const puntiRossi = puntiZoneRosse[municipio.sigla];
      if (stima.zone_rosse > 0 && puntiRossi?.length) {
        aggiungiPile(Math.round(stima.zone_rosse / categoria.unita_cubetto), () => {
          const punto = puntiRossi[Math.floor(casuale() * puntiRossi.length)];
          return new THREE.Vector2(punto.x + (casuale() - .5) * .5 + indiceCategoria * .15, punto.y + (casuale() - .5) * .5);
        });
      }
    }
    const materiale = new THREE.MeshStandardMaterial({ color: new THREE.Color(coloreCss(COLORE_CATEGORIA[categoria.chiave])), roughness: .7 });
    const istanze = new THREE.InstancedMesh(geometriaCubetto, materiale, posizioni.length);
    istanze.castShadow = true;
    scena.add(istanze);
    categorie[categoria.chiave] = { istanze, posizioni, materiale, visibile: true, comparsa: 0 };
  }

  // --- interruttori: uno per categoria di reato ---------------------------------------------------------------------
  const contenitore = document.getElementById("interruttori-mappa");
  for (const categoria of dati.categorie) {
    const bottone = document.createElement("button");
    bottone.type = "button";
    bottone.className = "interruttore";
    bottone.setAttribute("aria-pressed", "true");
    bottone.innerHTML = `<i style="background: var(${COLORE_CATEGORIA[categoria.chiave]})"></i>${categoria.nome}`;
    bottone.addEventListener("click", () => {
      const stato = categorie[categoria.chiave];
      stato.visibile = !stato.visibile;
      bottone.setAttribute("aria-pressed", String(stato.visibile));
    });
    contenitore.appendChild(bottone);
  }

  // --- scheda del municipio sotto il puntatore --------------------------------------------------------------------
  const raggio = new THREE.Raycaster();
  const puntatore = new THREE.Vector2();
  const scheda = document.getElementById("scheda-municipio");
  let evidenziato = null;
  const solidi = municipi.flatMap((m) => m.gruppo.children.filter((figlio) => figlio.isMesh).map((solido) => [solido, m]));
  function scegli(evento) {
    const rettangolo = tela.getBoundingClientRect();
    puntatore.set(((evento.clientX - rettangolo.left) / rettangolo.width) * 2 - 1, -((evento.clientY - rettangolo.top) / rettangolo.height) * 2 + 1);
    raggio.setFromCamera(puntatore, camera);
    const fermataColpita = raggio.intersectObjects(fermate.map((f) => f.presa))[0];
    const fermata = fermataColpita && fermate.find((f) => f.presa === fermataColpita.object);
    oggettoCartello.visible = Boolean(fermata);
    if (fermata) {
      cartello.innerHTML = fermata.linee.map((linea) => `<i style="background: var(${COLORE_METRO[linea]})">${linea}</i>`).join("") + `<b>${fermata.nome}</b>`;
      oggettoCartello.position.copy(fermata.gruppo.position);
    }
    const colpito = raggio.intersectObjects(solidi.map(([solido]) => solido))[0];
    const municipio = colpito ? solidi.find(([solido]) => solido === colpito.object)[1] : null;
    if (municipio === evidenziato) return;
    if (evidenziato) evidenziato.materiale.color.set(coloreCss("--gesso"));
    evidenziato = municipio;
    if (!municipio) return;
    municipio.materiale.color.set(new THREE.Color(coloreCss("--gesso")).multiplyScalar(0.82));
    const stime = dati.stime_per_municipio[municipio.sigla];
    const righe = dati.categorie.map((c) => {
      const s = stime[c.chiave];
      return `<span>${c.nome}: ~${formattaNumero(s.base + s.zone_rosse)}</span>`;
    }).join("");
    const zona = dati.zone_rosse[municipio.sigla] ? `<span>Zona rossa: ${dati.zone_rosse[municipio.sigla]} (peso +50%)</span>` : "";
    scheda.innerHTML = `<b>Municipio ${municipio.sigla}</b><span>${formattaNumero(dati.popolazione_municipi[municipio.sigla])} residenti (31/12/2024)</span>
      ${zona}${righe}<span class="stima">Stima, non un dato: vedi «Attenzione» in legenda.</span>`;
  }
  tela.addEventListener("pointermove", scegli);
  tela.addEventListener("pointerdown", scegli);

  function ridimensiona() {
    const { width, height } = area.getBoundingClientRect();
    renderer.setSize(width, height, false);
    etichette.setSize(width, height);
    camera.aspect = width / height;
    camera.position.multiplyScalar(width < 600 ? 1.25 : 1);
    camera.updateProjectionMatrix();
  }
  ridimensiona();
  new ResizeObserver(() => {
    const { width, height } = area.getBoundingClientRect();
    renderer.setSize(width, height, false); etichette.setSize(width, height);
    camera.aspect = width / height; camera.updateProjectionMatrix();
  }).observe(area);

  // cubetti che cadono la prima volta, poi seguono gli interruttori
  const matrice = new THREE.Matrix4();
  const inizio = performance.now();
  function aggiornaCubetti() {
    const tempo = (performance.now() - inizio) / 1000;
    let indiceCategoria = 0;
    for (const stato of Object.values(categorie)) {
      const obiettivo = stato.visibile ? 1 : 0;
      stato.comparsa += (obiettivo - stato.comparsa) * (movimentoRidotto() ? 1 : 0.12);
      stato.posizioni.forEach((posizione, indice) => {
        const ritardo = indiceCategoria * 0.6 + (indice % 40) * 0.03;
        const caduta = movimentoRidotto() ? 1 : Math.min(1, Math.max(0, (tempo - ritardo) / 0.7));
        const altezza = posizione.z + (1 - caduta * caduta) * 25;
        const scala = Math.max(0.0001, stato.comparsa);
        matrice.makeScale(scala, scala, scala).setPosition(posizione.x, posizione.y, altezza);
        stato.istanze.setMatrixAt(indice, matrice);
      });
      stato.istanze.instanceMatrix.needsUpdate = true;
      indiceCategoria++;
    }
  }
  function anima() {
    aggiornaCubetti();
    controlli.update();
    renderer.render(scena, camera);
    etichette.render(scena, camera);
    requestAnimationFrame(anima);
  }
  anima();
}

function riempiLegenda(dati) {
  const voci = document.getElementById("voci-legenda");
  for (const categoria of dati.categorie) {
    const fonte = dati.fonti[categoria.fonte];
    const voce = document.createElement("li");
    voce.innerHTML = `<i style="background: var(${COLORE_CATEGORIA[categoria.chiave]})"></i><div><b>${categoria.nome}</b>
      <span class="valore">${formattaNumero(categoria.totale)}</span> nel ${categoria.anno} a Roma
      <small>1 cubetto = ${categoria.unita_cubetto} · ${categoria.dettaglio}${categoria.nota ? ". " + categoria.nota : ""}
      Fonte: <a href="${fonte.url}">${categoria.fonte === "prefettura" ? "Ministero dell'Interno" : "Annuario di Roma Capitale"}</a></small></div>`;
    voci.appendChild(voce);
  }
  const colonna = document.getElementById("colonna-proporzioni");
  for (const categoria of dati.categorie) {
    const pezzo = document.createElement("span");
    Object.assign(pezzo.style, { width: `${categoria.quota_sui_tre * 100}%`, background: `var(${COLORE_CATEGORIA[categoria.chiave]})` });
    colonna.appendChild(pezzo);
  }
  document.getElementById("testo-proporzioni").textContent = dati.categorie
    .map((c) => `${c.nome.toLowerCase()} ${(c.quota_sui_tre * 100).toLocaleString("it-IT", { maximumFractionDigits: 1 })}%`).join(" · ")
    + ". Sulla mappa ogni colore ha la sua scala: le altezze tra colori diversi non si confrontano.";
  document.getElementById("testo-metodo").textContent = dati.metodo_stima.descrizione;
  const fonti = document.getElementById("elenco-fonti");
  for (const fonte of Object.values(dati.fonti)) {
    fonti.appendChild(Object.assign(document.createElement("li"), { innerHTML: `<a href="${fonte.url}">${fonte.nome}</a>` }));
  }
}
