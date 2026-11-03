// Argo — pannello delle telecamere (D56). Legge /api/stato e /api/eventi a intervalli; le immagini arrivano come MJPEG.
"use strict";

const CLASSI = ["normale", "violenza", "violenza_con_arma", "scippo"];
const NOMI = { normale: "Normal", violenza: "Violence", violenza_con_arma: "Armed violence", scippo: "Snatching" };
const $ = (selettore) => document.querySelector(selettore);

const riquadri = new Map();          // nome telecamera → elementi del riquadro
const ultimoStatoDi = new Map();     // nome telecamera → ultimo evento visto (per il registro dei soli cambiamenti)
let prossimoEvento = 0;

const percento = (valore) => `${Math.round(valore * 100)}%`;
const minutiSecondi = (secondi) => {
  const s = Math.max(0, Math.round(secondi));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

// ---------------------------------------------------------------------------------------------------- schede
for (const scheda of document.querySelectorAll(".schede button")) {
  scheda.addEventListener("click", () => {
    for (const altra of document.querySelectorAll(".schede button")) {
      const scelta = altra === scheda;
      altra.setAttribute("aria-selected", String(scelta));
      document.getElementById(altra.getAttribute("aria-controls")).hidden = !scelta;
    }
    if (scheda.id === "scheda-archivio") caricaArchivio();
  });
}

setInterval(() => { $("#orologio").textContent = new Date().toLocaleTimeString("en-GB"); }, 1000);

// ---------------------------------------------------------------------------------------------------- telecamere
function creaRiquadro(telecamera) {
  const nodo = $("#modello-riquadro").content.firstElementChild.cloneNode(true);
  nodo.querySelector(".nome-cam").textContent = telecamera.nome.toUpperCase();
  nodo.querySelector("img").src = `/flusso/${encodeURIComponent(telecamera.nome)}.mjpg`;
  nodo.querySelector("img").alt = `Camera ${telecamera.nome}, ${telecamera.sorgente}`;
  const barre = nodo.querySelector(".barre");
  const elementiBarre = {};
  for (const classe of CLASSI) {
    const nome = document.createElement("span");
    nome.textContent = NOMI[classe];
    const traccia = document.createElement("span");
    traccia.className = "traccia";
    const riempimento = document.createElement("span");
    riempimento.className = "riempimento";
    riempimento.style.display = "block";
    riempimento.style.background = `var(--c-${classe})`;
    traccia.append(riempimento);
    const valore = document.createElement("span");
    valore.className = "valore";
    barre.append(nome, traccia, valore);
    elementiBarre[classe] = { riempimento, valore };
  }
  $("#griglia").append(nodo);
  const riquadro = { nodo, elementiBarre };
  riquadri.set(telecamera.nome, riquadro);
  return riquadro;
}

function testoBadge(telecamera) {
  if (telecamera.fase === "errore") return ["ERROR", "errore"];
  if (telecamera.fase === "finita") return ["ENDED", ""];
  if (telecamera.fase !== "in analisi") return ["STARTING", ""];
  if (!telecamera.dal_vivo) return ["RECORDING", ""];
  if (telecamera.ritardo_secondi !== null && telecamera.ritardo_secondi < 3) return ["LIVE", "live"];
  return [`DELAYED −${minutiSecondi(telecamera.ritardo_secondi ?? 0)}`, "differita"];
}

function aggiornaRiquadro(riquadro, telecamera) {
  const { nodo, elementiBarre } = riquadro;
  const [testo, tipo] = testoBadge(telecamera);
  const badge = nodo.querySelector(".badge");
  badge.textContent = testo;
  badge.className = `badge ${tipo}`;
  const misure = [];
  if (telecamera.frame_al_secondo) misure.push(`${telecamera.frame_al_secondo} fps`);
  if (telecamera.secondi_analizzati) misure.push(`${minutiSecondi(telecamera.secondi_analizzati)} analysed`);
  nodo.querySelector(".misure").textContent = misure.join(" · ");
  const evento = telecamera.ultimo_evento;
  nodo.querySelector(".classe").textContent = evento?.classe ? NOMI[evento.classe] : (evento ? "Skeletons only" : "—");
  nodo.querySelector(".classe").style.color = evento?.classe && evento.classe !== "normale" ? `var(--c-${evento.classe})` : "";
  for (const classe of CLASSI) {
    const valore = evento?.probabilita ? evento.probabilita[classe] : 0;
    elementiBarre[classe].riempimento.style.width = percento(valore);
    elementiBarre[classe].valore.textContent = evento?.probabilita ? percento(valore) : "";
  }
  nodo.querySelector(".barre").hidden = !evento?.probabilita;
  nodo.querySelector(".spia-warning").classList.toggle("accesa", Boolean(evento?.warning_arma));
  nodo.querySelector(".spia-allarme").classList.toggle("accesa", Boolean(evento?.allarme_violenza));
  nodo.classList.toggle("in-allarme", Boolean(evento?.allarme_violenza));
  const errore = nodo.querySelector(".errore");
  errore.hidden = !telecamera.errore;
  errore.textContent = telecamera.errore || "";
}

async function aggiornaStato() {
  try {
    const stato = await (await fetch("/api/stato")).json();
    const modalita = $("#modalita");
    modalita.textContent = stato.modalita;
    modalita.classList.toggle("scheletri", stato.modalita === "skeletons only");
    modalita.title = stato.modalita === "skeletons only" ? "Weights not found: uv run argo scarica-pesi" : "";
    for (const telecamera of stato.telecamere) {
      aggiornaRiquadro(riquadri.get(telecamera.nome) || creaRiquadro(telecamera), telecamera);
    }
    if (!stato.telecamere.length && !$("#griglia .vuoto")) {
      const vuoto = document.createElement("p");
      vuoto.className = "vuoto";
      vuoto.textContent = "No camera connected: start with --telecamera name=rtsp://… (or /dev/video0, or a file). Past analyses are in the Archive.";
      $("#griglia").append(vuoto);
    }
    const inAllarme = stato.telecamere.filter((t) => t.ultimo_evento?.allarme_violenza).map((t) => t.nome.toUpperCase());
    const striscia = $("#striscia-allarme");
    striscia.hidden = !inAllarme.length;
    striscia.textContent = inAllarme.length ? `VIOLENCE ALARM — ${inAllarme.join(", ")}` : "";
  } catch (errore) {
    $("#modalita").textContent = "panel unreachable";
  }
}

// ---------------------------------------------------------------------------------------------------- registro
// Solo i cambiamenti: allarme acceso/spento, warning arma, classe diversa dalla precedente.
function cambiamenti(evento, precedente) {
  const voci = [];
  if (evento.allarme_violenza && !precedente?.allarme_violenza) voci.push(["VIOLENCE ALARM", "allarme"]);
  if (!evento.allarme_violenza && precedente?.allarme_violenza) voci.push(["alarm over", "fine"]);
  if (evento.warning_arma && !precedente?.warning_arma) voci.push(["WEAPON WARNING", "warning"]);
  if (evento.classe && evento.classe !== (precedente?.classe ?? "normale")) {
    const probabilita = evento.probabilita ? ` (${percento(evento.probabilita[evento.classe])})` : "";
    voci.push([`class: ${NOMI[evento.classe]}${probabilita}`, ""]);
  }
  return voci;
}

function vocePerRegistro(quando, dove, cosa, tipo) {
  const voce = document.createElement("li");
  voce.className = tipo;
  for (const [classe, testo] of [["quando", quando], ["dove", dove], ["cosa", cosa]]) {
    if (!testo) continue;
    const parte = document.createElement("span");
    parte.className = classe;
    parte.textContent = testo;
    voce.append(parte);
  }
  return voce;
}

async function aggiornaEventi() {
  try {
    const nuovi = await (await fetch(`/api/eventi?dopo=${prossimoEvento}`)).json();
    for (const evento of nuovi) {
      prossimoEvento = evento.numero + 1;
      for (const [cosa, tipo] of cambiamenti(evento, ultimoStatoDi.get(evento.telecamera))) {
        $("#registro").prepend(vocePerRegistro(evento.ora, evento.telecamera.toUpperCase(), cosa, tipo));
        $("#registro-vuoto").hidden = true;
      }
      ultimoStatoDi.set(evento.telecamera, evento);
    }
  } catch (errore) { /* riprova al prossimo giro */ }
}

// ---------------------------------------------------------------------------------------------------- archivio
const video = $("#video-archivio");
let eventiSessione = [];

async function caricaArchivio() {
  const sessioni = await (await fetch("/api/archivio")).json();
  const elenco = $("#sessioni");
  elenco.replaceChildren();
  if (!sessioni.length) {
    const vuoto = document.createElement("li");
    vuoto.className = "vuoto";
    vuoto.textContent = "No finished analysis yet: camera sessions appear here when they stop.";
    elenco.append(vuoto);
  }
  for (const sessione of sessioni) {
    const voce = document.createElement("li");
    const bottone = document.createElement("button");
    const nome = document.createElement("span");
    nome.className = "nome-sessione";
    nome.textContent = sessione.nome;
    const dati = document.createElement("span");
    dati.className = "dati-sessione";
    dati.textContent = `${minutiSecondi(sessione.durata_secondi)}`;
    if (sessione.finestre_con_allarme) {
      const allarme = document.createElement("b");
      allarme.textContent = " · alarm";
      dati.append(allarme);
    }
    if (sessione.finestre_con_warning) dati.append(" · weapon");
    bottone.append(nome, dati);
    bottone.addEventListener("click", () => {
      for (const altro of elenco.querySelectorAll("button")) altro.removeAttribute("aria-current");
      bottone.setAttribute("aria-current", "true");
      apriSessione(sessione);
    });
    voce.append(bottone);
    elenco.append(voce);
  }
}

async function apriSessione(sessione) {
  $("#archivio-vuoto").hidden = true;
  video.src = `/archivio/${sessione.numero}/video_annotato.mp4`;
  eventiSessione = await (await fetch(`/archivio/${sessione.numero}/eventi.json`)).json();
  const durata = Math.max(sessione.durata_secondi, 0.1);
  const striscia = $("#striscia-classi");
  for (const vecchio of striscia.querySelectorAll(".tratto, .segno")) vecchio.remove();
  let inizio = 0;
  for (const evento of eventiSessione) {   // finestre sovrapposte: ogni decisione colora dal suo inizio alla prossima
    const fine = evento.fine_secondi;
    const tratto = document.createElement("div");
    tratto.className = "tratto";
    tratto.style.left = `${(inizio / durata) * 100}%`;
    tratto.style.width = `${((fine - inizio) / durata) * 100}%`;
    tratto.style.background = evento.classe ? `var(--c-${evento.classe})` : "var(--linea)";
    striscia.append(tratto);
    for (const [chiave, classe] of [["allarme_violenza", "allarme"], ["warning_arma", "warning"]]) {
      if (!evento[chiave]) continue;
      const segno = document.createElement("div");
      segno.className = `segno ${classe}`;
      segno.style.left = tratto.style.left;
      segno.style.width = tratto.style.width;
      striscia.append(segno);
    }
    inizio = fine;
  }
  const registro = $("#registro-archivio");
  registro.replaceChildren();
  let precedente = null;
  for (const evento of eventiSessione) {
    for (const [cosa, tipo] of cambiamenti(evento, precedente)) {
      const voce = vocePerRegistro(minutiSecondi(evento.fine_secondi), "", cosa, tipo);
      voce.dataset.secondi = evento.fine_secondi;
      voce.addEventListener("click", () => { video.currentTime = Math.max(0, evento.fine_secondi - 1.6); video.play(); });
      registro.append(voce);
    }
    precedente = evento;
  }
  if (!registro.children.length) registro.append(vocePerRegistro("", "", "no alarm, no weapon: all normal", ""));
}

$("#striscia-classi").addEventListener("click", (clic) => {
  const riquadro = clic.currentTarget.getBoundingClientRect();
  if (video.duration) video.currentTime = ((clic.clientX - riquadro.left) / riquadro.width) * video.duration;
});

video.addEventListener("timeupdate", () => {
  if (!video.duration) return;
  $("#testina").style.left = `${(video.currentTime / video.duration) * 100}%`;
  let corrente = null;
  for (const voce of $("#registro-archivio").children) {
    voce.classList.remove("corrente");
    if (Number(voce.dataset.secondi) <= video.currentTime + 0.05) corrente = voce;
  }
  corrente?.classList.add("corrente");
});

// ---------------------------------------------------------------------------------------------------- avvio
aggiornaStato();
aggiornaEventi();
setInterval(aggiornaStato, 700);
setInterval(aggiornaEventi, 1000);
