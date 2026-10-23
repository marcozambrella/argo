// Pulviscolo nel fascio del lampione: granelli che fluttuano lenti e brillano solo dentro la luce.
// Il canvas ha la stessa forma del cono (clip-path nel CSS); qui i granelli sono più luminosi vicino alla lampada
// e al centro del fascio. Si ferma quando il titolo non si vede; con movimento ridotto resta un'immagine ferma.
import { movimentoRidotto } from "./comune.js?v=20261023175911";

const GRANELLI_PER_MEGAPIXEL = 260;
// forma del fascio in frazioni del canvas: larghezza in alto (bocca della campana) e in basso
const MEZZA_LARGHEZZA_ALTO = 0.031, MEZZA_LARGHEZZA_BASSO = 0.35;

export function avviaPulviscolo(tela) {
  const contesto = tela.getContext("2d");
  let granelli = [], larghezza = 0, altezza = 0, rapporto = 1, attivo = true, ultimoTempo = 0;

  function creaGranello(casuale = Math.random) {
    return {
      x: casuale(), y: Math.pow(casuale(), 1.4),                 // più fitti vicino alla lampada
      raggio: 0.5 + casuale() * 1.3, fase: casuale() * Math.PI * 2,
      velocitaX: (casuale() - 0.5) * 0.004, velocitaY: (casuale() - 0.35) * 0.003,
    };
  }

  function ridimensiona() {
    const rettangolo = tela.getBoundingClientRect();
    rapporto = Math.min(devicePixelRatio, 2);
    larghezza = rettangolo.width; altezza = rettangolo.height;
    tela.width = Math.round(larghezza * rapporto); tela.height = Math.round(altezza * rapporto);
    const quanti = Math.min(420, Math.round((larghezza * altezza / 1e6) * GRANELLI_PER_MEGAPIXEL));
    granelli = Array.from({ length: quanti }, () => creaGranello());
    if (movimentoRidotto()) requestAnimationFrame(disegna);   // il ciclo è fermo: ridisegno una volta
  }

  function luceIn(granello) {
    const mezzaLarghezza = MEZZA_LARGHEZZA_ALTO + (MEZZA_LARGHEZZA_BASSO - MEZZA_LARGHEZZA_ALTO) * granello.y;
    const dalCentro = Math.abs(granello.x - 0.5) / mezzaLarghezza;    // 0 al centro del fascio, 1 sul bordo
    if (dalCentro >= 1) return 0;
    return (1 - dalCentro * dalCentro) * Math.pow(1 - granello.y, 1.6);
  }

  function disegna(tempo) {
    const passo = Math.min(0.05, (tempo - ultimoTempo) / 1000 || 0);
    ultimoTempo = tempo;
    contesto.setTransform(rapporto, 0, 0, rapporto, 0, 0);
    contesto.clearRect(0, 0, larghezza, altezza);
    for (const granello of granelli) {
      if (passo > 0) {
        granello.x += (granello.velocitaX + Math.sin(tempo / 1700 + granello.fase) * 0.0025) * passo;
        granello.y += (granello.velocitaY + Math.cos(tempo / 2300 + granello.fase) * 0.002) * passo;
        if (granello.y < 0 || granello.y > 1 || granello.x < 0 || granello.x > 1) Object.assign(granello, creaGranello());
      }
      const luce = luceIn(granello);
      if (luce < 0.02) continue;
      const scintillio = 0.55 + 0.45 * Math.sin(tempo / 600 + granello.fase * 3);
      contesto.fillStyle = `rgba(255, 236, 196, ${(luce * scintillio * 0.9).toFixed(3)})`;
      contesto.beginPath();
      contesto.arc(granello.x * larghezza, granello.y * altezza, granello.raggio, 0, Math.PI * 2);
      contesto.fill();
    }
    if (attivo && !movimentoRidotto()) requestAnimationFrame(disegna);
  }

  new ResizeObserver(ridimensiona).observe(tela);
  new IntersectionObserver(([voce]) => {
    const eraAttivo = attivo;
    attivo = voce.isIntersecting;
    if (attivo && !eraAttivo) requestAnimationFrame(disegna);
  }).observe(tela.closest(".hero"));
  ridimensiona();
  requestAnimationFrame(disegna);
}
