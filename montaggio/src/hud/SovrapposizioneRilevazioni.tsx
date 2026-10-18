// Disegna sopra il video le rilevazioni VERE di un frame: box con angoli, scheletro a 17 punti, ID della traccia,
// armi e WARNING ARMA (stessa regola di src/sicurezza/demo.py: arma ≥ 0,5 vicino a una persona in ≥ 5 frame su 40).
import type React from "react";
import { FRAME_PER_FINESTRA, OSSA, SOGLIA_CONFIDENZA_ARMA, SOGLIA_VISIBILITA_KEYPOINT, type Tema } from "../tema";
import { frameDiTraccia, type Arma, type Persona, type Rilevazioni } from "../rilevazioni";

const FRAME_MINIMI_CON_ARMA = 5;
const MARGINE_VICINANZA = 0.5;
const ALTEZZA_MINIMA_ETICHETTA = 70;  // px: sotto questa altezza la persona è lontana, niente etichetta (leggibilità)

const armaVicinaAPersona = (arma: Arma, persone: Persona[]): boolean => {
  const centroX = (arma.box[0] + arma.box[2]) / 2;
  const centroY = (arma.box[1] + arma.box[3]) / 2;
  return persone.some(({ box: [x1, y1, x2, y2] }) => {
    const larghezza = x2 - x1;
    const altezza = y2 - y1;
    return centroX >= x1 - MARGINE_VICINANZA * larghezza && centroX <= x2 + MARGINE_VICINANZA * larghezza
      && centroY >= y1 - MARGINE_VICINANZA * altezza && centroY <= y2 + MARGINE_VICINANZA * altezza;
  });
};

export const warningArmaAttivo = (rilevazioni: Rilevazioni, frame: number): boolean => {
  let frameConArma = 0;
  for (let indice = Math.max(0, frame - FRAME_PER_FINESTRA + 1); indice <= frame; indice++) {
    const dati = rilevazioni.frame[indice];
    if (dati?.armi.some((arma) => arma.confidenza >= SOGLIA_CONFIDENZA_ARMA && armaVicinaAPersona(arma, dati.persone))) {
      frameConArma++;
    }
  }
  return frameConArma >= FRAME_MINIMI_CON_ARMA;
};

const AngoliBox: React.FC<{ x: number; y: number; larghezza: number; altezza: number; colore: string; spessore: number }> = (
  { x, y, larghezza, altezza, colore, spessore },
) => {
  const lato = Math.min(18, larghezza / 3, altezza / 3);
  const percorso = [
    `M${x},${y + lato} V${y} H${x + lato}`,
    `M${x + larghezza - lato},${y} H${x + larghezza} V${y + lato}`,
    `M${x + larghezza},${y + altezza - lato} V${y + altezza} H${x + larghezza - lato}`,
    `M${x + lato},${y + altezza} H${x} V${y + altezza - lato}`,
  ].join(" ");
  return <path d={percorso} fill="none" stroke={colore} strokeWidth={spessore} />;
};

export const SovrapposizioneRilevazioni: React.FC<{
  readonly rilevazioni: Rilevazioni;
  readonly frame: number;
  readonly larghezza: number;
  readonly altezza: number;
  readonly tema: Tema;
}> = ({ rilevazioni, frame, larghezza, altezza, tema }) => {
  const dati = rilevazioni.frame[Math.min(frame, rilevazioni.frame.length - 1)];
  if (!dati) return null;
  const fps = rilevazioni.fps;
  // etichette: se due si sovrappongono, quella più a destra sale di una riga
  const LARGHEZZA_ETICHETTA = 168;
  const righeEtichette = new Map<number, number>();
  const occupate: Array<{ x: number; riga: number }> = [];
  [...dati.persone].sort((a, b) => a.box[0] - b.box[0]).forEach((persona) => {
    const x = persona.box[0] * larghezza;
    let riga = 0;
    while (occupate.some((altra) => altra.riga === riga && Math.abs(altra.x - x) < LARGHEZZA_ETICHETTA)) riga++;
    occupate.push({ x, riga });
    righeEtichette.set(persona.id, riga);
  });

  return (
    <svg viewBox={`0 0 ${larghezza} ${altezza}`} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
      {dati.persone.map((persona) => {
        const x = persona.box[0] * larghezza;
        const y = persona.box[1] * altezza;
        const larghezzaBox = (persona.box[2] - persona.box[0]) * larghezza;
        const altezzaBox = (persona.box[3] - persona.box[1]) * altezza;
        const punti = persona.keypoint.map(([px, py, visibilita]) => ({
          x: px * larghezza, y: py * altezza, visibile: visibilita >= SOGLIA_VISIBILITA_KEYPOINT,
        }));
        const puntiVisibili = punti.filter((punto) => punto.visibile).length;
        const secondiDiTraccia = frameDiTraccia(rilevazioni, frame, persona.id) / fps;
        return (
          <g key={persona.id}>
            <rect x={x} y={y} width={larghezzaBox} height={altezzaBox} fill={tema.coloreLinee} opacity={0.04} />
            <AngoliBox x={x} y={y} larghezza={larghezzaBox} altezza={altezzaBox} colore={tema.coloreLinee}
              spessore={altezzaBox > ALTEZZA_MINIMA_ETICHETTA ? 2 : 1.2} />
            {OSSA.map(([inizio, fine]) => (punti[inizio].visibile && punti[fine].visibile ? (
              <line key={`${inizio}-${fine}`} x1={punti[inizio].x} y1={punti[inizio].y} x2={punti[fine].x} y2={punti[fine].y}
                stroke={tema.coloreAccento} strokeWidth={altezzaBox > ALTEZZA_MINIMA_ETICHETTA ? 2 : 1} opacity={0.9} />
            ) : null))}
            {punti.map((punto, indice) => (punto.visibile ? (
              <circle key={indice} cx={punto.x} cy={punto.y} r={altezzaBox > ALTEZZA_MINIMA_ETICHETTA ? 2.6 : 1.4}
                fill={tema.coloreLinee} />
            ) : null))}
            {altezzaBox > ALTEZZA_MINIMA_ETICHETTA ? (
              <g transform={`translate(${x}, ${Math.max(0, y - 34 - 34 * (righeEtichette.get(persona.id) ?? 0))})`}>
                <rect width={LARGHEZZA_ETICHETTA - 6} height={30} fill={tema.sfondoPannelli} />
                <rect width={4} height={30} fill={tema.coloreAccento} />
                <text x={10} y={13} fill={tema.coloreAccento} fontFamily={tema.fontMonospazio} fontSize={13}>
                  {`ID ${String(persona.id).padStart(3, "0")}`}
                </text>
                <text x={10} y={26} fill={tema.coloreTesto} fontFamily={tema.fontMonospazio} fontSize={10.5} opacity={0.85}>
                  {`CONF ${persona.confidenza.toFixed(2)} · ${puntiVisibili}/17 · ${secondiDiTraccia.toFixed(1)} s`}
                </text>
              </g>
            ) : null}
          </g>
        );
      })}
      {dati.armi.filter((arma) => arma.confidenza >= SOGLIA_CONFIDENZA_ARMA).map((arma, indice) => {
        const x = arma.box[0] * larghezza;
        const y = arma.box[1] * altezza;
        return (
          <g key={`arma-${indice}`}>
            <rect x={x} y={y} width={(arma.box[2] - arma.box[0]) * larghezza} height={(arma.box[3] - arma.box[1]) * altezza}
              fill="none" stroke={tema.coloreAllarme} strokeWidth={2.5} />
            <text x={x} y={y - 6} fill={tema.coloreAllarme} fontFamily={tema.fontMonospazio} fontSize={13}>
              {`${arma.classe.toUpperCase()} ${arma.confidenza.toFixed(2)}`}
            </text>
          </g>
        );
      })}
    </svg>
  );
};
