// Apertura e chiusura: righe tecniche che si scrivono come all'avvio di un sistema. Solo nomi ed etichette
// fattuali (nome del progetto, modelli, classi); i testi descrittivi li scrive Marco.
import { Interactive, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { Scanline } from "../hud/CorniceHud";
import { TEMA_SALA_DI_CONTROLLO, type Tema } from "../tema";

type PropsScenaTitolo = {
  readonly titolo: string;
  readonly righe: readonly string[];
  readonly tema?: Tema;
};

const CARATTERI_AL_SECONDO = 100;

const ScenaTitoloInterna: React.FC<PropsScenaTitolo> = ({ titolo, righe, tema = TEMA_SALA_DI_CONTROLLO }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const caratteriVisibili = Math.floor((frame / fps) * CARATTERI_AL_SECONDO);
  let caratteriUsati = 0;

  return (
    <>
      <div style={{ position: "absolute", inset: 0, background: "#040607" }} />
      <Scanline intensita={0.18} />
      <div style={{ position: "absolute", left: 120, right: 120, top: 210, color: tema.coloreTesto }}>
        <div style={{
          fontFamily: tema.fontTitoli, fontWeight: 700, fontSize: 104, letterSpacing: 12, lineHeight: 1,
          opacity: interpolate(frame, [0, 0.5 * fps], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
          clipPath: `inset(0 ${interpolate(frame, [0, 0.8 * fps], [100, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })}% 0 0)`,
        }}>
          {titolo}
        </div>
        <div style={{ height: 2, marginTop: 18, marginBottom: 26, background: tema.coloreAccento,
          width: `${interpolate(frame, [0.3 * fps, 1.1 * fps], [0, 100], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })}%` }} />
        {righe.map((riga) => {
          const inizio = caratteriUsati;
          caratteriUsati += riga.length;
          const visibile = riga.slice(0, Math.max(0, caratteriVisibili - inizio - 25));
          return (
            <div key={riga} style={{ fontFamily: tema.fontMonospazio, fontSize: 24, letterSpacing: 2, marginBottom: 10, minHeight: 30 }}>
              <span style={{ color: tema.coloreAccento }}>{visibile.length ? "> " : ""}</span>{visibile}
            </div>
          );
        })}
      </div>
    </>
  );
};

export const ScenaTitolo = Interactive.create({
  Component: ScenaTitoloInterna,
  layout: "absolute-fill",
});
