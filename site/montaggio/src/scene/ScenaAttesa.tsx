// SEGNAPOSTO: caso d'uso senza riprese adatte (violenza, scippo). Si sostituisce con le scene girate dal team.
import { Interactive, interpolate, random, useCurrentFrame, useVideoConfig } from "remotion";
import { Scanline, spostamentoGlitch } from "../hud/CorniceHud";
import { TEMA_SALA_DI_CONTROLLO, type Tema } from "../tema";

type PropsScenaAttesa = {
  readonly casoDUso: string;
  readonly numeroCamera: string;
  readonly tema?: Tema;
};

const ScenaAttesaInterna: React.FC<PropsScenaAttesa> = ({ casoDUso, numeroCamera, tema = TEMA_SALA_DI_CONTROLLO }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <>
      {/* rumore video deterministico */}
      <div style={{ position: "absolute", inset: 0, background: "#050607" }}>
        {Array.from({ length: 48 }, (_, riga) => (
          <div key={riga} style={{
            position: "absolute", left: 0, right: 0, top: riga * 15, height: 15,
            background: tema.coloreLinee, opacity: random(`rumore-${frame}-${riga}`) * 0.07,
          }} />
        ))}
      </div>
      <Scanline intensita={0.2} />
      <div style={{
        position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", gap: 18, color: tema.coloreTesto, translate: spostamentoGlitch(frame, casoDUso),
        opacity: interpolate(frame, [0, 0.3 * fps], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
      }}>
        <div style={{ fontFamily: tema.fontMonospazio, fontSize: 20, letterSpacing: 6, opacity: 0.7 }}>
          {`${numeroCamera} · SEGNALE ASSENTE`}
        </div>
        <div style={{ fontFamily: tema.fontMonospazio, fontSize: 22, letterSpacing: 8, color: tema.coloreAccento }}>
          CASO D&apos;USO
        </div>
        <div style={{ fontFamily: tema.fontTitoli, fontWeight: 700, fontSize: 96, letterSpacing: 10, lineHeight: 1 }}>
          {casoDUso}
        </div>
        <div style={{
          fontFamily: tema.fontMonospazio, fontSize: 18, letterSpacing: 3, padding: "8px 16px",
          border: `1px dashed ${tema.coloreLinee}`, opacity: 0.8,
        }}>
          SEGNAPOSTO · IN ATTESA DELLE RIPRESE DEL TEAM
        </div>
      </div>
    </>
  );
};

export const ScenaAttesa = Interactive.create({
  Component: ScenaAttesaInterna,
  layout: "absolute-fill",
});
