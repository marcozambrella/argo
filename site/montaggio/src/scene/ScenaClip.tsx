// Una clip reale con le sue rilevazioni vere e il titolo del caso d'uso.
// Il titolo è un CAPITOLO del montaggio, non una risposta del classificatore (che non è ancora addestrato).
import { Video } from "@remotion/media";
import { Easing, Interactive, interpolate, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { CorniceHud, Scanline, spostamentoGlitch } from "../hud/CorniceHud";
import { SovrapposizioneRilevazioni, warningArmaAttivo } from "../hud/SovrapposizioneRilevazioni";
import { useRilevazioni } from "../rilevazioni";
import { TEMA_SALA_DI_CONTROLLO, type Tema } from "../tema";

type PropsScenaClip = {
  readonly nomeClip: string;            // file in public/clip/ e public/rilevazioni/ (senza estensione)
  readonly secondoInizio: number;       // da quale secondo della clip partire
  readonly casoDUso: string;            // es. "NORMALE"
  readonly numeroCamera: string;
  readonly creditoClip: string;
  readonly statoClassificatore: string;
  readonly tema?: Tema;
};

const ScenaClipInterna: React.FC<PropsScenaClip> = ({
  nomeClip, secondoInizio, casoDUso, numeroCamera, creditoClip, statoClassificatore, tema = TEMA_SALA_DI_CONTROLLO,
}) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const rilevazioni = useRilevazioni(`${nomeClip}.json`);
  const frameClip = Math.round(secondoInizio * fps) + frame;

  return (
    <>
      <div style={{ position: "absolute", inset: 0, translate: spostamentoGlitch(frame, nomeClip) }}>
        <Video src={staticFile(`clip/${nomeClip}.mp4`)} trimBefore={Math.round(secondoInizio * fps)} muted
          objectFit="cover" style={{ position: "absolute", width: "100%", height: "100%", filter: "saturate(0.55) contrast(1.08)" }} />
        {rilevazioni ? (
          <SovrapposizioneRilevazioni rilevazioni={rilevazioni} frame={frameClip} larghezza={width} altezza={height} tema={tema} />
        ) : null}
      </div>
      <Scanline />
      <CorniceHud tema={tema} numeroCamera={numeroCamera} frameClip={frameClip} creditoClip={creditoClip}
        statoClassificatore={statoClassificatore}
        warningArma={rilevazioni ? warningArmaAttivo(rilevazioni, frameClip) : false} />
      {/* titolo del capitolo: entra, resta 2 s, poi si riduce in alto al centro */}
      <div style={{
        position: "absolute", left: "50%", top: interpolate(frame, [2.2 * fps, 2.8 * fps], [300, 36], {
          extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.bezier(0.65, 0, 0.35, 1),
        }),
        translate: "-50% 0", textAlign: "center", color: tema.coloreTesto,
        opacity: interpolate(frame, [0, 0.4 * fps], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
        scale: interpolate(frame, [2.2 * fps, 2.8 * fps], [1, 0.42], {
          extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.bezier(0.65, 0, 0.35, 1),
        }),
      }}>
        <div style={{ fontFamily: tema.fontMonospazio, fontSize: 22, letterSpacing: 8, color: tema.coloreAccento }}>
          CASO D&apos;USO
        </div>
        <div style={{
          fontFamily: tema.fontTitoli, fontWeight: 700, fontSize: 96, letterSpacing: 10, lineHeight: 1,
          padding: "6px 24px", background: tema.sfondoPannelli, borderTop: `2px solid ${tema.coloreAccento}`,
          borderBottom: `2px solid ${tema.coloreAccento}`,
        }}>
          {casoDUso}
        </div>
      </div>
    </>
  );
};

export const ScenaClip = Interactive.create({
  Component: ScenaClipInterna,
  layout: "absolute-fill",
});
