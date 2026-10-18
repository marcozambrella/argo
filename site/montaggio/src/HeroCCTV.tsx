// Video hero del sito: montaggio dei casi d'uso con le rilevazioni vere della pipeline.
// Tema parametrico: una sola composizione, varianti di colore/font renderizzate quando Marco sceglie lo stile.
import { Track, useVideoConfig } from "remotion";
import { ScenaAttesa } from "./scene/ScenaAttesa";
import { ScenaClip } from "./scene/ScenaClip";
import { ScenaTitolo } from "./scene/ScenaTitolo";
import { TEMA_SALA_DI_CONTROLLO, type Tema } from "./tema";

type PropsHero = {
  readonly tema?: Tema;
  readonly statoClassificatore: string;
};

export const HeroCCTV: React.FC<PropsHero> = ({ tema = TEMA_SALA_DI_CONTROLLO, statoClassificatore }) => {
  const { fps } = useVideoConfig();
  return (
    <Track name="Scene">
      <ScenaTitolo name="Apertura" durationInFrames={3 * fps} tema={tema} titolo="ROMA SAFE CCTV"
        righe={["YOLO26s-pose · TrackTrack · detector armi", "finestre di 40 frame · Graph-Transformer",
          "normale · violenza · violenza con arma · scippo"]} />
      <ScenaClip name="Normale · vicolo" from={3 * fps} durationInFrames={7 * fps} tema={tema}
        nomeClip="normale_vicolo_roma" secondoInizio={2} casoDUso="NORMALE" numeroCamera="CAM 01"
        creditoClip="VIDEO: PEXELS #31524454" statoClassificatore={statoClassificatore} />
      <ScenaAttesa name="Violenza (segnaposto)" from={10 * fps} durationInFrames={3 * fps} tema={tema}
        casoDUso="VIOLENZA" numeroCamera="CAM 02" />
      <ScenaClip name="Arma · vicolo graffiti" from={13 * fps} durationInFrames={8 * fps} tema={tema}
        nomeClip="arma_vicolo_graffiti" secondoInizio={12} casoDUso="ARMA" numeroCamera="CAM 03"
        creditoClip="VIDEO: PEXELS #7230785" statoClassificatore={statoClassificatore} />
      <ScenaAttesa name="Scippo (segnaposto)" from={21 * fps} durationInFrames={3 * fps} tema={tema}
        casoDUso="SCIPPO" numeroCamera="CAM 04" />
      <ScenaTitolo name="Chiusura" from={24 * fps} durationInFrames={3 * fps} tema={tema} titolo="ROMA SAFE CCTV"
        righe={["[SEGNAPOSTO · TESTO DI MARCO]"]} />
    </Track>
  );
};
