import { Composition, Folder } from "remotion";
import { ArgoApertura } from "./argo/ArgoApertura";
import { CARTELLI, CASI } from "./argo/scaletta";

// durata: unità di 2 battiti a 86,1 bpm (35 frame), intro 1 + città 2 ciascuna + Argo 2 + casi 4 ciascuno, + 1,8 s di finale
const FRAME_ARGO = 35 * (1 + 2 * CARTELLI.length + 2 + 4 * CASI.length) + 45;
import { HeroCCTV } from "./HeroCCTV";
import { ScenaAttesa } from "./scene/ScenaAttesa";
import { ScenaClip } from "./scene/ScenaClip";
import { ScenaTitolo } from "./scene/ScenaTitolo";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Folder name="Argo">
        {/* D61: apertura brutalista + casi d'uso, ~45 s. Musica: presentation/musica/candidati (D62), tratto da analizza_musica.py */}
        <Composition
          id="ArgoApertura"
          component={ArgoApertura}
          durationInFrames={FRAME_ARGO}
          fps={25}
          width={1920}
          height={1080}
          defaultProps={{ brano: "clash_defiant.mp3", inizioBrano: 127.34, bpm: 86.1 }}
        />
      </Folder>
      <Composition
        id="HeroCCTV"
        component={HeroCCTV}
        durationInFrames={675}
        fps={25}
        width={1280}
        height={720}
        defaultProps={{ statoClassificatore: "CLASSIFICATORE: IN ADDESTRAMENTO" }}
      />
      <Folder name="Scene">
        <Composition
          id="ScenaClip"
          component={ScenaClip}
          durationInFrames={175}
          fps={25}
          width={1280}
          height={720}
          defaultProps={{
            nomeClip: "normale_vicolo_roma",
            secondoInizio: 2,
            casoDUso: "NORMALE",
            numeroCamera: "CAM 01",
            creditoClip: "VIDEO: PEXELS #31524454",
            statoClassificatore: "CLASSIFICATORE: IN ADDESTRAMENTO",
          }}
        />
        <Composition
          id="ScenaAttesa"
          component={ScenaAttesa}
          durationInFrames={75}
          fps={25}
          width={1280}
          height={720}
          defaultProps={{ casoDUso: "VIOLENZA", numeroCamera: "CAM 02" }}
        />
        <Composition
          id="ScenaTitolo"
          component={ScenaTitolo}
          durationInFrames={75}
          fps={25}
          width={1280}
          height={720}
          defaultProps={{ titolo: "ROMA SAFE CCTV", righe: ["normale · violenza · violenza con arma · scippo"] }}
        />
      </Folder>
    </>
  );
};
