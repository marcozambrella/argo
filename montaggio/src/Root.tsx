import { Composition, Folder } from "remotion";
import { HeroCCTV } from "./HeroCCTV";
import { ScenaAttesa } from "./scene/ScenaAttesa";
import { ScenaClip } from "./scene/ScenaClip";
import { ScenaTitolo } from "./scene/ScenaTitolo";

export const RemotionRoot: React.FC = () => {
  return (
    <>
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
