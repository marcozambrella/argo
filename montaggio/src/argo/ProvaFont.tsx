// Prova di Megrim "in grassetto" (scelta di Marco): Megrim ha un solo peso, lo spessore si ottiene con un contorno
// dello stesso colore (-webkit-text-stroke).
import { loadFont } from "@remotion/google-fonts/Megrim";
import { AbsoluteFill } from "remotion";

const { fontFamily: megrim } = loadFont();
const SPESSORI = [["normale", 0], ["grassetto 1 (contorno 4 px)", 4], ["grassetto 2 (contorno 8 px)", 8], ["grassetto 3 (contorno 12 px)", 12]] as const;

export const ProvaFont: React.FC = () => (
  <AbsoluteFill style={{ background: "#0b0b0b", display: "grid", gridTemplateColumns: "1fr 1fr", gridTemplateRows: "1fr 1fr", gap: 6, padding: 6 }}>
    {SPESSORI.map(([nome, contorno], indice) => {
      const colore = indice % 3 ? "#0b0b0b" : "#FFE600";
      return (
        <div key={nome} style={{ background: indice % 3 ? "#FFE600" : "#141414", position: "relative", display: "flex",
          flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 10 }}>
          <span style={{ fontFamily: megrim, fontSize: 190, letterSpacing: 14, color: colore, WebkitTextStroke: `${contorno}px ${colore}` }}>ARGO</span>
          <span style={{ fontFamily: megrim, fontSize: 110, letterSpacing: 6, color: colore, WebkitTextStroke: `${contorno * 0.6}px ${colore}` }}>NEW YORK</span>
          <span style={{ position: "absolute", left: 14, top: 8, fontFamily: "monospace", fontSize: 22, color: indice % 3 ? "#0b0b0b" : "#f4f1ea" }}>Megrim {nome}</span>
        </div>
      );
    })}
  </AbsoluteFill>
);
