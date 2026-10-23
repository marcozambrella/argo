// Argo — il pezzo di Claude nel video (D61): apertura brutalista con le città e i numeri veri, l'occhio di Argo,
// i casi d'uso con l'uscita vera del sistema. Tagli sulla griglia della musica: un'"unità" ≈ 1,6 s (una battuta a
// 152 bpm; con brani più lenti si prendono meno battiti per unità).
import { Audio, Video } from "@remotion/media";
import { loadFont as caricaAnton } from "@remotion/google-fonts/Anton";
import { loadFont as caricaMegrim } from "@remotion/google-fonts/Megrim";
import { loadFont as caricaArchivoBlack } from "@remotion/google-fonts/ArchivoBlack";
import { loadFont as caricaPlexMono } from "@remotion/google-fonts/IBMPlexMono";
import { AbsoluteFill, Easing, interpolate, Sequence, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { CARTELLI, CASI, CASI_VETRINA, SECONDI_VETRINA, type Cartello, type CasoDUso } from "./scaletta";
import { decisioneAl, useEventi } from "./eventi";

const { fontFamily: anton } = caricaAnton("normal", { weights: ["400"], subsets: ["latin"] });
// D63: ARGO e le città in Megrim "grassetto 1" (Megrim ha un solo peso: contorno dello stesso colore, 0,021 em)
const { fontFamily: megrim } = caricaMegrim();
const grassetto = (colore: string) => ({ fontFamily: megrim, WebkitTextStroke: `0.021em ${colore}` });
const { fontFamily: archivo } = caricaArchivoBlack("normal", { weights: ["400"], subsets: ["latin"] });
const { fontFamily: mono } = caricaPlexMono("normal", { weights: ["400", "600"], subsets: ["latin"] });

const GIALLO = "#FFE600";
const NERO = "#0b0b0b";
const BIANCO = "#f4f1ea";
const ROSSO = "#ff2d1a";
const fisso = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;
const NOMI_CLASSI: Record<string, string> = {   // D64: tutto in inglese
  normale: "NORMAL", violenza: "VIOLENCE", violenza_con_arma: "ARMED VIOLENCE", scippo: "SNATCHING",
};

export type PropsArgoApertura = {
  readonly brano: string;          // file in public/argo/
  readonly inizioBrano: number;    // secondi: il tratto più energico (presentation/analizza_musica.py)
  readonly bpm: number;
};

/** Frame di un'unità di montaggio: il numero intero di battiti più vicino a 1,6 s. */
const frameUnita = (bpm: number, fps: number): number => {
  const battito = 60 / bpm;
  return Math.round(Math.max(1, Math.round(1.6 / battito)) * battito * fps);
};

// ------------------------------------------------------------------------------------------------- pezzi comuni
const Lampo: React.FC = () => {
  const frame = useCurrentFrame();
  return <AbsoluteFill style={{ background: BIANCO, opacity: interpolate(frame, [0, 3], [0.85, 0], fisso) }} />;
};

const OcchioArgo: React.FC<{ readonly lato: number; readonly colore: string; readonly apertura?: number }> = ({
  lato, colore, apertura = 1,
}) => (
  <svg width={lato} height={lato} viewBox="0 0 64 64" style={{ transform: `scaleY(${apertura})` }}>
    <path d="M4 32 C17 13 47 13 60 32 C47 51 17 51 4 32 Z" fill="none" stroke={colore} strokeWidth={3.2} strokeLinejoin="round" />
    <circle cx={32} cy={32} r={12} fill="none" stroke={colore} strokeWidth={2.6} />
    {Array.from({ length: 12 }, (_, i) => {
      const angolo = (2 * Math.PI * i) / 12 - Math.PI / 2;
      return <circle key={i} cx={32 + 8 * Math.cos(angolo)} cy={32 + 8 * Math.sin(angolo)} r={1.25} fill={colore} />;
    })}
    <circle cx={32} cy={32} r={4.2} fill={ROSSO} />
  </svg>
);

const numeroCheSale = (numero: string, avanzamento: number): string => {
  const cifre = numero.replace(/[.,]/g, "");
  const trovato = cifre.match(/^(\d+)(%?)$/);
  if (!trovato) return numero;                       // "1ª", "3ª": si mostra com'è
  const valore = Math.round(Number(trovato[1]) * avanzamento);
  return `${String(valore).replace(/\B(?=(\d{3})+(?!\d))/g, ",")}${trovato[2]}`;   // in inglese: 71,867
};

// ------------------------------------------------------------------------------------------------- scene
const Intro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const secondi = frame / fps;
  const codice = `00:00:${String(Math.floor(secondi)).padStart(2, "0")}:${String(frame % fps).padStart(2, "0")}`;
  return (
    <AbsoluteFill style={{ background: NERO, color: BIANCO, fontFamily: mono }}>
      <div style={{ position: "absolute", left: 80, top: 70, fontSize: 34, letterSpacing: 4, display: "flex", gap: 18, alignItems: "center" }}>
        <span style={{ width: 26, height: 26, borderRadius: 13, background: ROSSO, opacity: Math.floor(frame / 6) % 2 ? 1 : 0.2 }} />
        REC · CAM 01
      </div>
      <div style={{ position: "absolute", right: 80, top: 70, fontSize: 34, letterSpacing: 4 }}>{codice}</div>
      <div style={{ position: "absolute", left: 80, bottom: 80, fontSize: 30, letterSpacing: 6, opacity: 0.7 }}>
        EVERY DAY, IN EVERY CITY
      </div>
    </AbsoluteFill>
  );
};

const ScenaCitta: React.FC<{ readonly cartello: Cartello; readonly indice: number; readonly unita: number }> = ({
  cartello, indice, unita,
}) => {
  const frame = useCurrentFrame();
  const chiaro = indice % 2 === 0;
  const fondo = chiaro ? GIALLO : NERO;
  const inchiostro = chiaro ? NERO : GIALLO;
  const testo = chiaro ? NERO : BIANCO;
  const colpo = interpolate(frame, [0, 4], [1.18, 1], { ...fisso, easing: Easing.out(Easing.cubic) });
  const avanzamento = interpolate(frame, [unita * 0.25, unita * 0.9], [0, 1], { ...fisso, easing: Easing.out(Easing.quad) });
  const entra = (ritardo: number) => interpolate(frame, [ritardo, ritardo + 4], [0, 1], fisso);
  return (
    <AbsoluteFill style={{ background: fondo, color: testo, overflow: "hidden" }}>
      <div style={{ position: "absolute", left: 0, right: 0, top: 0, height: 92, borderBottom: `14px solid ${inchiostro}`,
        display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0 80px",
        fontFamily: mono, fontWeight: 600, fontSize: 30, letterSpacing: 6, color: inchiostro }}>
        <span>ARGO</span>
        <span>{String(indice + 1).padStart(2, "0")} / {String(CARTELLI.length).padStart(2, "0")}</span>
      </div>
      {/* indice delle città a destra, quella corrente piena */}
      <div style={{ position: "absolute", right: 0, top: 106, bottom: 0, width: 300, borderLeft: `14px solid ${inchiostro}`,
        display: "flex", flexDirection: "column" }}>
        {CARTELLI.map((altro, numero) => (
          <div key={altro.citta} style={{ flex: 1, display: "flex", alignItems: "center", padding: "0 28px",
            fontFamily: megrim, WebkitTextStroke: "0.021em currentColor", fontSize: 40, letterSpacing: 2,
            background: numero === indice ? inchiostro : "transparent", color: numero === indice ? fondo : inchiostro,
            borderBottom: numero < CARTELLI.length - 1 ? `6px solid ${inchiostro}` : "none",
            opacity: numero <= indice ? 1 : 0.35 }}>
            {altro.citta}
          </div>
        ))}
      </div>
      <div style={{ position: "absolute", left: 70, top: 120, ...grassetto(testo), fontSize: 250, lineHeight: 0.95,
        letterSpacing: 6, transform: `scale(${colpo})`, transformOrigin: "left top", whiteSpace: "nowrap" }}>
        {cartello.citta}
      </div>
      <div style={{ position: "absolute", left: 76, top: 430, fontFamily: anton, fontSize: 250, lineHeight: 1,
        color: inchiostro === NERO ? NERO : GIALLO, opacity: entra(unita * 0.2) }}>
        {numeroCheSale(cartello.numero, avanzamento)}
      </div>
      <div style={{ position: "absolute", left: 80, top: 720, width: 1360, fontFamily: archivo, fontSize: 50,
        lineHeight: 1.12, textTransform: "uppercase", opacity: entra(unita * 0.55),
        translate: `0 ${interpolate(frame, [unita * 0.55, unita * 0.55 + 6], [30, 0], fisso)}px` }}>
        {cartello.testo}
      </div>
      <div style={{ position: "absolute", left: 80, bottom: 46, fontFamily: mono, fontSize: 24, letterSpacing: 1,
        opacity: entra(unita * 0.9) * 0.85 }}>
        SOURCE: {cartello.fonte}
      </div>
      <Sequence durationInFrames={4}><Lampo /></Sequence>
    </AbsoluteFill>
  );
};

const RivelaArgo: React.FC<{ readonly unita: number }> = ({ unita }) => {
  const frame = useCurrentFrame();
  const apertura = interpolate(frame, [0, unita * 0.4], [0.04, 1], { ...fisso, easing: Easing.out(Easing.back(1.6)) });
  const scritta = interpolate(frame, [unita * 0.45, unita * 0.6], [0, 1], fisso);
  return (
    <AbsoluteFill style={{ background: NERO, alignItems: "center", justifyContent: "center" }}>
      <OcchioArgo lato={330} colore={BIANCO} apertura={apertura} />
      <div style={{ ...grassetto(GIALLO), fontSize: 330, lineHeight: 0.9, color: GIALLO, letterSpacing: 40, marginTop: -10,
        opacity: scritta, transform: `scale(${interpolate(scritta, [0, 1], [1.12, 1])})` }}>
        ARGO
      </div>
      <div style={{ fontFamily: mono, fontWeight: 600, fontSize: 34, letterSpacing: 8, color: BIANCO, marginTop: 30,
        opacity: interpolate(frame, [unita * 1.0, unita * 1.2], [0, 1], fisso) }}>
        VIOLENCE · WEAPONS · SNATCHING — FROM CCTV CAMERAS
      </div>
      <Sequence durationInFrames={4}><Lampo /></Sequence>
    </AbsoluteFill>
  );
};

const PannelloDecisione: React.FC<{ readonly nome: string; readonly secondi: number }> = ({ nome, secondi }) => {
  const frame = useCurrentFrame();
  const eventi = useEventi(nome);
  const decisione = eventi ? decisioneAl(eventi, secondi) : null;
  const lampeggio = Math.floor(frame / 5) % 2 === 0;
  return (
    <div style={{ position: "absolute", right: 60, top: 60, width: 560, background: "rgba(11,11,11,0.9)",
      border: `8px solid ${GIALLO}`, padding: "26px 30px", color: BIANCO, fontFamily: mono }}>
      <div style={{ display: "flex", alignItems: "center", gap: 14, fontSize: 24, letterSpacing: 5, fontWeight: 600, color: GIALLO }}>
        <OcchioArgo lato={44} colore={GIALLO} /> ARGO · ANALYSIS
      </div>
      <div style={{ fontFamily: anton, fontSize: 76, lineHeight: 1.05, marginTop: 14,
        color: decisione?.classe && decisione.classe !== "normale" ? (decisione.classe === "violenza_con_arma" ? ROSSO : GIALLO) : BIANCO }}>
        {decisione?.classe ? NOMI_CLASSI[decisione.classe] : "WAITING…"}
      </div>
      {["normale", "violenza", "violenza_con_arma", "scippo"].map((classe) => {
        const valore = decisione?.probabilita?.[classe] ?? 0;
        return (
          <div key={classe} style={{ display: "grid", gridTemplateColumns: "200px 1fr 80px", alignItems: "center", gap: 12,
            fontSize: 22, marginTop: 10 }}>
            <span>{NOMI_CLASSI[classe].replace("ARMED VIOLENCE", "ARMED")}</span>
            <span style={{ height: 16, background: "#2a2a2a" }}>
              <span style={{ display: "block", height: "100%", width: `${valore * 100}%`, background: GIALLO }} />
            </span>
            <span style={{ textAlign: "right" }}>{Math.round(valore * 100)}%</span>
          </div>
        );
      })}
      <div style={{ display: "flex", gap: 12, marginTop: 22 }}>
        <div style={{ flex: 1, padding: "10px 0", textAlign: "center", fontFamily: anton, fontSize: 28, letterSpacing: 1, whiteSpace: "nowrap",
          background: decisione?.warning_arma && lampeggio ? GIALLO : "transparent", color: decisione?.warning_arma ? NERO : "#555",
          border: `4px solid ${decisione?.warning_arma ? GIALLO : "#444"}` }}>
          WEAPON WARNING
        </div>
        <div style={{ flex: 1, padding: "10px 0", textAlign: "center", fontFamily: anton, fontSize: 28, letterSpacing: 1, whiteSpace: "nowrap",
          background: decisione?.allarme_violenza && lampeggio ? ROSSO : "transparent", color: decisione?.allarme_violenza ? BIANCO : "#555",
          border: `4px solid ${decisione?.allarme_violenza ? ROSSO : "#444"}` }}>
          ALARM
        </div>
      </div>
    </div>
  );
};

const ScenaCaso: React.FC<{ readonly caso: CasoDUso; readonly numero: number }> = ({ caso, numero }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const secondi = caso.inizio + frame / fps;
  const entra = interpolate(frame, [0, 6], [-120, 0], { ...fisso, easing: Easing.out(Easing.cubic) });
  // i video dei dataset sono scheletri su nero puro: stesso fondo, così non si vede il bordo del video
  const fondo = caso.nome.startsWith("dataset_scheletri") ? "#000" : NERO;
  return (
    <AbsoluteFill style={{ background: fondo }}>
      <Video src={staticFile(`argo/${caso.nome}.mp4`)} trimBefore={Math.round(caso.inizio * fps)} muted objectFit={caso.adatta}
        style={{ position: "absolute", width: "100%", height: "100%", filter: "contrast(1.06) saturate(0.85)",
          ...(caso.zoom ? { transformOrigin: "0 0", transform:   // il punto (cx, cy) finisce in (0,45; 0,55)
            `translate(${(0.45 - caso.zoom.scala * caso.zoom.cx) * 100}%, ${(0.55 - caso.zoom.scala * caso.zoom.cy) * 100}%) scale(${caso.zoom.scala})` } : {}) }} />
      <div style={{ position: "absolute", left: 60, top: 60, translate: `${entra}px 0` }}>
        <div style={{ background: GIALLO, color: NERO, padding: "12px 28px 6px" }}>
          <div style={{ fontFamily: mono, fontWeight: 600, fontSize: 26, letterSpacing: 6 }}>
            USE CASE {String(numero).padStart(2, "0")}
          </div>
          <div style={{ fontFamily: anton, fontSize: 150, lineHeight: 1 }}>{caso.capitolo}</div>
        </div>
        <div style={{ display: "inline-block", background: NERO, color: BIANCO, fontFamily: anton, fontSize: 64,
          padding: "6px 28px", letterSpacing: 2 }}>
          {caso.titolo}
        </div>
      </div>
      <PannelloDecisione nome={caso.nome} secondi={secondi} />
      <div style={{ position: "absolute", left: 60, bottom: 40, background: "rgba(11,11,11,0.85)", color: BIANCO,
        fontFamily: mono, fontSize: 22, padding: "8px 16px" }}>
        {caso.credito}
      </div>
      <Sequence durationInFrames={4}><Lampo /></Sequence>
    </AbsoluteFill>
  );
};

const Finale: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{ background: NERO, alignItems: "center", justifyContent: "center", gap: 20 }}>
      <OcchioArgo lato={160} colore={BIANCO} apertura={interpolate(frame, [0, 6], [1, 0.08], fisso)} />
      <div style={{ ...grassetto(GIALLO), fontSize: 120, letterSpacing: 24, color: GIALLO }}>ARGO</div>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------------------------------------- composizione
export const ArgoApertura: React.FC<PropsArgoApertura> = ({ brano, inizioBrano, bpm }) => {
  const { fps, durationInFrames } = useVideoConfig();
  const unita = frameUnita(bpm, fps);
  const inizioCasi = unita * (1 + CARTELLI.length * 2 + 2);
  const unitaPerCaso = unita < 1.5 * fps ? 4 : 3;   // con brani lenti l'unità è più corta: i casi d'uso restano ~5 s
  const inizioFinale = inizioCasi + unita * unitaPerCaso * CASI.length;
  return (
    <AbsoluteFill style={{ background: NERO }}>
      <Audio src={staticFile(`argo/${brano}`)} trimBefore={Math.round(inizioBrano * fps)}
        volume={(f) => interpolate(f, [durationInFrames - fps * 1.2, durationInFrames - 2], [1, 0], fisso)} />
      <Sequence durationInFrames={unita}><Intro /></Sequence>
      {CARTELLI.map((cartello, indice) => (
        <Sequence key={cartello.citta} from={unita * (1 + indice * 2)} durationInFrames={unita * 2}>
          <ScenaCitta cartello={cartello} indice={indice} unita={unita} />
        </Sequence>
      ))}
      <Sequence from={unita * (1 + CARTELLI.length * 2)} durationInFrames={unita * 2}><RivelaArgo unita={unita} /></Sequence>
      {CASI.map((caso, indice) => (
        <Sequence key={caso.nome} from={inizioCasi + unita * unitaPerCaso * indice} durationInFrames={unita * unitaPerCaso}>
          <ScenaCaso caso={caso} numero={indice + 1} />
        </Sequence>
      ))}
      <Sequence from={inizioFinale}><Finale /></Sequence>
    </AbsoluteFill>
  );
};

// ------------------------------------------------------------------------------------------------- vetrina (hero)
/** Frame di ogni caso della vetrina. */
export const frameVetrina = (fps: number): number[] => CASI_VETRINA.map((caso) => Math.round((caso.durata ?? SECONDI_VETRINA) * fps));

export const ArgoVetrina: React.FC = () => {
  const { fps } = useVideoConfig();
  const durate = frameVetrina(fps);
  return (
    <AbsoluteFill style={{ background: NERO }}>
      {CASI_VETRINA.map((caso, indice) => (
        <Sequence key={caso.nome} from={durate.slice(0, indice).reduce((a, b) => a + b, 0)} durationInFrames={durate[indice]}>
          <ScenaCaso caso={caso} numero={indice + 1} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
