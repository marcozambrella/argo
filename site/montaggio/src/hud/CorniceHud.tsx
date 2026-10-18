// Cornice fissa dell'interfaccia: camera e REC, contatore dei frame, barra della finestra di 40 frame (passo 10),
// stato del classificatore, credito della clip, scanline e vignettatura. Solo dati veri o etichette tecniche.
import type React from "react";
import { interpolate, random, useCurrentFrame, useVideoConfig } from "remotion";
import { FRAME_PER_FINESTRA, PASSO_FINESTRE, type Tema } from "../tema";

export const Scanline: React.FC<{ readonly intensita?: number }> = ({ intensita = 0.12 }) => {
  const frame = useCurrentFrame();
  return (
    <>
      <div style={{
        position: "absolute", inset: 0, pointerEvents: "none", opacity: intensita,
        backgroundImage: "repeating-linear-gradient(0deg, rgba(0,0,0,0.9) 0px, rgba(0,0,0,0.9) 1px, transparent 1px, transparent 3px)",
      }} />
      <div style={{
        position: "absolute", left: 0, right: 0, height: 120, pointerEvents: "none", opacity: 0.06,
        top: `${(frame * 9) % 840 - 120}px`,
        background: "linear-gradient(180deg, transparent, white, transparent)",
      }} />
      <div style={{
        position: "absolute", inset: 0, pointerEvents: "none",
        background: "radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,0.55) 100%)",
      }} />
    </>
  );
};

// Ingresso "glitch" nei primi frame di una scena: spostamento orizzontale casuale ma deterministico
export const spostamentoGlitch = (frame: number, seme: string): string => {
  if (frame > 8) return "0px 0px";
  const ampiezza = interpolate(frame, [0, 8], [26, 0]);
  return `${(random(`${seme}-${frame}`) - 0.5) * ampiezza}px 0px`;
};

export const CorniceHud: React.FC<{
  readonly tema: Tema;
  readonly numeroCamera: string;
  readonly frameClip: number;
  readonly creditoClip: string;
  readonly statoClassificatore: string;
  readonly warningArma: boolean;
}> = ({ tema, numeroCamera, frameClip, creditoClip, statoClassificatore, warningArma }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const recAcceso = Math.floor(frame / (fps / 2)) % 2 === 0;
  const inizioFinestra = Math.max(0, Math.floor((frameClip - FRAME_PER_FINESTRA + 1) / PASSO_FINESTRE) * PASSO_FINESTRE);
  const testoBase: React.CSSProperties = { fontFamily: tema.fontMonospazio, color: tema.coloreTesto, fontSize: 15, letterSpacing: 1 };

  return (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
      {/* angoli dell'inquadratura */}
      {[[24, 24, "0deg"], [1256, 24, "90deg"], [1256, 696, "180deg"], [24, 696, "270deg"]].map(([x, y, rotazione]) => (
        <div key={`${x}-${y}`} style={{
          position: "absolute", left: Number(x) - 1, top: Number(y) - 1, width: 34, height: 34, rotate: String(rotazione),
          transformOrigin: "1px 1px", borderLeft: `2px solid ${tema.coloreLinee}`, borderTop: `2px solid ${tema.coloreLinee}`,
          opacity: 0.8,
        }} />
      ))}

      <div style={{ ...testoBase, position: "absolute", left: 48, top: 40, display: "flex", gap: 14, alignItems: "center" }}>
        <span style={{ width: 11, height: 11, borderRadius: 6, background: tema.coloreAllarme, opacity: recAcceso ? 1 : 0.15 }} />
        <span>REC</span>
        <span style={{ opacity: 0.6 }}>|</span>
        <span>{numeroCamera}</span>
      </div>

      <div style={{ ...testoBase, position: "absolute", right: 48, top: 40, textAlign: "right" }}>
        {`25 FPS · FRAME ${String(frameClip).padStart(5, "0")} · T+${(frameClip / fps).toFixed(2)} s`}
      </div>

      {/* barra della finestra: gli ultimi 40 frame che il classificatore guarda, una nuova ogni 10 */}
      <div style={{ position: "absolute", left: 48, right: 48, bottom: 40 }}>
        <div style={{ ...testoBase, fontSize: 13, display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
          <span>{`FINESTRA ${inizioFinestra}–${inizioFinestra + FRAME_PER_FINESTRA - 1} · 40 FRAME = 1,6 s · PASSO 10`}</span>
          <span style={{ color: tema.coloreAccento }}>{statoClassificatore}</span>
        </div>
        <div style={{ display: "flex", gap: 3 }}>
          {Array.from({ length: FRAME_PER_FINESTRA }, (_, indice) => {
            const frameDellaCella = frameClip - (FRAME_PER_FINESTRA - 1) + indice;
            return (
              <div key={indice} style={{
                flex: 1, height: indice % PASSO_FINESTRE === 0 ? 14 : 9, alignSelf: "flex-end",
                background: frameDellaCella >= 0 ? tema.coloreAccento : tema.coloreLinee,
                opacity: frameDellaCella >= 0 ? 0.35 + 0.65 * (indice / FRAME_PER_FINESTRA) : 0.12,
              }} />
            );
          })}
        </div>
      </div>

      {warningArma ? (
        <div style={{
          position: "absolute", left: "50%", top: 92, translate: "-50% 0", padding: "8px 22px",
          border: `2px solid ${tema.coloreAllarme}`, background: "rgba(40,0,0,0.6)", color: tema.coloreAllarme,
          fontFamily: tema.fontTitoli, fontWeight: 700, fontSize: 30, letterSpacing: 6, opacity: recAcceso ? 1 : 0.65,
        }}>
          WARNING ARMA
        </div>
      ) : null}

      <div style={{ ...testoBase, position: "absolute", right: 48, bottom: 88, fontSize: 11, opacity: 0.55 }}>
        {creditoClip}
      </div>
    </div>
  );
};
