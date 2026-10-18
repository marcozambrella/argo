// Caricamento delle rilevazioni vere (JSON prodotti da strumenti/estrai_rilevazioni.py).
import { useEffect, useState } from "react";
import { cancelRender, continueRender, delayRender, staticFile } from "remotion";

export type Persona = {
  readonly id: number;
  readonly box: readonly [number, number, number, number];   // x1, y1, x2, y2 normalizzati 0–1
  readonly confidenza: number;
  readonly keypoint: ReadonlyArray<readonly [number, number, number]>;  // 17 × (x, y, visibilità)
};

export type Arma = {
  readonly classe: string;
  readonly box: readonly [number, number, number, number];
  readonly confidenza: number;
};

export type Rilevazioni = {
  readonly video: string;
  readonly fps: number;
  readonly larghezza: number;
  readonly altezza: number;
  readonly modello_pose: string;
  readonly modello_armi: string | null;
  readonly frame: ReadonlyArray<{ readonly persone: Persona[]; readonly armi: Arma[] }>;
};

const giaCaricate = new Map<string, Promise<Rilevazioni>>();

export const useRilevazioni = (nomeFile: string): Rilevazioni | null => {
  const [rilevazioni, impostaRilevazioni] = useState<Rilevazioni | null>(null);
  const [attesa] = useState(() => delayRender(`Rilevazioni ${nomeFile}`));

  useEffect(() => {
    if (!giaCaricate.has(nomeFile)) {
      giaCaricate.set(nomeFile, fetch(staticFile(`rilevazioni/${nomeFile}`)).then((risposta) => risposta.json()));
    }
    giaCaricate.get(nomeFile)!
      .then((dati) => {
        impostaRilevazioni(dati);
        continueRender(attesa);
      })
      .catch((errore) => cancelRender(errore));
  }, [nomeFile, attesa]);

  return rilevazioni;
};

// Da quanti frame consecutivi (fino a `frame`) è visibile la traccia `id`: per mostrare la durata della traccia
export const frameDiTraccia = (rilevazioni: Rilevazioni, frame: number, id: number): number => {
  let conteggio = 0;
  for (let indice = frame; indice >= 0; indice--) {
    if (!rilevazioni.frame[indice]?.persone.some((persona) => persona.id === id)) break;
    conteggio++;
  }
  return conteggio;
};
