// Eventi veri di Argo (eventi.json di `argo analizza-video`): una decisione ogni 0,4 s su finestre di 1,6 s.
import { useEffect, useState } from "react";
import { cancelRender, continueRender, delayRender, staticFile } from "remotion";

export type Evento = {
  readonly inizio_secondi: number;
  readonly fine_secondi: number;
  readonly classe: "normale" | "violenza" | "violenza_con_arma" | "scippo" | null;
  readonly warning_arma: boolean;
  readonly allarme_violenza: boolean;
  readonly probabilita: Record<string, number> | null;
};

const caricati = new Map<string, Promise<Evento[]>>();

export const useEventi = (nome: string): Evento[] | null => {
  const [eventi, impostaEventi] = useState<Evento[] | null>(null);
  const [attesa] = useState(() => delayRender(`Eventi ${nome}`));
  useEffect(() => {
    if (!caricati.has(nome)) {
      caricati.set(nome, fetch(staticFile(`argo/${nome}.json`)).then((risposta) => risposta.json()));
    }
    caricati.get(nome)!
      .then((dati) => { impostaEventi(dati); continueRender(attesa); })
      .catch((errore) => cancelRender(errore));
  }, [nome, attesa]);
  return eventi;
};

/** La decisione in vigore al secondo `secondi` della clip: l'ultima finestra già conclusa. */
export const decisioneAl = (eventi: Evento[], secondi: number): Evento | null => {
  let corrente: Evento | null = null;
  for (const evento of eventi) {
    if (evento.fine_secondi <= secondi + 1e-6) corrente = evento;
    else break;
  }
  return corrente;
};
