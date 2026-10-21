"""Prepara i dati della mappa di Roma: dati/roma.json (numeri ufficiali + stima per municipio) e
dati/mappa-roma.geojson (municipi, metro, stazioni, vie delle zone rosse da OpenStreetMap).

Scelte di Marco (18/10/2026):
- categorie: furti alle persone (scippi + borseggi), aggressioni (percosse e lesioni), rapine nei negozi;
- la distribuzione per municipio NON è un dato: nessuna fonte pubblica dà i reati per municipio. È una STIMA
  dichiarata: totale cittadino × quota di popolazione, con +50% per i municipi con zone rosse (I e VII);
  i cubetti in più si posano sulle vie delle zone rosse.

Uso (dalla cartella site/dati/):
    python3 strumenti/prepara_mappa.py municipi_geom.json osm_strati.json fermate_metro.json
dove i JSON sono le risposte di Overpass (il terzo, facoltativo, sono le fermate della metro) (query in fondo al file).
"""
import json
import sys

# --- numeri ufficiali (mai modificare senza fonte) ---------------------------------------------------------
FONTI = {
    "annuario_sicurezza": {
        "nome": "Roma Capitale, Annuario statistico 2025, cap. 15 Sicurezza urbana (elaborazioni su dati Istat)",
        "url": "https://www.comune.roma.it/web-resources/cms/documents/15_SicurezzaUrbana_Annuario_2025.pdf"},
    "prefettura": {
        "nome": "Ministero dell'Interno / Prefettura di Roma, delitti commessi a Roma Capitale 2021-2024",
        "url": "https://prefettura.interno.gov.it/sites/default/files/57/2025-09/delitti-commessi-roma-capitale-2021.2022.2023-e-2024-con-perc.pdf"},
    "annuario_popolazione": {
        "nome": "Roma Capitale, Annuario statistico 2025, cap. 2 Popolazione, tab. 2.7 (iscritti in anagrafe al 31/12/2024)",
        "url": "https://www.comune.roma.it/web-resources/cms/documents/02_Popolazione_Annuario_2025.pdf"},
    "zone_rosse": {
        "nome": "Prefettura di Roma, ordinanza sulle zone a vigilanza rafforzata dell'8 gennaio 2025 (riportata da RomaToday)",
        "url": "https://www.romatoday.it/cronaca/zone-rosse-roma-quali-strade.html"},
    "osm": {"nome": "© OpenStreetMap contributors, ODbL 1.0", "url": "https://www.openstreetmap.org/copyright"},
}

CATEGORIE = [
    {"chiave": "furti_persone", "nome": "Furti alle persone", "totale": 30524 + 1621, "anno": 2023,
     "fonte": "annuario_sicurezza", "unita_cubetto": 250,
     "dettaglio": "furti con destrezza 30.524 + furti con strappo 1.621 (tab. 15.3)",
     "nota": "Non solo in strada: anche su bus, metro e nei locali."},
    {"chiave": "aggressioni", "nome": "Aggressioni", "totale": 3308, "anno": 2023,
     "fonte": "annuario_sicurezza", "unita_cubetto": 25,
     "dettaglio": "percosse e lesioni (tab. 15.2)", "nota": ""},
    {"chiave": "rapine_negozi", "nome": "Rapine nei negozi", "totale": 381, "anno": 2023,
     "fonte": "prefettura", "unita_cubetto": 5,
     "dettaglio": "rapine in esercizi commerciali (Viminale, delitti commessi)",
     "nota": "Fonte diversa dalle altre due categorie (delitti commessi, non denunciati)."},
]

POPOLAZIONE_31_12_2024 = {  # tab. 2.7, colonna Totale ("Non localizzati": 568, esclusi)
    "I": 161647, "II": 163097, "III": 202442, "IV": 168039, "V": 239925, "VI": 242069, "VII": 311895,
    "VIII": 127086, "IX": 183043, "X": 226842, "XI": 151057, "XII": 138694, "XIII": 130333, "XIV": 189909,
    "XV": 160620,
}
MUNICIPI_CON_ZONE_ROSSE = {"I": "Termini ed Esquilino", "VII": "Tuscolano"}
PESO_EXTRA_ZONE_ROSSE = 0.5   # scelta di Marco: +50%

VIE_ZONE_ROSSE = {
    "I": ["Piazza dei Cinquecento", "Via Giovanni Giolitti", "Via Giovanni Amendola", "Via Filippo Turati",
          "Via Principe Amedeo", "Via Daniele Manin", "Piazza Manfredo Fanti", "Via Carlo Cattaneo",
          "Via Enrico Cialdini", "Via Urbano Rattazzi", "Piazza Vittorio Emanuele II"],
    "VII": ["Via Monselice", "Via Mestre", "Piazza Ragusa"],
}
STAZIONI = {"Roma Termini": (12.50253, 41.90058), "Roma Tiburtina": (12.53156, 41.91119),
            "Roma Ostiense": (12.48655, 41.87200), "Roma Trastevere": (12.46609, 41.87255)}
LINEE_METRO = {"A": 1720955, "B": 207926, "B1": 2172805, "C": 398053}   # relazioni OSM (la D non esiste ancora)
# stessa fermata con nomi diversi su linee diverse (OSM)
NOMI_UNIFICATI = {"Colosseo – Fori Imperiali": "Colosseo"}
ROMANI = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11,
          "XII": 12, "XIII": 13, "XIV": 14, "XV": 15}


def unisci_anelli(tratti):
    tratti = [list(tratto) for tratto in tratti]
    anelli = []
    while tratti:
        anello = tratti.pop(0)
        cambiato = True
        while anello[0] != anello[-1] and cambiato:
            cambiato = False
            for indice, tratto in enumerate(tratti):
                if tratto[0] == anello[-1]: anello += tratto[1:]
                elif tratto[-1] == anello[-1]: anello += tratto[::-1][1:]
                elif tratto[-1] == anello[0]: anello = tratto[:-1] + anello
                elif tratto[0] == anello[0]: anello = tratto[::-1][:-1] + anello
                else: continue
                tratti.pop(indice)
                cambiato = True
                break
        if anello[0] == anello[-1] and len(anello) > 3:
            anelli.append(anello)
    return anelli


def semplifica(punti, tolleranza):
    if len(punti) < 3:
        return punti
    (x1, y1), (x2, y2) = punti[0], punti[-1]
    massima, indice_massimo = 0, 0
    for indice in range(1, len(punti) - 1):
        x0, y0 = punti[indice]
        denominatore = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 or 1e-12
        distanza = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / denominatore
        if distanza > massima:
            massima, indice_massimo = distanza, indice
    if massima <= tolleranza:
        return [punti[0], punti[-1]]
    return semplifica(punti[:indice_massimo + 1], tolleranza)[:-1] + semplifica(punti[indice_massimo:], tolleranza)


def semplifica_anello(anello, tolleranza=0.0005):
    meta = len(anello) // 2
    return semplifica(anello[:meta + 1], tolleranza)[:-1] + semplifica(anello[meta:], tolleranza)


def punti(geometria):
    return [(round(punto["lon"], 5), round(punto["lat"], 5)) for punto in geometria]


def stima_per_municipio():
    pesi = {m: pop * (1 + (PESO_EXTRA_ZONE_ROSSE if m in MUNICIPI_CON_ZONE_ROSSE else 0))
            for m, pop in POPOLAZIONE_31_12_2024.items()}
    somma_pesi = sum(pesi.values())
    stime = {}
    for municipio, popolazione in POPOLAZIONE_31_12_2024.items():
        stime[municipio] = {}
        for categoria in CATEGORIE:
            base = categoria["totale"] * popolazione / somma_pesi
            extra = categoria["totale"] * (pesi[municipio] - popolazione) / somma_pesi
            stime[municipio][categoria["chiave"]] = {"base": round(base, 1), "zone_rosse": round(extra, 1)}
    return stime, somma_pesi


if __name__ == "__main__":
    municipi_osm = json.load(open(sys.argv[1]))
    strati_osm = json.load(open(sys.argv[2]))
    elementi = []
    for relazione in municipi_osm["elements"]:
        tratti = [punti(m["geometry"]) for m in relazione["members"] if m["type"] == "way" and m.get("role") == "outer"]
        sigla = relazione["tags"]["name"].split()[-1]
        anelli = [semplifica_anello(anello) for anello in unisci_anelli(tratti)]
        elementi.append({"type": "Feature", "properties": {"tipo": "municipio", "sigla": sigla, "numero": ROMANI[sigla]},
                         "geometry": {"type": "MultiPolygon", "coordinates": [[list(map(list, a))] for a in anelli]}})
    relazioni = {e["id"]: e for e in strati_osm["elements"] if e["type"] == "relation"}
    for linea, identificativo in LINEE_METRO.items():
        tratti = [punti(m["geometry"]) for m in relazioni[identificativo]["members"]
                  if m["type"] == "way" and m.get("geometry") and m.get("role", "") in ("", "forward", "backward")]
        elementi.append({"type": "Feature", "properties": {"tipo": "metro", "linea": linea},
                         "geometry": {"type": "MultiLineString", "coordinates": [list(map(list, semplifica(t, 0.0002))) for t in tratti]}})
    if len(sys.argv) > 3:   # fermate della metro: una per nome, con tutte le linee che ci passano
        fermate_osm = json.load(open(sys.argv[3]))
        nodi = {e["id"]: e for e in fermate_osm["elements"] if e["type"] == "node"}
        fermate = {}
        for linea, identificativo in LINEE_METRO.items():
            relazione = next(e for e in fermate_osm["elements"] if e["type"] == "relation" and e["id"] == identificativo)
            for membro in relazione["members"]:
                if not membro["role"].startswith("stop") or membro["ref"] not in nodi: continue
                nodo = nodi[membro["ref"]]
                nome = NOMI_UNIFICATI.get(nodo["tags"]["name"], nodo["tags"]["name"])
                fermata = fermate.setdefault(nome, {"linee": [], "punti": []})
                sigla = "B" if linea == "B1" else linea   # sui cartelli la B1 è un ramo della B
                if sigla not in fermata["linee"]: fermata["linee"].append(sigla)
                fermata["punti"].append((nodo["lon"], nodo["lat"]))
        for nome, fermata in fermate.items():
            lon = sum(p[0] for p in fermata["punti"]) / len(fermata["punti"])
            lat = sum(p[1] for p in fermata["punti"]) / len(fermata["punti"])
            elementi.append({"type": "Feature", "properties": {"tipo": "fermata", "nome": nome, "linee": sorted(fermata["linee"])},
                             "geometry": {"type": "Point", "coordinates": [round(lon, 5), round(lat, 5)]}})
    for nome, (lon, lat) in STAZIONI.items():
        elementi.append({"type": "Feature", "properties": {"tipo": "stazione", "nome": nome},
                         "geometry": {"type": "Point", "coordinates": [lon, lat]}})
    for municipio, vie in VIE_ZONE_ROSSE.items():
        for via in strati_osm["elements"]:
            if via["type"] == "way" and via["tags"].get("name") in vie:
                elementi.append({"type": "Feature", "properties": {"tipo": "zona_rossa", "municipio": municipio, "nome": via["tags"]["name"]},
                                 "geometry": {"type": "LineString", "coordinates": list(map(list, punti(via["geometry"])))}})
    json.dump({"type": "FeatureCollection", "_fonte": FONTI["osm"]["nome"], "features": elementi},
              open("mappa-roma.geojson", "w"), separators=(",", ":"))

    stime, somma_pesi = stima_per_municipio()
    totale_tre = sum(c["totale"] for c in CATEGORIE)
    json.dump({
        "_nota": "Numeri ufficiali con fonte; la ripartizione per municipio è una STIMA dichiarata (vedi metodo_stima).",
        "fonti": FONTI,
        "categorie": [{**c, "quota_sui_tre": round(c["totale"] / totale_tre, 4)} for c in CATEGORIE],
        "popolazione_municipi": POPOLAZIONE_31_12_2024,
        "zone_rosse": MUNICIPI_CON_ZONE_ROSSE,
        "metodo_stima": {
            "descrizione": "Nessuna fonte pubblica dà i reati per municipio. Stima: totale cittadino × peso del municipio / somma dei pesi; "
                           "peso = popolazione residente, +50% nei municipi con zone rosse (I e VII). Scelta del team, non un dato.",
            "peso_extra_zone_rosse": PESO_EXTRA_ZONE_ROSSE, "somma_pesi": somma_pesi},
        "stime_per_municipio": stime,
        "altri_dati": {"totale_delitti_2023": {"valore": 205918, "fonte": "annuario_sicurezza"},
                       "voci_viminale_con_armi": {"valore": 0, "nota": "Nessuna delle 34 voci del Viminale distingue i reati commessi con armi.", "fonte": "prefettura"}},
    }, open("roma.json", "w"), ensure_ascii=False, indent=1)
    print(len(elementi), "elementi;", "stima VII furti:", stime["VII"]["furti_persone"])

# Query Overpass usate (18/10/2026):
#   municipi:  area["name"="Roma"]["admin_level"="8"]["boundary"="administrative"]->.roma;
#              rel(area.roma)["boundary"="administrative"]["admin_level"="10"];out geom;
#   strati:    rel(area.roma)["route"="subway"];out geom;
#   fermate:   rel(id:1720955,207926,2172805,398053);out body;node(r:"stop");out body;  way(area.roma)["highway"]["name"~"^(…vie…)$"];out geom;
