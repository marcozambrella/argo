"""Confini dei 15 municipi di Roma da OpenStreetMap (© OpenStreetMap contributors, licenza ODbL) → GeoJSON semplificato.

Uso:
    curl -A "roma-safe-cctv" https://overpass-api.de/api/interpreter --data-urlencode 'data=[out:json][timeout:120];
      area["name"="Roma"]["admin_level"="8"]["boundary"="administrative"]->.roma;
      rel(area.roma)["boundary"="administrative"]["admin_level"="10"];out geom;' -o municipi_geom.json
    python3 municipi_da_osm.py municipi_geom.json ../municipi-roma.geojson
"""
import json
import sys


def unisci_anelli(tratti):
    """Unisce i tratti (liste di punti) in anelli chiusi facendo combaciare gli estremi."""
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
    """Douglas–Peucker."""
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


ROMANI = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11,
          "XII": 12, "XIII": 13, "XIV": 14, "XV": 15}

if __name__ == "__main__":
    dati = json.load(open(sys.argv[1]))
    elementi = []
    for relazione in dati["elements"]:
        tratti = [[(round(punto["lon"], 5), round(punto["lat"], 5)) for punto in membro["geometry"]]
                  for membro in relazione["members"] if membro["type"] == "way" and membro.get("role") == "outer"]
        # un anello chiuso si semplifica in due metà (Douglas–Peucker vuole estremi diversi)
        anelli = []
        for anello in unisci_anelli(tratti):
            meta = len(anello) // 2
            anelli.append(semplifica(anello[:meta + 1], 0.0006)[:-1] + semplifica(anello[meta:], 0.0006))
        sigla = relazione["tags"]["name"].split()[-1]
        elementi.append({"type": "Feature",
                         "properties": {"nome": relazione["tags"]["name"], "numero": ROMANI[sigla], "sigla": sigla,
                                        "osm_id": relazione["id"]},
                         "geometry": {"type": "MultiPolygon", "coordinates": [[list(map(list, anello))] for anello in anelli]}})
    elementi.sort(key=lambda elemento: elemento["properties"]["numero"])
    json.dump({"type": "FeatureCollection",
               "_fonte": "© OpenStreetMap contributors, ODbL 1.0 — https://www.openstreetmap.org/copyright",
               "features": elementi}, open(sys.argv[2], "w"), separators=(",", ":"))
    for elemento in elementi:
        print(elemento["properties"]["sigla"], [len(anello[0]) for anello in elemento["geometry"]["coordinates"]])
