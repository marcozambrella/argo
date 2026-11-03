# Argo v1.1 (D70) — versione passata

Copia ferma di `src/sicurezza` alla Release v1.1 (commit e2b4a7a), tenuta per rifare il vecchio flusso e
confrontarlo con quello nuovo (D71, richiesta di Marco: "tienili in una cartella dedicata delle versioni passate").

Il flusso v1.1, per ogni frame a 25 fps:
- pose YOLO26s a 640 px + tracker, detector delle armi v3 (riquadri disegnati e nodi arma degli scheletri);
- ST-GCN sulle finestre di 40 frame, violenza fusa con la rete violenza R(2+1)D (D52/D53, ogni 0,4 s);
- b1 (112 px) decide lo scippo ogni 0,8 s, b2 (224 px) dà il WARNING ARMA ogni 4 s.

~14,9 s di calcolo per secondo di video su un portatile senza GPU (misura del 03/11, `runs/notte_0311`).

Si usa con i pesi della v1.1 (non più pubblici dal 04/11: Release cancellata, copia in
dati1tb/argo/pesi_release/v1.1), mettendo questa cartella davanti al codice di oggi:

```bash
PYTHONPATH=versioni_passate/v1.1/src uv run python -m sicurezza.demo analizza-video --video X.mp4 \
  --classificatore pesi_v1.1/stgcn.pt --testa-pixel pesi_v1.1/testa_violenza.npz \
  --rete-pixel pesi_v1.1/rete_violenza.pt --modello-armi pesi_v1.1/armi_v3.pt \
  --taratura pesi_v1.1/taratura_warning.json --rete-scippi pesi_v1.1/rete_scippi_b1.pt \
  --rete-armi pesi_v1.1/rete_armi_b2.pt --uscita runs/demo_v1.1/X
```
