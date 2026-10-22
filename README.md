<p align="center"><img src="immagini/argo_banner.jpg" alt="Argo: un occhio stilizzato e la scritta ARGO accanto alla coda di pavone con gli occhi di Argo, dal dipinto di Rubens" width="100%"></p>

# Argo

**Argo guarda le telecamere CCTV e riconosce violenza, violenza con arma e scippi, con un WARNING ARMA quando vede un'arma in mano.**
Il nome viene dal gigante dai cento occhi del mito greco, che non dormiva mai con tutti gli occhi chiusi.

> **[TESTO DI MARCO: il perché del progetto, 3–4 righe]**

Argo si collega alle telecamere (RTSP, webcam, telefono, file) e mostra su un pannello web, sul tuo computer:
- gli **scheletri** delle persone, con gli ID del tracking;
- le **armi** viste;
- la **classe della scena** con le probabilità;
- il **WARNING ARMA** e l'**ALLARME VIOLENZA**;
- un registro degli eventi e l'archivio delle registrazioni.

Tutto quello che vede si salva anche in file leggibili da altri programmi (keypoint, eventi, video annotato).

| | |
|---|---|
| **Violenza sì/no**, pezzi da 30 s | **0,90** di accuratezza (IC 95% 0,84–0,94), 11% di falsi allarmi |
| Violenza sì/no, pezzi da 10 s / per finestra di 1,6 s | 0,82 (IC 0,77–0,89) / 0,78 |
| 4 classi (normale, violenza, violenza con arma, scippo) | macro-F1 0,40: la violenza con arma e lo scippo sono ancora deboli ([limiti](#limiti)) |

## Come funziona

```
telecamera ─ffmpeg 25 fps─► YOLO26s-pose + tracking ─► scheletri (17 punti per persona) ─┐
                         └► detector armi YOLO26s ─► pistola / coltello / fucile ───────┤
                                                                                         ▼
              finestre di 40 frame (1,6 s), una ogni 10 frame ─► ST-GCN sulle relazioni tra persone e armi ─┐
              gli stessi 40 frame ─► rete video R(2+1)D-18 fine-tunata sulla violenza ─────────────────────┤
                                                                                                             ▼
                      normale / violenza / violenza con arma / scippo ─► ALLARME VIOLENZA dopo 7 finestre violente di fila
                      arma vicino al polso in ≥ 5 frame su 40 (confidenza ≥ 0,7) ─► WARNING ARMA
```

1. **Scheletri.** YOLO26s-pose trova le persone e i loro 17 punti (testa, spalle, gomiti, polsi…). Il tracker (TrackTrack) dà a ogni persona un ID che resta nel tempo.
2. **Armi.** Un detector YOLO26s addestrato su immagini CCTV di armi (Roboflow, US Mock Attack, Sohas) cerca pistole, coltelli e fucili. Un'arma conta solo se è vicina al polso di qualcuno: così le scarpe nere non diventano pistole.
3. **Il movimento nel tempo.** Ogni 0,4 s gli ultimi 1,6 s di scheletri e armi diventano un grafo spazio-temporale. Una **ST-GCN** lo classifica nelle 4 classi.
4. **I pixel.** Una rete video **R(2+1)D-18**, fine-tunata sulla violenza, guarda gli stessi 1,6 s. La sua probabilità di violenza si media con quella degli scheletri. Insieme passano da 0,78 a 0,90 sui pezzi da 30 s.
5. **Allarmi.** L'ALLARME VIOLENZA scatta dopo 7 finestre violente di fila (2,8 s). È la regola scelta sul validation set e con cui sono misurati i numeri sopra.

Più dettagli nelle slide della consegna (*La journey del progetto* e *Il sistema in profondità*).

## Installazione

Serve **Python 3.12** con [uv](https://docs.astral.sh/uv/) e **ffmpeg**. Provato su Linux (Arch); le webcam come `/dev/video0` sono solo Linux.

```bash
git clone https://github.com/marcozambrella/argo.git
cd argo
uv sync                 # ambiente e dipendenze (torch CPU; con una GPU NVIDIA installa torch con CUDA)
uv run argo scarica-pesi  # ~96 MB in pesi/, dalla Release v1.0 (solo ricerca, vedi sotto)
```

Senza pesi Argo parte lo stesso in **modalità solo scheletri**: persone, keypoint e tracking, senza classi né allarmi.
Il modello delle pose si scarica da solo la prima volta.

## Avvio: il pannello con le telecamere

```bash
uv run argo pannello --telecamera ingresso=rtsp://utente:password@192.168.1.20:554/stream1
```

Apri **http://127.0.0.1:8765**. Le telecamere sono più di una ripetendo `--telecamera nome=sorgente`.

| Sorgente | Esempio |
|---|---|
| Telecamera IP (RTSP) | `--telecamera ingresso=rtsp://utente:password@192.168.1.20:554/stream1`. L'indirizzo esatto dipende dalla marca: cercalo nel manuale come "RTSP URL" |
| Webcam | `--telecamera webcam=/dev/video0` (Linux) |
| Telefono come telecamera | un'app "IP Webcam" sul telefono, poi `--telecamera telefono=http://192.168.1.30:8080/video` |
| Flusso http(s), HLS, MJPEG | `--telecamera strada=https://…/stream.m3u8` |
| Video registrato | `--telecamera prova=video.mp4 --come-telecamera` (letto a velocità reale, come una telecamera) |

Il pannello ha due viste:
- **Telecamere**: immagine annotata, classe e probabilità, spie di WARNING ARMA e ALLARME VIOLENZA, e un registro con i cambiamenti (allarmi, armi, cambi di classe).
- **Archivio**: ogni sessione si rivede con la striscia colorata delle classi nel tempo; cliccando un evento si salta a quel momento.

Le password delle telecamere non compaiono mai nel pannello. Il pannello ascolta solo su questo computer: `--host 0.0.0.0` lo apre alla rete locale, ma senza password, quindi va usato solo su una rete di cui ti fidi.

**Velocità.**
- **Con una GPU NVIDIA** Argo va in **tempo reale**: su una T4 l'estrazione di scheletri e armi viaggia a 83,6 frame/s.
- **Su un portatile senza GPU** analizza circa 2 frame/s. Le telecamere vanno avanti a 25: Argo registra tutto a segmenti di 10 s e analizza dietro, **in differita**. Il ritardo è scritto sul riquadro della telecamera, e nessun frame si perde.

Per analizzare un solo video, senza pannello:

```bash
uv run argo analizza-video --video video.mp4 --classificatore pesi/stgcn.pt \
  --testa-pixel pesi/testa_violenza.npz --rete-pixel pesi/rete_violenza.pt \
  --modello-armi pesi/armi_v3.pt --taratura pesi/taratura_warning.json --uscita uscita/
```

## Cosa esce dal computer

Ogni telecamera salva una sessione in `sessioni/<telecamera>/<data-ora>/`. `analizza-video` mette gli stessi file nella cartella `--uscita`.

| File | Contenuto |
|---|---|
| `video_annotato.mp4` | il video con scheletri, ID e armi disegnati |
| `eventi.jsonl` / `eventi.json` | una riga ogni 0,4 s: `inizio_secondi`, `fine_secondi`, `classe`, `probabilita` delle 4 classi, `warning_arma`, `allarme_violenza` |
| `keypoint.jsonl` | una riga per frame: per ogni persona `id`, `box` [x1, y1, x2, y2] e 17 `keypoint` [x, y, confidenza] in pixel (ordine COCO); per ogni arma `classe`, `confidenza`, `box` |
| `probabilita.csv` | gli eventi in tabella |
| `segmenti/` | la registrazione originale della telecamera, a pezzi da 10 s |

Esempio di riga di `eventi.jsonl`:
```json
{"inizio_secondi": 6.4, "fine_secondi": 8.0, "classe": "violenza", "warning_arma": false, "allarme_violenza": true,
 "probabilita": {"normale": 0.18, "violenza": 0.81, "violenza_con_arma": 0.0, "scippo": 0.003}}
```

Gli stessi dati sono anche su un'**API JSON locale**, per collegare Argo ad altri sistemi:
- `GET /api/stato`: telecamere, ritardo e ultima decisione;
- `GET /api/eventi?dopo=N`: gli eventi nuovi;
- `GET /flusso/<telecamera>.mjpg`: l'immagine annotata dal vivo.

## Risultati

Misure sul **test set**: 15% dei video, divisi per video (mai lo stesso video in addestramento e in test).

**Violenza sì/no, il sistema consegnato (scheletri + rete video)**

| Misura | Accuratezza | Recall | Falsi allarmi |
|---|---|---|---|
| per finestra (1,6 s) | 0,78 | 0,82 | 0,25 |
| pezzi da 10 s (272) | 0,82 (IC 95% 0,77–0,89) | 0,80 | 0,16 |
| **pezzi da 30 s (167)** | **0,90 (IC 95% 0,84–0,94)** | **0,90** | **0,11** |

Per sorgente, sui pezzi da 30 s:

| Sorgente | Accuratezza |
|---|---|
| RWF-2000 | 0,91 |
| CCTV-Fights | 0,88 |
| UCF-Crime | 0,82 |
| SCVD | 1,00 |

Solo scheletri: 0,78. Solo rete video: 0,84. Insieme: 0,90.

**4 classi** (ST-GCN, 32.995 finestre di test)

| | Normale | Violenza | Violenza con arma | Scippo |
|---|---|---|---|---|
| F1 | 0,77 | 0,58 | 0,10 | 0,13 |
| recall | 0,81 | 0,51 | 0,18 | 0,15 |

Macro-F1 0,40. A confronto, il Graph-Transformer arriva a 0,29.

**Detector delle armi (v3)**, mAP50:
- 0,77 sulle armi impugnate di Sohas;
- 0,20 sulla CCTV vera di US Mock Attack (pistole 0,32, fucili 0,28, coltelli ~0);
- 2 falsi allarmi su 326 foto di coltelli da cucina, telefoni e forbici.

Un detector più grande (YOLO26m a 1280 px) vede più pistole nella CCTV vera (0,41), ma non migliora i coltelli ed è 3–4 volte più lento: è rimasto un esperimento.

**Folle.** Nelle scene affollate mai viste (stazioni, campus, strade) l'ST-GCN dà la classe giusta nell'82–98% delle finestre.

## Limiti

Li diciamo chiaramente:
- **Violenza con arma e scippo sono deboli** (F1 0,10 e 0,13). Ci sono pochi esempi pubblici: circa 600 finestre di scippo contro decine di migliaia di normali.
- **Il WARNING ARMA scatta di rado**: precisione 0,35, recall 0,03. È tarato per non dare falsi allarmi, ma così perde molte armi vere.
- **I coltelli nella CCTV vera non si vedono quasi mai.** Sono piccoli, a bassa risoluzione e spesso coperti dalla mano.
- **UCF-Crime è la sorgente più difficile**: video lunghi, di bassa qualità e con molte persone.
- **Nelle folle molto dense** il tracking a volte perde le persone.
- **Senza GPU** il pannello lavora in differita, non in tempo reale.
- Argo è un **aiuto per chi guarda le telecamere**, non un sistema che decide da solo. Ogni allarme va controllato da una persona.

## Addestrare da zero

Il calcolo pesante è stato fatto su **Kaggle** (2× T4). I notebook sono in [`kaggle/`](kaggle) e si lanciano con `kaggle/lancia.sh`. L'ordine è:
1. detector delle armi: `detector_armi`;
2. estrazione di scheletri e armi da tutti i video: `estrazione`, `aggiungi_armi`;
3. finestre: `finestre`;
4. addestramento delle 4 classi: `addestramento`;
5. fine-tuning della rete video: `finetune_rgb`.

Il codice è in [`src/sicurezza/`](src/sicurezza): `dati.py`, `visione.py`, `modelli.py`, `demo.py` (con il pannello) e `strumenti.py`. I test si lanciano con `uv run pytest`.

## Dati, pesi e licenze

- **Codice**: [AGPL-3.0](LICENSE), come Ultralytics YOLO su cui si basa.
- **Pesi** (Release v1.0): **solo ricerca e uso non commerciale**. Sono addestrati anche su dataset distribuiti solo per uso accademico.
- **Dataset di addestramento** (non inclusi nel repo):
  - [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection): uso accademico, gli autori chiedono di non ridistribuirlo;
  - CCTV-Fights (copia Kaggle [`mohamedtariki/vsd-cctv-fights`](https://www.kaggle.com/datasets/mohamedtariki/vsd-cctv-fights));
  - [SCVD](https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd);
  - [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) con le annotazioni [UCA](https://github.com/Xuange923/Surveillance-Video-Understanding) (Apache 2.0, in `annotations/uca/`);
  - [Chain-snatching](https://www.kaggle.com/datasets/aniketsahu09/chain-snatching-cctv-dataset);
  - video di scippi con licenza CC BY: attribuzioni in [`annotations/phone_snatching/attribuzioni.csv`](annotations/phone_snatching/attribuzioni.csv);
  - per le scene di folla: MOT17/MOT20, HT21, ShanghaiTech, UBI-Fights;
  - per le armi: Roboflow ([Master Weapon CCTV](https://universe.roboflow.com/beoeoa/master_weapon_cctv), [CCTV Weapon Detect](https://universe.roboflow.com/prospace-gi6e4/cctv-weapon-detect-4vuah)), US Mock Attack, Sohas.
- **Video della presentazione e del sito**:
  - clip CC BY da YouTube e video Pexels, con le attribuzioni in [`annotations/video_pubblici/attribuzioni.csv`](annotations/video_pubblici/attribuzioni.csv);
  - i video dei dataset compaiono solo come scheletri su fondo nero.
- **Immagine**: Peter Paul Rubens, *Giunone e Argo* (1610 circa), pubblico dominio ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Peter_Paul_Rubens_-_Juno_and_Argus_-_WGA20280.jpg)).

## Crediti

> **[TESTO DI MARCO: squadra, hackathon, ringraziamenti]**

Sviluppato con l'aiuto di Claude (Anthropic) come assistente di programmazione. Le scelte sono di Marco e sono annotate passo per passo.
