<p align="center"><img src="immagini/argo_banner.jpg" alt="ARGO: a stylised eye and the word ARGO next to a peacock tail full of eyes, from a painting by Rubens" width="100%"></p>

# ARGO

**ARGO watches CCTV cameras and recognises violence, armed violence and snatching, with a WEAPON WARNING when it sees a weapon in someone's hand.**
The name comes from Argus, the hundred-eyed giant of Greek myth who never slept with all his eyes closed.

ARGO connects to cameras (RTSP, webcam, phone, video file) and shows on a web panel, on your own computer:
- the **skeletons** of the people, with their tracking IDs;
- the **weapons** it sees;
- the **class of the scene**, with probabilities;
- the **WEAPON WARNING** and the **VIOLENCE ALARM**;
- an event log and an archive of the recordings.

Everything it sees is also saved to files that other programs can read: keypoints, events, annotated video.

| | |
|---|---|
| **Violence yes/no**, 30 s segments | **0.90** accuracy (95% CI 0.84–0.94), 11% false alarms |
| Violence yes/no, 10 s segments / per 1.6 s window | 0.82 (CI 0.77–0.89) / 0.78 |
| 4 classes (normal, violence, armed violence, snatching) | macro-F1 0.40: armed violence and snatching are still weak ([limits](#limits)) |

## How it works

```
camera ─ffmpeg 25 fps─► YOLO26s-pose + tracking ─► skeletons (17 points per person) ──┐
                     └► YOLO26s weapon detector ─► pistol / knife / rifle ────────────┤
                                                                                       ▼
          40-frame windows (1.6 s), one every 10 frames ─► ST-GCN on people and weapons ─┐
          the same 40 frames ─► R(2+1)D-18 video network fine-tuned on violence ─────────┤
                                                                                          ▼
              normal / violence / armed violence / snatching ─► VIOLENCE ALARM after 7 violent windows in a row
              weapon near a wrist in ≥ 5 of 40 frames (confidence ≥ 0.7) ─► WEAPON WARNING
```

1. **Skeletons.** YOLO26s-pose finds people and their 17 keypoints: head, shoulders, elbows, wrists and so on. The tracker (TrackTrack) gives each person an ID that lasts over time.
2. **Weapons.** A YOLO26s detector trained on CCTV weapon images (Roboflow, US Mock Attack, Sohas) looks for pistols, knives and rifles. A weapon only counts when it is near someone's wrist, so black shoes do not turn into pistols.
3. **Movement over time.** Every 0.4 s the last 1.6 s of skeletons and weapons become a spatio-temporal graph. An **ST-GCN** sorts it into the 4 classes.
4. **Pixels.** An **R(2+1)D-18** video network, fine-tuned on violence, looks at the same 1.6 s. Its violence probability is averaged with the skeletons' one. Together they go from 0.78 to 0.90 on 30 s segments.
5. **Alarms.** The VIOLENCE ALARM fires after 7 violent windows in a row (2.8 s). That rule was chosen on the validation set, and it is the one the numbers above are measured with.

More details in the hackathon slides (*The project journey* and *The system in depth*).

## Installation

You need **Python 3.12** with [uv](https://docs.astral.sh/uv/) and **ffmpeg**. Tested on Linux (Arch). Webcams as `/dev/video0` are Linux only.

```bash
git clone https://github.com/marcozambrella/argo.git
cd argo
uv sync                   # environment and dependencies (CPU torch; with an NVIDIA GPU install torch with CUDA)
uv run argo scarica-pesi  # ~96 MB into pesi/, from Release v1.0 (research only, see below)
```

Without the weights ARGO still starts, in **skeletons-only mode**: people, keypoints and tracking, with no classes or alarms. The pose model downloads itself the first time.

## Start: the camera panel

```bash
uv run argo pannello --telecamera entrance=rtsp://user:password@192.168.1.20:554/stream1
```

Open **http://127.0.0.1:8765**. For more cameras, repeat `--telecamera name=source`.

| Source | Example |
|---|---|
| IP camera (RTSP) | `--telecamera entrance=rtsp://user:password@192.168.1.20:554/stream1`. The exact address depends on the brand: look for "RTSP URL" in the manual |
| Webcam | `--telecamera webcam=/dev/video0` (Linux) |
| Phone as a camera | an "IP Webcam" app on the phone, then `--telecamera phone=http://192.168.1.30:8080/video` |
| http(s), HLS, MJPEG stream | `--telecamera street=https://…/stream.m3u8` |
| Recorded video | `--telecamera test=video.mp4 --come-telecamera` (read at real speed, like a camera) |

The panel has two views:
- **Cameras**: annotated image, class and probabilities, WEAPON WARNING and VIOLENCE ALARM lights, and a log of the changes (alarms, weapons, class changes).
- **Archive**: every session can be replayed with a coloured strip of the classes over time; clicking an event jumps to that moment.

Camera passwords never appear in the panel. The panel only listens on this computer: `--host 0.0.0.0` opens it to the local network, but without a password, so only use it on a network you trust.

**Speed.**
- **With an NVIDIA GPU** ARGO runs in **real time**: on a T4 the skeleton and weapon extraction runs at 83.6 frames/s.
- **On a laptop without a GPU** it analyses about 2 frames/s. Cameras run at 25, so ARGO records everything in 10 s segments and analyses behind them, **delayed**. The delay is shown on the camera tile, and no frame is lost.

To analyse a single video, without the panel:

```bash
uv run argo analizza-video --video video.mp4 --classificatore pesi/stgcn.pt \
  --testa-pixel pesi/testa_violenza.npz --rete-pixel pesi/rete_violenza.pt \
  --modello-armi pesi/armi_v3.pt --taratura pesi/taratura_warning.json --uscita output/
```

## What comes out on your computer

Each camera saves a session in `sessioni/<camera>/<date-time>/`. `analizza-video` writes the same files to its `--uscita` folder. (Code, folders and field names are in Italian.)

| File | Content |
|---|---|
| `video_annotato.mp4` | the video with skeletons, IDs and weapons drawn on it |
| `eventi.jsonl` / `eventi.json` | one line every 0.4 s: `inizio_secondi` (start), `fine_secondi` (end), `classe` (class), `probabilita` of the 4 classes, `warning_arma` (weapon warning), `allarme_violenza` (violence alarm) |
| `keypoint.jsonl` | one line per frame: for each person `id`, `box` [x1, y1, x2, y2] and 17 `keypoint` [x, y, confidence] in pixels (COCO order); for each weapon `classe`, `confidenza`, `box` |
| `probabilita.csv` | the events as a table |
| `segmenti/` | the original camera recording, in 10 s pieces |

Example line of `eventi.jsonl`:
```json
{"inizio_secondi": 6.4, "fine_secondi": 8.0, "classe": "violenza", "warning_arma": false, "allarme_violenza": true,
 "probabilita": {"normale": 0.18, "violenza": 0.81, "violenza_con_arma": 0.0, "scippo": 0.003}}
```

The same data is also on a **local JSON API**, to connect ARGO to other systems:
- `GET /api/stato`: cameras, delay and latest decision;
- `GET /api/eventi?dopo=N`: new events;
- `GET /flusso/<camera>.mjpg`: the live annotated image.

## Results

Measured on the **test set**: 15% of the videos, split by video (never the same video in training and test).

**Violence yes/no, the delivered system (skeletons + video network)**

| Measure | Accuracy | Recall | False alarms |
|---|---|---|---|
| per window (1.6 s) | 0.78 | 0.82 | 0.25 |
| 10 s segments (272) | 0.82 (95% CI 0.77–0.89) | 0.80 | 0.16 |
| **30 s segments (167)** | **0.90 (95% CI 0.84–0.94)** | **0.90** | **0.11** |

By source, on 30 s segments:

| Source | Accuracy |
|---|---|
| RWF-2000 | 0.91 |
| CCTV-Fights | 0.88 |
| UCF-Crime | 0.82 |
| SCVD | 1.00 |

Skeletons only: 0.78. Video network only: 0.84. Together: 0.90.

**4 classes** (ST-GCN, 32,995 test windows)

| | Normal | Violence | Armed violence | Snatching |
|---|---|---|---|---|
| F1 | 0.77 | 0.58 | 0.10 | 0.13 |
| recall | 0.81 | 0.51 | 0.18 | 0.15 |

Macro-F1 0.40. For comparison, the Graph-Transformer reaches 0.29.

**Weapon detector (v3)**, mAP50:
- 0.77 on the hand-held weapons of Sohas;
- 0.20 on the real CCTV of US Mock Attack (pistols 0.32, rifles 0.28, knives ~0);
- 2 false alarms on 326 photos of kitchen knives, phones and scissors.

A bigger detector (YOLO26m at 1280 px) sees more pistols in real CCTV (0.41), but does not improve knives and is 3–4 times slower: it stayed an experiment.

**Crowds.** In crowded scenes never seen in training (stations, campuses, streets) the ST-GCN gives the right class in 82–98% of the windows.

## Limits

We say them plainly:
- **Armed violence and snatching are weak** (F1 0.10 and 0.13). There are few public examples: about 600 snatching windows against tens of thousands of normal ones.
- **The WEAPON WARNING rarely fires**: precision 0.35, recall 0.03. It is tuned not to raise false alarms, and so it misses many real weapons.
- **Knives are almost never seen in real CCTV.** They are small, low resolution and often hidden by the hand.
- **UCF-Crime is the hardest source**: long, low-quality videos with many people.
- **In very dense crowds** tracking sometimes loses people.
- **Without a GPU** the panel works delayed, not in real time.
- **On real videos never seen before**, taken from YouTube under CC BY:
  - metro, escalators and carriages always stay normal, with no false alarms;
  - an armed robbery is recognised as violence, with an ALARM, but without a WEAPON WARNING;
  - a street snatching where the victim fights back is recognised as violence, with an ALARM;
  - a fight filmed vertically with a phone and two real snatchings (a bag pulled away, a snatching at gunpoint) are not recognised.

  The analyses are in `runs/argo_casi_uso/`, the sources in [`annotations/video_pubblici/attribuzioni.csv`](annotations/video_pubblici/attribuzioni.csv).
- ARGO is an **aid for whoever watches the cameras**, not a system that decides on its own. Every alarm must be checked by a person.

## Training from scratch

The heavy computing was done on **Kaggle** (2× T4). The notebooks are in [`kaggle/`](kaggle) and are launched with `kaggle/lancia.sh`. The order is:
1. weapon detector: `detector_armi`;
2. skeleton and weapon extraction from all videos: `estrazione`, `aggiungi_armi`;
3. windows: `finestre`;
4. 4-class training: `addestramento`;
5. video network fine-tuning: `finetune_rgb`.

The code is in [`src/sicurezza/`](src/sicurezza): `dati.py`, `visione.py`, `modelli.py`, `demo.py` (with the panel) and `strumenti.py`. Run the tests with `uv run pytest`.

## Data, weights and licences

- **Code**: [AGPL-3.0](LICENSE), like Ultralytics YOLO, on which it is built.
- **Weights** (Release v1.0): **research and non-commercial use only**. They are also trained on datasets distributed for academic use only.
- **Training datasets** (not included in the repo):
  - [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection): academic use, the authors ask not to redistribute it;
  - CCTV-Fights (Kaggle copy [`mohamedtariki/vsd-cctv-fights`](https://www.kaggle.com/datasets/mohamedtariki/vsd-cctv-fights));
  - [SCVD](https://www.kaggle.com/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd);
  - [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) with the [UCA](https://github.com/Xuange923/Surveillance-Video-Understanding) annotations (Apache 2.0, in `annotations/uca/`);
  - [Chain-snatching](https://www.kaggle.com/datasets/aniketsahu09/chain-snatching-cctv-dataset);
  - CC BY snatching videos: attributions in [`annotations/phone_snatching/attribuzioni.csv`](annotations/phone_snatching/attribuzioni.csv);
  - for crowd scenes: MOT17/MOT20, HT21, ShanghaiTech, UBI-Fights;
  - for weapons: Roboflow ([Master Weapon CCTV](https://universe.roboflow.com/beoeoa/master_weapon_cctv), [CCTV Weapon Detect](https://universe.roboflow.com/prospace-gi6e4/cctv-weapon-detect-4vuah)), US Mock Attack, Sohas.
- **Videos in the presentation and on the site**:
  - CC BY clips from YouTube and Pexels videos, with attributions in [`annotations/video_pubblici/attribuzioni.csv`](annotations/video_pubblici/attribuzioni.csv);
  - dataset videos only appear as skeletons on a black background.
- **Music in the video**: "Clash Defiant" Kevin MacLeod (incompetech.com), licensed under Creative Commons: By Attribution 4.0 License.
- **Image**: Peter Paul Rubens, *Juno and Argus* (c. 1610), public domain ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Peter_Paul_Rubens_-_Juno_and_Argus_-_WGA20280.jpg)).
- **Typeface of the ARGO name**: [Megrim](https://fonts.google.com/specimen/Megrim) (SIL Open Font License).

## Credits

Built with the help of Claude (Anthropic) as a programming assistant. The choices are Marco's and are logged step by step.
