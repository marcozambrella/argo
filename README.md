<p align="center"><img src="media/argo_banner.jpg" alt="ARGO: a stylised eye and the word ARGO next to a peacock tail full of eyes, from a painting by Rubens" width="100%"></p>

# ARGO

**ARGO watches CCTV cameras and recognises violence, armed violence and snatching, with a WEAPON WARNING when it sees a weapon in someone's hand.**
The name comes from Argus, the hundred-eyed giant of Greek myth who never slept with all his eyes closed.

ARGO connects to cameras (RTSP, webcam, phone, video file) and shows on a web panel, on your own computer:
- the **skeletons** of the people, with their tracking IDs;
- the **class of the scene**, with probabilities;
- the **WEAPON WARNING** and the **VIOLENCE ALARM**;
- an event log and an archive of the recordings.

Everything it sees is also saved to files that other programs can read: keypoints, events, annotated video.

**See it working**: [the site](https://marcozambrella.github.io/argo/).

| | |
|---|---|
| **Violence yes/no**, 30 s segments | **0.90** accuracy (95% CI 0.84–0.94), 11% false alarms |
| Violence yes/no, 10 s segments / per 1.6 s window | 0.82 (CI 0.77–0.89) / 0.78 |
| **Weapon yes/no** (video network b2, every 4 s), 10 s segments | **0.79** (CI 0.70–0.87); firearms 0.71, knives 0.86 (only 12 clips with a knife) |
| **Snatching yes/no** (video network b1, every 0.8 s), 10 s segments | **0.82** (CI 0.76–0.89), partly by recognising the snatching dataset ([limits](#limits)) |
| 4 classes from the skeletons alone (normal, violence, armed violence, snatching) | macro-F1 0.40 |

**v1.2 is twice as fast on a CPU** (7.6 instead of 14.5 seconds of computing per second of video, same laptop, one right after the other) with the same accuracy: it drops the weapon detector and the separate violence network, and takes the violence from the video network b1, which was already running. On 128 never-seen clips (10 s segments) violence is 0.885 against 0.853 of v1.1; weapons and snatching do not change. v1.1 is kept in [`versioni_passate/v1.1`](versioni_passate/v1.1). The numbers above were measured with v1.1.

## How it works

```
camera ─ffmpeg 25 fps─► YOLO26s-pose + tracking ─► skeletons (17 points per person)
                                                              │
          40-frame windows (1.6 s), one every 10 frames ─► ST-GCN on each skeleton ─┐
          the same 40 frames, every 0.8 s ─► video network b1 (112 px) ─► violence ─┤
                                                                                     ▼
              normal / violence / armed violence ─► VIOLENCE ALARM after 7 violent windows in a row
                                     b1 (the same call) ─► SNATCHING
          the same 40 frames, every 4 s ─► video network b2 (224 px) ─► WEAPON WARNING
```

1. **Skeletons.** YOLO26s-pose finds people and their 17 keypoints: head, shoulders, elbows, wrists and so on. The tracker (TrackTrack) gives each person an ID that lasts over time.
2. **Movement over time.** Every 0.4 s the last 1.6 s of skeletons go to an **ST-GCN**: 10 blocks, each one a convolution on the graph of the bones (every joint gathers information from the joints it is linked to) and then a convolution over time (every joint looks at itself in 9 nearby frames). Each person goes through the same network and the most suspicious one counts. Out come the probabilities of the 4 classes. A Graph-Transformer that links every joint to all the others was tested too, and discarded (macro-F1 0.29 against 0.40).
3. **Pixels.** Two **R(2+1)D-18** video networks, fine-tuned with five yes/no answers (violence, weapon, firearm, knife, snatching), look at the same 40 frames. **b1** at 112 px runs every 0.8 s (a snatching lasts 2–3 s): its violence is averaged with the skeletons' one, and the skeletons then pick the class (violence or armed violence; normal or snatching); its snatching decides the snatching class, unless the scene is armed violence. **b2** at 224 px runs every 4 s (calling it more often does not help: a weapon stays in the scene) and gives the WEAPON WARNING; violence plus WEAPON WARNING is armed violence. On a CPU both run with ONNX Runtime (20% faster, same outputs).
4. **Up to v1.1** a YOLO26s weapon detector (trained on Roboflow, US Mock Attack and Sohas) drew the weapons and fed the skeletons, and a separate video network gave the violence. Without them the accuracy stays the same and the system takes half the time.
5. **Alarms.** The VIOLENCE ALARM fires after 7 violent windows in a row (they cover 4 s of video). That rule was chosen on the validation set, and it is the one the numbers above are measured with.

More details in the hackathon slides (*The project journey* and *The system in depth*).

## Installation

You need **Python 3.12** with [uv](https://docs.astral.sh/uv/) and **ffmpeg**. Tested on Linux (Arch). Webcams as `/dev/video0` are Linux only.

```bash
git clone https://github.com/marcozambrella/argo.git
cd argo
uv sync                   # environment and dependencies (CPU torch; with an NVIDIA GPU install torch with CUDA)
```

The trained weights are **not public** (see [Data, weights and licences](#data-weights-and-licences)). Without them ARGO still starts, in **skeletons-only mode**: people, keypoints and tracking, with no classes or alarms. The pose model downloads itself the first time.

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
- **On a laptop without a GPU** it analyses about 3 frames/s. Cameras run at 25, so ARGO records everything in 10 s segments and analyses behind them, **delayed**. The delay is shown on the camera tile, and no frame is lost.

To analyse a single video, without the panel:

```bash
uv run argo analizza-video --video video.mp4 --classificatore pesi/stgcn.pt \
  --rete-scippi pesi/rete_scippi_b1.onnx --rete-armi pesi/rete_armi_b2.onnx --uscita output/
```

(With a GPU use the `.pt` files of b1 and b2 and `--dispositivo cuda`.)

## What comes out on your computer

Each camera saves a session in `sessioni/<camera>/<date-time>/`. `analizza-video` writes the same files to its `--uscita` folder. (Code, folders and field names are in Italian.)

| File | Content |
|---|---|
| `video_annotato.mp4` | the video with skeletons, IDs and the class panel drawn on it |
| `eventi.jsonl` / `eventi.json` | one line every 0.4 s: `inizio_secondi` (start), `fine_secondi` (end), `classe` (class), `probabilita` of the 4 classes, `warning_arma` (weapon warning), `allarme_violenza` (violence alarm); with the video networks also `probabilita_arma_video` (b2), `probabilita_violenza_video` and `probabilita_scippo_video` (b1) and `allarme_scippo` (snatching) |
| `keypoint.jsonl` | one line per frame: for each person `id`, `box` [x1, y1, x2, y2] and 17 `keypoint` [x, y, confidence] in pixels (COCO order); `armi` (weapons) stays empty since v1.2, which has no weapon detector |
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

**Weapon detector (v3, used up to v1.1)**, mAP50:
- 0.77 on the hand-held weapons of Sohas;
- 0.20 on the real CCTV of US Mock Attack (pistols 0.32, rifles 0.28, knives ~0);
- 2 false alarms on 326 photos of kitchen knives, phones and scissors.

A bigger detector (YOLO26m at 1280 px) sees more pistols in real CCTV (0.41), but does not improve knives and is 3–4 times slower: it stayed an experiment.

**Crowds.** In crowded scenes never seen in training (stations, campuses, streets) the ST-GCN gives the right class in 82–98% of the windows.

## Limits

We say them plainly:
- **From the skeletons alone, armed violence and snatching are weak** (F1 0.10 and 0.13): that is why v1.1 gives them to the video networks.
- **The video networks are measured on few clips**: 18 armed clips in the test set (38 with the cross measure), so the intervals are wide. The WEAPON WARNING of b2 catches 64% of the armed 10 s segments and fires on 14% of the segments of normal crowds.
- **Snatching**: b1 is measured mostly on chain-snatching videos, which have no normal videos of their own; inside UCF-Crime it is still at chance. More snatching videos with normal footage from the same cameras are being annotated.
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

The trained weights are not public; these notebooks are how they were made.

The heavy computing was done on **Kaggle** (2× T4). The notebooks that rebuild the delivered weights are in [`training/`](training) and are launched with `training/lancia.sh <name>`. The order is:
1. weapon detector: `detector_armi`;
2. skeleton and weapon extraction from all videos: `estrazione`, `aggiungi_armi`;
3. windows: `finestre`;
4. 4-class training: `addestramento`;
5. video network fine-tuning: `finetune_rgb` (the violence network of v1.1); b1 and b2 come from [`experiments_and_future_work/rete_video`](experiments_and_future_work/rete_video).

The weapon detector is still needed for the extraction (steps 2–3), because the ST-GCN was trained with the weapons as extra nodes; at analysis time v1.2 leaves them empty, which does not change its answers.

Discarded tries and ideas not shipped (bigger detector, hand zoom, longer training…) are in [`experiments_and_future_work/`](experiments_and_future_work), each with its outcome. The previous version of the system (v1.1) is in [`versioni_passate/`](versioni_passate).

The code is in [`src/sicurezza/`](src/sicurezza): `dati.py` (videos, labels, splits), `visione.py` (skeletons, tracking, weapons), `modelli.py` (windows, ST-GCN, training), `demo.py` (video analysis and the panel) and `strumenti.py` (annotation and download tools). It is written to be easy for a team to read: plain loops and `if`/`else`, one step per line, with comments that explain why. Run the tests with `uv run pytest`.

## Data, weights and licences

- **Code**: [AGPL-3.0](LICENSE), like Ultralytics YOLO, on which it is built.
- **Weights**: **not public**. On request, for **research and non-commercial use only**: they are also trained on datasets distributed for academic use only.
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
- **Music in the video**: "Clash Defiant" and "Echoes of Time v2" Kevin MacLeod (incompetech.com), licensed under Creative Commons: By Attribution 4.0 License.
- **Image**: Peter Paul Rubens, *Juno and Argus* (c. 1610), public domain ([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Peter_Paul_Rubens_-_Juno_and_Argus_-_WGA20280.jpg)).
- **Typeface of the ARGO name**: [Megrim](https://fonts.google.com/specimen/Megrim) (SIL Open Font License).

## Credits

Built with the help of Claude (Anthropic) as a programming assistant. The choices are Marco's and are logged step by step.
