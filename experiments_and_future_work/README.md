# Experiments and future work

Nothing in this folder is needed to run ARGO or to rebuild the delivered weights (that is [`../training/`](../training)).
These are the tries that were discarded, measured and not adopted, or used only to take a decision.
They stay here so every decision in the project can be checked.

| What | What it was | Outcome |
|---|---|---|
| `prova_rgb/` | First test of a video network on pixels: R(2+1)D-18 (Kinetics-400), frozen, plus a logistic head for violence yes/no | It worked, and led to the pixel branch; replaced by the fine-tuned network in `training/finetune_rgb` |
| `prova_rgb_clip/` | The same frozen network on every window of whole clips, to measure the alarm rule on 10/30 s pieces | Measurement only |
| `testa_rgb.npz` | Logistic head of the frozen video network | Replaced by `testa_violenza.npz` (fine-tuned network, Release v1.0) |
| `addestramento_lungo/` | Longer training of the two skeleton classifiers on a second Kaggle account | +1.5–3 points of 4-class macro-F1, nothing on violence: not adopted |
| `detector_armi_v4/` | Bigger weapon detector: YOLO26m at 1280 px instead of YOLO26s at 640 | More pistols in real CCTV (0.41 vs 0.32), knives not improved, 3–4× slower on CPU: v3 kept |
| `colab_addestra_mani.ipynb` | Zoom on the hands: crops around the wrists plus a dedicated classifier, for small knives | Good on its test set, did not discriminate on real videos: declared as a limit |
| `confronto_addestramenti/` | Training 1 vs training 2 on the same test windows | Decided which skeleton classifier to ship (training 1) |
| `analisi/` | Metrics per source and per video of the test set, CPU only | Used to understand errors and to pick demo videos |
| `gpu_esterna/addestra.sh` | Training 2 on a GPU outside Kaggle (RunPod, Vast, any NVIDIA PC) | Ready but never used: the longer training ran on Kaggle (`addestramento_lungo/`) |
| `valutazione/` | CPU-only evaluation of the whole system after a training done outside Kaggle | Companion of `gpu_esterna/`, not needed |

## Future work

- Re-extract the tracks with a tracker that loses fewer people in crowds (BoT-SORT).
- Small knives in CCTV: still about 0 for every detector tried; it needs higher resolution or a different approach.
- A GPU for real-time analysis: on a laptop CPU the camera panel runs with a delay.

Kernels in this folder are launched like the training ones: `training/lancia.sh ../experiments_and_future_work/<name>`.
