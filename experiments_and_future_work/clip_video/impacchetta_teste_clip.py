"""Goal 80% (08/11): le teste CLIP B/16 scelte in un file solo per il sistema (pesi/clip_teste.pt, mai nel repo):
armi = 2 frame × 3 viste (G8), scippo = 2 frame × 3 viste + differenze (G9), 3 semi ciascuna; soglie scelte sul val
unito (vecchi + nuovi) alle cadenze del sistema (armi ogni 4 s, scippo ogni 0,8 s), con
misura_vecchi_e_nuovi.py (qui accanto) sui punteggi diradati."""
import json
from pathlib import Path

import torch

C = Path("runs/esperimenti_armi_scippi/clip_video")
armi = json.loads(Path("runs/esperimenti_armi_scippi/misura_clipB_2frame.json").read_text())["clipB_2frame_4s"]
scippo = json.loads(Path("runs/esperimenti_armi_scippi/misura_clipB_diff.json").read_text())["clipB_diff2f_0.8s"]
teste_armi = []
teste_scippo = []
for seme in (0, 1, 2):
    teste_armi.append(torch.load(C / f"testa_ViT-B-16_{seme}_viste3_frame2_diff0.pt"))
    teste_scippo.append(torch.load(C / f"testa_ViT-B-16_{seme}_viste3_frame2_diff1.pt"))
pacchetto = {"modello_clip": "ViT-B-16", "pesi_clip": "openai", "armi": teste_armi, "scippo": teste_scippo,
             "soglie": {"arma": armi["arma"]["soglia"], "fuoco": armi["fuoco"]["soglia"],
                        "coltello": armi["coltello"]["soglia"], "scippo": scippo["scippo"]["soglia"]}}
torch.save(pacchetto, "pesi/clip_teste.pt")
print("pesi/clip_teste.pt:", pacchetto["soglie"])
