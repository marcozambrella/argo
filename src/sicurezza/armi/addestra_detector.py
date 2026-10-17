"""Fine-tuning di YOLO26s sul dataset armi unito (D4). Iperparametri PROVVISORI (DA DECIDERE).

Uso (su Kaggle, GPU):
    python -m sicurezza.armi.addestra_detector --dati armi_unite/data.yaml --uscita runs/armi
"""

import argparse


def addestra_detector(percorso_data_yaml: str, cartella_uscita: str, modello_base: str = "yolo26s.pt",
                      epoche: int = 100, pazienza: int = 20, dimensione_immagine: int = 640, lotto: int = 16,
                      dispositivo: str | int | None = None):
    from ultralytics import YOLO

    modello = YOLO(modello_base)
    modello.train(data=percorso_data_yaml, epochs=epoche, patience=pazienza, imgsz=dimensione_immagine,
                  batch=lotto, device=dispositivo, project=cartella_uscita, name="yolo26s_armi", exist_ok=True,
                  seed=0)
    return modello.val(data=percorso_data_yaml, split="test" if _ha_test(percorso_data_yaml) else "val")


def _ha_test(percorso_data_yaml: str) -> bool:
    import yaml
    return "test" in yaml.safe_load(open(percorso_data_yaml))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dati", required=True)
    parser.add_argument("--uscita", required=True)
    parser.add_argument("--epoche", type=int, default=100)
    parser.add_argument("--dispositivo", default=None)
    argomenti = parser.parse_args()
    metriche = addestra_detector(argomenti.dati, argomenti.uscita, epoche=argomenti.epoche,
                                 dispositivo=argomenti.dispositivo)
    print(f"mAP50 {metriche.box.map50:.4f}  mAP50-95 {metriche.box.map:.4f}")


if __name__ == "__main__":
    main()
