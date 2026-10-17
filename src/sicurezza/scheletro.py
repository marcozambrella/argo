"""Scheletro COCO a 17 keypoint (formato dei modelli YOLO pose)."""

NOMI_KEYPOINT = (
    "naso", "occhio_sx", "occhio_dx", "orecchio_sx", "orecchio_dx", "spalla_sx", "spalla_dx",
    "gomito_sx", "gomito_dx", "polso_sx", "polso_dx", "anca_sx", "anca_dx",
    "ginocchio_sx", "ginocchio_dx", "caviglia_sx", "caviglia_dx",
)
NUMERO_KEYPOINT = len(NOMI_KEYPOINT)
POLSI = (NOMI_KEYPOINT.index("polso_sx"), NOMI_KEYPOINT.index("polso_dx"))

OSSA = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)

# per il ribaltamento orizzontale (data augmentation): sinistra e destra si scambiano
SPECCHIO_KEYPOINT = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)
CENTRO_SCHELETRO = 0  # naso come radice per la partizione "spaziale" di ST-GCN
