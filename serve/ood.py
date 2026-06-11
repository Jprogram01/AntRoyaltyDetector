"""
Out-of-distribution ("is this even an ant?") gate.

The caste model is trained only on ants, so it will confidently mislabel any
non-ant image (a dog, a meme, food) as queen/worker. To catch that, we run the
input through a stock **ImageNet-pretrained** classifier and sum the probability
mass over insect/arachnid classes — an "arthropod-likeness" score.

Empirically (measured on real AntWeb + field ants vs. gray/noise):
  - real ants:  median ~0.7, but a ~5-10% tail scores low (odd macro crops)
  - non-ants:   ~0.05
There's overlap in the tail, so this is a SOFT signal: below the threshold we
*warn* that the input may not be an ant rather than hard-blocking (which would
false-reject real ants). Threshold is tunable via OOD_THRESHOLD.
"""

import os

import torch
from loguru import logger

from data.dataset import VAL_TRANSFORMS

# ImageNet-1k indices for insects (300–326: beetles → butterflies) and
# arachnids/myriapods (70–79: spiders, scorpion, tick, centipede). Summing
# these is far more robust than the single "ant" class (310), which is noisy.
_INSECT_IDX = sorted(set(range(300, 327)) | set(range(70, 80)))

OOD_THRESHOLD = float(os.getenv("OOD_THRESHOLD", "0.10"))
_GATE_BACKBONE = os.getenv("OOD_BACKBONE", "efficientnet_b0")
DEVICE = os.getenv("DEVICE", "cpu")

_gate = None


def _load():
    global _gate
    if _gate is None:
        import timm

        logger.info(f"Loading OOD gate ({_GATE_BACKBONE}, ImageNet-1k)")
        _gate = timm.create_model(_GATE_BACKBONE, pretrained=True, num_classes=1000)
        _gate.eval().to(DEVICE)
    return _gate


def warmup() -> None:
    """Pre-load the gate model at startup (downloads weights on first run)."""
    try:
        _load()
        logger.info("OOD gate ready")
    except Exception as exc:
        logger.warning(f"OOD gate unavailable (will skip gating): {exc}")


@torch.no_grad()
def arthropod_score(img) -> float:
    """Return P(insect/arachnid) in [0,1] for a PIL image. Reuses the caste
    model's ImageNet preprocessing (224, ImageNet normalization)."""
    model = _load()
    x = VAL_TRANSFORMS(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x), dim=1)[0]
    return float(probs[_INSECT_IDX].sum())


def check(img) -> tuple[bool, float]:
    """Returns (likely_ant, score). likely_ant is False when the score is below
    OOD_THRESHOLD (input probably isn't an ant). Fails open (True) on error."""
    try:
        score = arthropod_score(img)
        return score >= OOD_THRESHOLD, round(score, 4)
    except Exception as exc:
        logger.warning(f"OOD check failed, passing through: {exc}")
        return True, -1.0
