"""
Evaluate a trained checkpoint on the held-out test split, with a domain-sliced
breakdown (AntWeb lab specimens vs. sorted field/mixed images).

Reproduces the exact train/val/test split from make_loaders (same seed), so the
test set here is genuinely held out from training.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --raw-dir data/raw
"""

import argparse
from pathlib import Path

import torch
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from data.dataset import make_loaders
from model.classifier import AntCasteClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    logger.info(f"Device: {DEVICE} | checkpoint: {args.checkpoint}")
    _, _, test_loader = make_loaders(
        args.raw_dir, batch_size=args.batch_size, num_workers=0
    )

    model = AntCasteClassifier.load(args.checkpoint, device=DEVICE)
    model.to(DEVICE)
    model.eval()

    labels, preds, probs = [], [], []
    for imgs, ys in test_loader:
        logits = model(imgs.to(DEVICE))
        p_queen = torch.sigmoid(logits).cpu()
        probs.extend(p_queen.tolist())
        preds.extend((p_queen >= 0.5).long().tolist())
        labels.extend(ys.tolist())

    acc = sum(a == b for a, b in zip(preds, labels)) / len(labels)
    auc = roc_auc_score(labels, probs)
    logger.info(f"Overall test acc={acc:.4f} | AUC={auc:.4f} (n={len(labels)})")
    print(confusion_matrix(labels, preds))
    print(classification_report(labels, preds, target_names=["worker", "queen"], digits=3))

    # Domain slice: AntWeb (lab) vs sorted (field/mixed)
    test_subset = test_loader.dataset
    paths = [str(test_subset.dataset.samples[i][0]) for i in test_subset.indices]
    is_sorted = ["sorted_" in p for p in paths]
    logger.info("--- domain-sliced accuracy ---")
    for name, want in (("AntWeb (lab)", False), ("sorted (field/mixed)", True)):
        idxs = [i for i in range(len(paths)) if is_sorted[i] == want]
        if not idxs:
            continue
        dom_acc = sum(preds[i] == labels[i] for i in idxs) / len(idxs)
        dom_auc = roc_auc_score(
            [labels[i] for i in idxs], [probs[i] for i in idxs]
        )
        logger.info(f"  [{name}] acc={dom_acc:.4f} | AUC={dom_auc:.4f} (n={len(idxs)})")


if __name__ == "__main__":
    main()
