"""
Retraining script — fine-tunes an existing checkpoint on new data.

Usage:
    python retrain.py --checkpoint checkpoints/best.pt --raw-dir data/raw --epochs 5
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import make_loaders
from model.classifier import AntCasteClassifier
from train import compute_class_weights, eval_epoch, train_epoch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-checkpoint", type=Path, default=Path("checkpoints/retrained.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logger.info(f"Retraining from {args.checkpoint} on {DEVICE}")
    model = AntCasteClassifier.load(args.checkpoint, device=DEVICE)
    model.to(DEVICE)

    train_loader, val_loader, _ = make_loaders(
        args.raw_dir, batch_size=args.batch_size, num_workers=args.workers
    )

    pos_weight = compute_class_weights(train_loader)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    if scaler is not None:
        logger.info("CUDA AMP (mixed precision) enabled")

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc, val_auc = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | val_auc={val_auc:.4f}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            model.save(args.out_checkpoint)
            logger.info(f"  ✓ Saved → {args.out_checkpoint}")

    logger.info(f"Retraining complete. Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
