"""
Training script for AntCasteClassifier.

Usage:
    python train.py --raw-dir data/raw --epochs 20 --backbone efficientnet_b2
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import classification_report, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset import make_loaders
from model.classifier import AntCasteClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_class_weights(loader) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss from the training loader."""
    n_pos = n_neg = 0
    for _, labels in loader:
        n_pos += labels.sum().item()
        n_neg += (1 - labels).sum().item()
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-6)], device=DEVICE)
    logger.info(f"pos_weight (queen): {pos_weight.item():.2f}")
    return pos_weight


def train_epoch(model, loader, optimizer, criterion, scaler=None) -> float:
    """One training epoch. Uses CUDA AMP (mixed precision) when `scaler` is
    provided — halves VRAM and speeds up Ampere GPUs (important on 4GB cards)."""
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.float().to(DEVICE)
        optimizer.zero_grad()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion) -> tuple[float, float, float]:
    """Returns (loss, accuracy, roc_auc)."""
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels_f = labels.float().to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, labels_f)
        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits).cpu()
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return total_loss / len(loader.dataset), acc, auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--backbone", default="efficientnet_b2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--baseline", action="store_true",
        help="Use Hymenoptera ants/bees data to validate training loop"
    )
    args = parser.parse_args()

    logger.info(f"Device: {DEVICE}")
    logger.info(f"Backbone: {args.backbone}")

    if args.baseline:
        logger.info("BASELINE MODE: using Hymenoptera ants/bees data")

    train_loader, val_loader, test_loader = make_loaders(
        args.raw_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        baseline=args.baseline,
    )
    logger.info(
        f"Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | "
        f"Test: {len(test_loader.dataset)}"
    )

    model = AntCasteClassifier(backbone=args.backbone).to(DEVICE)
    pos_weight = compute_class_weights(train_loader)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Mixed precision on CUDA only (no-op on CPU)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None
    if scaler is not None:
        logger.info("CUDA AMP (mixed precision) enabled")

    # Freeze backbone for first 5 epochs, then unfreeze for fine-tuning
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0.0
    best_path = args.checkpoint_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone at epoch 6
        if epoch == 6:
            logger.info("Unfreezing backbone for fine-tuning")
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=args.lr * 0.1)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch + 1
            )

        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss, val_acc, val_auc = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_auc={val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            model.save(best_path)
            logger.info(f"  ✓ New best AUC={best_auc:.4f} → saved to {best_path}")

    # Final evaluation on test set
    logger.info("--- Test set evaluation ---")
    best_model = AntCasteClassifier.load(best_path, device=DEVICE)
    best_model.to(DEVICE)
    _, test_acc, test_auc = eval_epoch(best_model, test_loader, criterion)
    logger.info(f"Test acc={test_acc:.4f} | Test AUC={test_auc:.4f}")

    # Full classification report
    best_model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            logits = best_model(imgs.to(DEVICE))
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    print(
        classification_report(
            all_labels, all_preds, target_names=["worker", "queen"]
        )
    )

    # Domain-sliced accuracy: separate AntWeb (lab) vs sorted (field/mixed) test
    # images to verify the out-of-distribution gap actually closed after mixing.
    test_subset = test_loader.dataset
    test_paths = [
        str(test_subset.dataset.samples[i][0]) for i in test_subset.indices
    ]
    is_sorted = ["sorted_" in p for p in test_paths]
    for domain, want_sorted in (("sorted (field/mixed)", True), ("AntWeb (lab)", False)):
        idxs = [j for j in range(len(test_paths)) if is_sorted[j] == want_sorted]
        if not idxs:
            continue
        dom_acc = sum(all_preds[j] == all_labels[j] for j in idxs) / len(idxs)
        logger.info(f"  [{domain}] test acc={dom_acc:.4f} (n={len(idxs)})")


if __name__ == "__main__":
    main()
