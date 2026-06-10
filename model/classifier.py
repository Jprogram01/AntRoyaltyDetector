"""
Queen/worker classifier built on a pretrained backbone via timm.
Default backbone: efficientnet_b2 — good accuracy/speed tradeoff for specimen images.
"""

from pathlib import Path
from typing import Optional

import timm
import torch
import torch.nn as nn


class AntCasteClassifier(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b2", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),  # binary: queen (1) vs worker (0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats).squeeze(1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backbone": self.backbone.default_cfg["architecture"],
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "AntCasteClassifier":
        # weights_only=True: our checkpoint is just a backbone-name str + a
        # state_dict of tensors, so the safe loader handles it (and avoids the
        # arbitrary-pickle security risk flagged by recent torch versions).
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = cls(backbone=ckpt["backbone"], pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
