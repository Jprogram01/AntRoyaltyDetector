"""Smoke tests — no real data needed."""

import torch
from model.classifier import AntCasteClassifier


def test_forward_pass():
    model = AntCasteClassifier(backbone="efficientnet_b2", pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2,), f"Expected (2,), got {out.shape}"


def test_save_load(tmp_path):
    model = AntCasteClassifier(backbone="efficientnet_b2", pretrained=False)
    path = tmp_path / "test.pt"
    model.save(path)
    loaded = AntCasteClassifier.load(path)
    loaded.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = loaded(x)
    assert out.shape == (1,)
