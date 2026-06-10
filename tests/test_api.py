"""API smoke tests with a synthetic checkpoint."""

import io
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from model.classifier import AntCasteClassifier


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    ckpt_dir = tmp_path_factory.mktemp("checkpoints")
    ckpt_path = ckpt_dir / "best.pt"
    model = AntCasteClassifier(backbone="efficientnet_b2", pretrained=False)
    model.save(ckpt_path)

    import os
    os.environ["MODEL_PATH"] = str(ckpt_path)

    # Import after env var is set so startup picks it up
    import importlib
    import serve.app as app_module
    importlib.reload(app_module)

    return TestClient(app_module.app)


def make_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict(client):
    img_bytes = make_jpeg_bytes()
    resp = client.post(
        "/predict",
        files={"file": ("ant.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["caste"] in ("queen", "worker")
    assert 0.0 <= data["queen_probability"] <= 1.0
    assert data["latency_ms"] > 0


def test_predict_invalid_file(client):
    resp = client.post(
        "/predict",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400
