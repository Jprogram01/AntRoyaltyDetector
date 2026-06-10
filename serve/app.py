"""
FastAPI inference server for AntCasteClassifier.

POST /predict   — upload an image, get queen/worker prediction + confidence
GET  /health    — liveness probe
GET  /metrics   — Prometheus metrics (via prometheus-fastapi-instrumentator)
"""

import io
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from data.dataset import VAL_TRANSFORMS
from model.classifier import AntCasteClassifier

MODEL_PATH = Path(os.getenv("MODEL_PATH", "checkpoints/best.pt"))
DEVICE = os.getenv("DEVICE", "cpu")

app = FastAPI(
    title="Ant Caste Classifier",
    description="Classifies ant specimen images as queen or worker.",
    version="0.1.0",
)
Instrumentator().instrument(app).expose(app)

_model: AntCasteClassifier | None = None


def get_model() -> AntCasteClassifier:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model checkpoint not found at {MODEL_PATH}")
        logger.info(f"Loading model from {MODEL_PATH} on {DEVICE}")
        _model = AntCasteClassifier.load(MODEL_PATH, device=DEVICE)
        _model.to(DEVICE)
    return _model


@app.on_event("startup")
async def startup_event():
    try:
        get_model()
        logger.info("Model loaded and ready.")
    except RuntimeError as e:
        logger.warning(f"Model not pre-loaded at startup: {e}")


class PredictionResponse(BaseModel):
    caste: str
    confidence: float
    queen_probability: float
    latency_ms: float


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    tensor = VAL_TRANSFORMS(img).unsqueeze(0).to(DEVICE)

    model = get_model()
    t0 = time.perf_counter()
    with torch.no_grad():
        logit = model(tensor)
    latency_ms = (time.perf_counter() - t0) * 1000

    queen_prob = torch.sigmoid(logit).item()
    caste = "queen" if queen_prob >= 0.5 else "worker"
    confidence = queen_prob if caste == "queen" else 1.0 - queen_prob

    logger.info(
        f"predict | caste={caste} | queen_prob={queen_prob:.4f} | "
        f"latency={latency_ms:.1f}ms | file={file.filename}"
    )

    return PredictionResponse(
        caste=caste,
        confidence=round(confidence, 4),
        queen_probability=round(queen_prob, 4),
        latency_ms=round(latency_ms, 2),
    )
