"""
FastAPI inference server for AntCasteClassifier.

GET  /          — interactive upload page (browser demo)
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
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from data.dataset import VAL_TRANSFORMS
from model.classifier import AntCasteClassifier

MODEL_PATH = Path(os.getenv("MODEL_PATH", "checkpoints/combined_final.pt"))
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


_LANDING_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ant Royalty Detector</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 3rem auto;
         padding: 0 1rem; line-height: 1.5; }
  h1 { margin-bottom: .25rem; }
  .sub { color: #888; margin-top: 0; }
  .card { border: 1px solid #8884; border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem; }
  button { font-size: 1rem; padding: .6rem 1.2rem; border-radius: 8px; border: 0;
           background: #6d28d9; color: #fff; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  #result { margin-top: 1.25rem; font-size: 1.1rem; }
  .caste { font-size: 1.6rem; font-weight: 700; }
  .queen { color: #d97706; } .worker { color: #2563eb; }
  img#preview { max-width: 100%; border-radius: 8px; margin-top: 1rem; display: none; }
  .bar { height: 10px; background: #8883; border-radius: 5px; overflow: hidden; margin-top:.5rem; }
  .bar > span { display:block; height:100%; background:#6d28d9; }
</style></head>
<body>
  <h1>🐜 Ant Royalty Detector</h1>
  <p class="sub">Upload an ant photo — the model predicts <b>queen</b> vs <b>worker</b>.</p>
  <div class="card">
    <input type="file" id="file" accept="image/*">
    <button id="go" disabled>Classify</button>
    <img id="preview">
    <div id="result"></div>
  </div>
  <p class="sub">EfficientNet-B2 · trained on AntWeb + field images · <a href="/docs">API docs</a></p>
<script>
const fileEl=document.getElementById('file'), go=document.getElementById('go'),
      res=document.getElementById('result'), prev=document.getElementById('preview');
fileEl.onchange=()=>{ go.disabled=!fileEl.files.length; res.innerHTML='';
  if(fileEl.files.length){ prev.src=URL.createObjectURL(fileEl.files[0]); prev.style.display='block'; } };
go.onclick=async()=>{
  go.disabled=true; res.textContent='Classifying…';
  const fd=new FormData(); fd.append('file', fileEl.files[0]);
  try{
    const r=await fetch('/predict',{method:'POST',body:fd});
    if(!r.ok){ res.textContent='Error: '+(await r.json()).detail; go.disabled=false; return; }
    const d=await r.json(); const pct=(d.queen_probability*100).toFixed(1);
    res.innerHTML=`<div class="caste ${d.caste}">${d.caste.toUpperCase()}</div>`+
      `confidence ${(d.confidence*100).toFixed(1)}% · P(queen)=${pct}%`+
      `<div class="bar"><span style="width:${pct}%"></span></div>`+
      `<div class="sub">${d.latency_ms.toFixed(0)} ms</div>`;
  }catch(e){ res.textContent='Request failed: '+e; }
  go.disabled=false;
};
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
async def landing():
    return _LANDING_HTML


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
