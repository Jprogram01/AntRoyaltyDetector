"""
FastAPI inference server for AntCasteClassifier.

GET  /          — interactive upload page (browser demo)
POST /predict   — upload an image, get queen/worker prediction + confidence
POST /feedback  — user rates a prediction; image + label saved for retraining
GET  /health    — liveness probe
GET  /metrics   — Prometheus metrics (via prometheus-fastapi-instrumentator)
"""

import io
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from data.dataset import VAL_TRANSFORMS
from model.classifier import AntCasteClassifier
from serve import storage

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
    storage.init()


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
  #rate { margin-top: 1.25rem; display:none; }
  #rate .q { font-size:.95rem; margin-bottom:.5rem; }
  .rbtn { background:#0000; border:1px solid #8886; color:inherit; margin-right:.5rem; }
  .disclaimer { font-size:.78rem; color:#999; margin-top:.6rem; }
  #thanks { color:#16a34a; font-weight:600; margin-top:.5rem; display:none; }
</style></head>
<body>
  <h1>🐜 Ant Royalty Detector</h1>
  <p class="sub">Upload an ant photo — the model predicts <b>queen</b> vs <b>worker</b>.</p>
  <div class="card">
    <input type="file" id="file" accept="image/*">
    <button id="go" disabled>Classify</button>
    <div class="sub" style="margin-top:.4rem">…or paste an image (Ctrl/⌘+V)</div>
    <img id="preview">
    <div id="result"></div>
    <div id="rate">
      <div class="q">Was this right?</div>
      <button class="rbtn" id="yes">👍 Correct</button>
      <button class="rbtn" id="no">👎 It's the other one</button>
      <div class="disclaimer">By rating, you agree your uploaded image and label
        may be stored and used to improve the model.</div>
      <div id="thanks">✓ Thanks — saved for future training!</div>
    </div>
  </div>
  <p class="sub">EfficientNet-B2 · trained on AntWeb + field images · <a href="/docs">API docs</a></p>
<script>
const fileEl=document.getElementById('file'), go=document.getElementById('go'),
      res=document.getElementById('result'), prev=document.getElementById('preview'),
      rate=document.getElementById('rate'), thanks=document.getElementById('thanks'),
      yes=document.getElementById('yes'), no=document.getElementById('no');
let lastCaste=null, currentFile=null;
function setImage(file){
  currentFile=file||null; go.disabled=!currentFile; res.innerHTML='';
  rate.style.display='none'; thanks.style.display='none';
  if(currentFile){ prev.src=URL.createObjectURL(currentFile); prev.style.display='block'; }
}
fileEl.onchange=()=>setImage(fileEl.files[0]);
document.addEventListener('paste', e=>{
  const items=(e.clipboardData||{}).items||[];
  for(const it of items){ if(it.type && it.type.startsWith('image/')){ setImage(it.getAsFile()); break; } }
});
go.onclick=async()=>{
  if(!currentFile) return;
  go.disabled=true; res.textContent='Classifying…'; rate.style.display='none'; thanks.style.display='none';
  const fd=new FormData(); fd.append('file', currentFile);
  try{
    const r=await fetch('/predict',{method:'POST',body:fd});
    if(!r.ok){ res.textContent='Error: '+(await r.json()).detail; go.disabled=false; return; }
    const d=await r.json(); const pct=(d.queen_probability*100).toFixed(1);
    lastCaste=d.caste;
    res.innerHTML=`<div class="caste ${d.caste}">${d.caste.toUpperCase()}</div>`+
      `confidence ${(d.confidence*100).toFixed(1)}% · P(queen)=${pct}%`+
      `<div class="bar"><span style="width:${pct}%"></span></div>`+
      `<div class="sub">${d.latency_ms.toFixed(0)} ms</div>`;
    yes.disabled=no.disabled=false; rate.style.display='block';
  }catch(e){ res.textContent='Request failed: '+e; }
  go.disabled=false;
};
async function sendFeedback(correct){
  yes.disabled=no.disabled=true;
  const fd=new FormData();
  fd.append('file', currentFile);
  fd.append('predicted_caste', lastCaste);
  fd.append('correct_caste', correct ? lastCaste : (lastCaste==='queen'?'worker':'queen'));
  fd.append('consent', 'true');
  try{ await fetch('/feedback',{method:'POST',body:fd}); thanks.style.display='block'; }
  catch(e){ thanks.textContent='Could not save feedback.'; thanks.style.display='block'; }
}
yes.onclick=()=>sendFeedback(true);
no.onclick=()=>sendFeedback(false);
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

    # Monitoring log — metadata only, NO image (drift / volume / latency)
    storage.log_prediction(
        {
            "caste": caste,
            "queen_probability": round(queen_prob, 4),
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2),
            "img_w": img.width,
            "img_h": img.height,
        }
    )

    return PredictionResponse(
        caste=caste,
        confidence=round(confidence, 4),
        queen_probability=round(queen_prob, 4),
        latency_ms=round(latency_ms, 2),
    )


@app.post("/feedback")
async def feedback(
    file: UploadFile = File(...),
    predicted_caste: str = Form(...),
    correct_caste: str = Form(...),
    consent: bool = Form(...),
):
    """
    Record a user's rating of a prediction. The image + corrected label are
    saved to the feedback dataset (retraining corpus) ONLY with explicit consent.
    `correct_caste` is what the user says it actually is (queen/worker).
    """
    if not consent:
        raise HTTPException(status_code=400, detail="Consent required to store feedback")
    if correct_caste not in ("queen", "worker"):
        raise HTTPException(status_code=400, detail="correct_caste must be queen or worker")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    fb_id = storage.save_feedback(
        contents,
        {
            "predicted_caste": predicted_caste,
            "correct_caste": correct_caste,
            "agreed": predicted_caste == correct_caste,
        },
    )
    return {"status": "thanks", "id": fb_id}
