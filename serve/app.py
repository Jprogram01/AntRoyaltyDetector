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
from serve import ood, storage

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
    ood.warmup()
    storage.init()


class PredictionResponse(BaseModel):
    caste: str
    confidence: float
    queen_probability: float
    latency_ms: float
    likely_ant: bool
    ant_likelihood: float


_LANDING_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ant Royalty Detector</title>
<style>
  :root { color-scheme: light dark; --muted:#888; --line:#8883; --bd:#8886; }
  * { box-sizing:border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; max-width: 560px;
         margin: 4rem auto; padding: 0 1.25rem; line-height: 1.55; }
  h1 { font-size: 1.4rem; font-weight: 600; margin: 0 0 .25rem; letter-spacing: -.01em; }
  .sub { color: var(--muted); margin: 0; font-size: .9rem; }
  .card { border: 1px solid var(--bd); padding: 1.25rem; margin-top: 1.5rem; }
  input[type=file] { font-size: .88rem; max-width: 100%; }
  button { font: inherit; font-size: .88rem; padding: .45rem 1rem; border: 1px solid currentColor;
           background: transparent; color: inherit; cursor: pointer; }
  button:hover:not(:disabled) { background: #8882; }
  button:disabled { opacity: .4; cursor: default; }
  .hint { color: var(--muted); font-size: .82rem; margin-top: .5rem; }
  #preview { max-width: 100%; margin-top: 1rem; display: none; border: 1px solid var(--line); }
  #result { margin-top: 1.25rem; }
  .caste { font-size: 1.5rem; font-weight: 600; letter-spacing: .03em; }
  .meta { color: var(--muted); font-size: .85rem; margin-top: .15rem; }
  .bar { height: 4px; background: var(--line); margin-top: .6rem; }
  .bar > span { display:block; height:100%; background: currentColor; }
  #rate { margin-top: 1.5rem; display:none; }
  #rate .q { font-size: .9rem; margin-bottom: .5rem; }
  .rbtn { margin-right: .5rem; }
  .disclaimer { color: var(--muted); font-size: .75rem; margin-top: .65rem; max-width: 44ch; }
  #thanks { color: var(--muted); font-size: .85rem; margin-top: .65rem; display:none; }
  .notant { border: 1px solid var(--bd); padding: .6rem .75rem; margin-bottom: .85rem;
            font-size: .86rem; color: var(--muted); }
  footer { margin-top: 1.5rem; color: var(--muted); font-size: .8rem; }
  footer a { color: inherit; }
</style></head>
<body>
  <h1>Ant Royalty Detector</h1>
  <p class="sub">Classifies an ant photo as queen or worker.</p>
  <div class="card">
    <input type="file" id="file" accept="image/*">
    <button id="go" disabled>Classify</button>
    <div class="hint">or paste an image (Ctrl/Cmd-V)</div>
    <img id="preview">
    <div id="result"></div>
    <div id="rate">
      <div class="q">Was this correct?</div>
      <button class="rbtn" id="yes">Correct</button>
      <button class="rbtn" id="no">Incorrect</button>
      <div class="disclaimer">By rating, you agree your uploaded image and label
        may be stored and used to improve the model.</div>
      <div id="thanks">Saved. Thanks for the feedback.</div>
    </div>
  </div>
  <footer>EfficientNet-B2 · AntWeb + field images · <a href="/docs">API</a></footer>
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
    const warn = d.likely_ant ? '' :
      `<div class="notant">This does not look like an ant (arthropod score `+
      `${(d.ant_likelihood*100).toFixed(0)}%); the prediction below is probably unreliable.</div>`;
    res.innerHTML=warn+
      `<div class="caste">${d.caste.toUpperCase()}</div>`+
      `<div class="meta">confidence ${(d.confidence*100).toFixed(1)}% · P(queen) ${pct}%</div>`+
      `<div class="bar"><span style="width:${pct}%"></span></div>`+
      `<div class="meta">${d.latency_ms.toFixed(0)} ms · arthropod ${(d.ant_likelihood*100).toFixed(0)}%</div>`;
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

    # OOD gate: is this even an ant? (soft — caste is still returned)
    likely_ant, ant_likelihood = ood.check(img)

    logger.info(
        f"predict | caste={caste} | queen_prob={queen_prob:.4f} | "
        f"likely_ant={likely_ant} ({ant_likelihood}) | "
        f"latency={latency_ms:.1f}ms | file={file.filename}"
    )

    # Monitoring log — metadata only, NO image (drift / volume / latency / OOD)
    storage.log_prediction(
        {
            "caste": caste,
            "queen_probability": round(queen_prob, 4),
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2),
            "likely_ant": likely_ant,
            "ant_likelihood": ant_likelihood,
            "img_w": img.width,
            "img_h": img.height,
        }
    )

    return PredictionResponse(
        caste=caste,
        confidence=round(confidence, 4),
        queen_probability=round(queen_prob, 4),
        latency_ms=round(latency_ms, 2),
        likely_ant=likely_ant,
        ant_likelihood=ant_likelihood,
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
