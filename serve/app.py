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
  :root {
    --bg:#fafaf9; --surface:#fff; --ink:#1f2d33; --text:#3a4248; --muted:#76808a;
    --line:#e4e3dd; --accent:#2c6e63; --accent-dark:#235850;
    --queen:#9a6b1f; --worker:#2c6e63;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); line-height: 1.6;
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
         max-width: 620px; margin: 0 auto; padding: 3rem 1.25rem; }
  header { border-bottom: 1px solid var(--line); padding-bottom: 1.1rem; margin-bottom: 1.6rem; }
  h1 { font-family: Georgia, "Times New Roman", serif; font-size: 1.8rem; color: var(--ink);
       margin: 0; display: inline-block; border-bottom: 3px solid var(--accent); padding-bottom: .35rem; }
  .tagline { color: var(--muted); margin: .75rem 0 0; font-size: .95rem; }
  .card { background: var(--surface); border: 1px solid var(--line); border-radius: 6px;
          box-shadow: 0 1px 3px rgba(0,0,0,.06); padding: 1.4rem; }
  .row { display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; }
  input[type=file] { font-size: .9rem; max-width: 100%; }
  button { font: inherit; font-size: .9rem; cursor: pointer; border-radius: 4px; }
  .primary { background: var(--accent); color: #fff; border: 1px solid var(--accent); padding: .5rem 1.1rem; }
  .primary:hover:not(:disabled) { background: var(--accent-dark); border-color: var(--accent-dark); }
  .primary:disabled { opacity: .45; cursor: default; }
  .hint { color: var(--muted); font-size: .82rem; margin-top: .6rem; }
  #preview { max-width: 100%; margin-top: 1rem; display: none; border: 1px solid var(--line); border-radius: 4px; }
  #result { margin-top: 1.4rem; }
  .caste { font-size: 1.6rem; font-weight: 700; letter-spacing: .02em; }
  .caste.queen { color: var(--queen); } .caste.worker { color: var(--worker); }
  .meta { color: var(--muted); font-size: .85rem; margin-top: .2rem; }
  .bar { height: 6px; background: var(--line); border-radius: 3px; margin-top: .6rem; overflow: hidden; }
  .bar > span { display: block; height: 100%; background: var(--accent); }
  hr { border: 0; border-top: 1px solid var(--line); margin: 1.4rem 0; }
  #rate { display: none; }
  #rate .q { font-size: .92rem; color: var(--ink); font-weight: 600; margin-bottom: .6rem; }
  .rbtn { background: transparent; color: var(--accent); border: 1px solid var(--accent);
          padding: .4rem .9rem; margin-right: .5rem; }
  .rbtn:hover:not(:disabled) { background: var(--accent); color: #fff; }
  .rbtn:disabled { opacity: .45; cursor: default; }
  .disclaimer { color: var(--muted); font-size: .76rem; margin-top: .7rem; max-width: 46ch; }
  #thanks { color: var(--accent); font-weight: 600; font-size: .86rem; margin-top: .7rem; display: none; }
  .notant { background: #fdf6ec; border: 1px solid #e7c98a; color: #8a6516; border-radius: 4px;
            padding: .65rem .8rem; margin-bottom: .9rem; font-size: .86rem; }
  footer { border-top: 1px solid var(--line); margin-top: 1.8rem; padding-top: 1rem;
           color: var(--muted); font-size: .82rem; }
  footer a { color: var(--accent); text-decoration: none; }
  footer a:hover { text-decoration: underline; }
</style></head>
<body>
  <header>
    <h1>Ant Royalty Detector</h1>
    <p class="tagline">A computer-vision classifier that distinguishes ant queens from workers.</p>
  </header>
  <div class="card">
    <div class="row">
      <input type="file" id="file" accept="image/*">
      <button id="go" class="primary" disabled>Classify</button>
    </div>
    <div class="hint">or paste an image (Ctrl/Cmd-V)</div>
    <img id="preview">
    <div id="result"></div>
    <div id="rate">
      <hr>
      <div class="q">Was this correct?</div>
      <button class="rbtn" id="yes">Correct</button>
      <button class="rbtn" id="no">Incorrect</button>
      <div class="disclaimer">By rating, you agree your uploaded image and label
        may be stored and used to improve the model.</div>
      <div id="thanks">Saved. Thanks for the feedback.</div>
    </div>
  </div>
  <footer>EfficientNet-B2 &middot; trained on AntWeb specimens and field images
    &middot; <a href="/docs">API documentation</a></footer>
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
      `<div class="caste ${d.caste}">${d.caste.toUpperCase()}</div>`+
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
