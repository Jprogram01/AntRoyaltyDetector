---
title: Ant Royalty Detector
emoji: 🐜
colorFrom: purple
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Ant Royalty Detector

Binary image classifier that distinguishes **ant queens from workers** using transfer learning on AntWeb specimen photographs. Built as a production-grade ML Engineering portfolio project.

> **Live demo:** this repo doubles as a [Hugging Face Space](https://huggingface.co) — the YAML header above configures it. Open the Space URL to upload an ant photo and get a queen/worker prediction in the browser.

## Why this is interesting

Queen/worker classification is understudied in the literature — existing ant datasets are mostly species-level or single-class "ant" detection. AntWeb is the only large-scale caste-labeled source, and the queen/worker split is heavily skewed toward workers. The engineering story is: **build the data pipeline, fix the imbalance, serve it properly**.

## Architecture

End-to-end lifecycle: **build → package → deploy → serve → collect → retrain**.

```
 BUILD (local GPU)
   GBIF mirror ──► data/download_gbif.py ──► data/dataset.py ──► train.py ──► combined_final.pt
   (caste in DwC `sex`,   (verbatim de-contam,   (WeightedRandomSampler   (EfficientNet-B2,
    Cloudflare CDN)        label-image filter)     + pos_weight)            2-phase fine-tune)

 SHIP (git push, two remotes)
   GitHub  ◄── source mirror (read-only)
   HF Space ◄── push triggers Docker build  ── model shipped via Git LFS

 RUN (Hugging Face Space, Docker container)
   serve/app.py  FastAPI ── /predict ─┬─ caste classifier (EfficientNet-B2)
                                      └─ serve/ood.py  "is it an ant?" gate (ImageNet B0)
                 ── /feedback ── user rating + image (with consent)
                 ── /health /metrics (Prometheus)

 FLYWHEEL (persist → retrain)
   /predict  ──► HF Dataset: ant-predictions   (metadata only — monitoring/drift)
   /feedback ──► HF Dataset: ant-feedback      (image + corrected label)
                      │
                      └──► retrain.py (fine-tune) ──► evaluate.py (gate) ──► push ──► redeploy
```

**GitHub vs. Hugging Face:** GitHub is the read-only *source mirror*; the HF Space is the
*running deployment*. A Space is itself a git repo — pushing to it triggers a Docker build &
container run (git-push-to-deploy). The same local repo pushes to both remotes.

**Docker:** the `Dockerfile` packages app + deps + model into one reproducible image that runs
identically locally (`docker compose up`) and on HF. The Space's `README` YAML header
(`sdk: docker`, `app_port: 7860`) configures it.

**Continual maintenance:** predictions and user ratings persist to HF Datasets (Spaces have an
ephemeral filesystem, so writes go to Dataset repos via `huggingface_hub.CommitScheduler`).
`retrain.py` fine-tunes on accumulated feedback; `evaluate.py` gates the release with a
domain-sliced check; redeploy closes the loop.

## Quickstart

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download data

```bash
python -m data.download_antweb --queens 2000 --workers 2000
```

Images land in `data/raw/queen/` and `data/raw/worker/`.

### 3. Train

```bash
python train.py --raw-dir data/raw --epochs 20 --backbone efficientnet_b2
```

Best checkpoint saved to `checkpoints/best.pt`.

### 4. Serve locally

```bash
uvicorn serve.app:app --reload
# POST http://localhost:8000/predict  (multipart image upload)
# GET  http://localhost:8000/health
# GET  http://localhost:8000/metrics
```

### 5. Docker

```bash
docker compose up --build
```

Prometheus scrapes `/metrics` every 15 s — view at `http://localhost:9090`.

### 6. Retrain on new data

```bash
python retrain.py --checkpoint checkpoints/best.pt --raw-dir data/raw --epochs 5
```

## Imbalance handling

| Technique | Where |
|---|---|
| `WeightedRandomSampler` | `data/dataset.py` — oversamples queen class during training |
| `BCEWithLogitsLoss(pos_weight=…)` | `train.py` — up-weights queen loss proportional to class ratio |
| Heavy augmentation on train set | `data/dataset.py` — crop, flip, rotation, colour jitter |

## Tests

```bash
pytest tests/ -v
```

## Project layout

```
├── data/
│   ├── download_antweb.py   # AntWeb API scraper
│   └── dataset.py           # Dataset + DataLoader factory
├── model/
│   └── classifier.py        # EfficientNet-B2 + classification head
├── serve/
│   └── app.py               # FastAPI inference server
├── tests/
│   ├── test_model.py        # forward pass + save/load
│   └── test_api.py          # endpoint smoke tests
├── monitoring/
│   └── prometheus.yml
├── train.py                 # main training loop
├── retrain.py               # fine-tune on new data
├── Dockerfile
└── docker-compose.yml
```

## Results

### Baseline — Hymenoptera (ants vs. bees)

Validates the full training loop end-to-end (transfer learning, 2-phase
freeze→fine-tune, weighted sampling, checkpoint-by-AUC) before AntWeb caste
data is available. EfficientNet-B2, 10 epochs, CPU.

| Metric | Value |
|---|---|
| Test Accuracy | 0.923 |
| Test AUC | 0.979 |
| Best Val AUC | 0.991 |
| Worker F1 | 0.88 |

Val AUC climbed 0.947 → 0.991 across 10 epochs; backbone unfroze at epoch 6.

### Production — AntWeb queen vs. worker

Data pulled via the **GBIF mirror** of AntWeb (the native v3.1 API was down —
see [`data/download_gbif.py`](data/download_gbif.py)). **6,204 queens + 8,999
workers** (the full queen census on AntWeb + workers to a realistic ~1.5:1
imbalance), caste verbatim-confirmed (DwC `sex`), label-card images excluded.
EfficientNet-B2, 14 epochs, RTX 3050 Ti (4GB) with mixed-precision (AMP).

| Metric | Value |
|---|---|
| Test Accuracy | 0.925 |
| Test AUC | 0.974 |
| Queen P / R / F1 | 0.90 / 0.91 / 0.91 |
| Worker P / R / F1 | 0.94 / 0.93 / 0.94 |

The headline: **despite a 1.5:1 class imbalance, queen recall is 0.91** — the
`WeightedRandomSampler` keeps the model from neglecting the scarce class. Val
AUC climbed 0.78 (frozen) → 0.90 (epoch 6 unfreeze) → 0.974 (final).

> A balanced 2k/2k subset scored 0.88 acc / 0.94 AUC — the full imbalanced set
> beat it on every metric, confirming the imbalance handling works at scale.

**Data-quality note (the real engineering work):** GBIF's full-text caste
filter is contaminated — workers from nest series whose record text mentions
"queen" leak in (~half the naive queen set). The downloader confirms each
specimen against the *verbatim* Darwin Core `sex` field and rejects mismatches,
and filters out AntWeb label-card photos (shot type `l`) that aren't ants.

### Domain generalization — lab specimens → field photos

The AntWeb-trained model is excellent on lab specimens but **collapses on
field/mixed photos** (different backgrounds, lighting, pose) — a textbook
distribution shift. Diagnosed with a domain-sliced eval, then fixed by mixing
~5.4k field/mixed images into training and retraining ([`evaluate.py`](evaluate.py)
reports the per-domain breakdown).

| Test slice | AntWeb-only model | Combined model |
|---|---|---|
| **Field / mixed** (acc / AUC) | 0.595 / 0.656 | **0.822 / 0.909** |
| **AntWeb lab** (acc / AUC) | 0.925 / 0.974 | 0.922 / 0.971 |

Mixing distributions lifted field accuracy **0.60 → 0.82** (AUC 0.66 → 0.91)
with no meaningful regression on the lab domain — one model that handles both.
Combined held-out test (n=2,061): **acc 0.896, AUC 0.961, queen F1 0.884**.
