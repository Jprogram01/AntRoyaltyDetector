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

Binary image classifier that distinguishes 1. Ants from non-ants and 2. Ant queens from workers using transfer learning on AntWeb & iNaturalist specimen photographs. Built as a production-grade ML Engineering portfolio project.

> **Live demo:** this repo doubles as a [Hugging Face Space](https://huggingface.co) — the YAML header above configures it. Open the Space URL to upload an ant photo and get a queen/worker prediction in the browser.

## Why?

Within the antkeeping world there is a common problem for new antkeepers: Correctly identifying if an ant is a queen ant. While there are physical traits that allow for more experienced antkeepers to quickly identify queen ants those can be hard to spot for beginners. The goal of this tool is to help newcomers to the antkeeping world in identifying queens.

## Problems

There were no datasets that classified ant caste, as well as a data imbalance problems in the available ant image datasets. This then meant a dataset had to be made manually by me. The data imbalance problems can cause problems in model training, but this can be mitigated during training by tactics such as class weighting.

## AI Disclaimer 

The system design, data selection/labeling, class imbalance decisions, deployment were all decided and handled by me, but Claude Code wrote the code and implemented the solutions. Much of this README was also written by Claude Code. I wanted to develop an ML project from idea to deployment while also familiarizing myself with AI tools. 



## Architecture

```
AntWeb API
    │
    ▼
data/download_antweb.py          # paginated API pull, ~2k queens / ~2k workers
    │
    ▼
data/dataset.py                  # WeightedRandomSampler + BCEWithLogitsLoss(pos_weight)
    │
    ▼
model/classifier.py              # EfficientNet-B2 backbone (timm) + 2-layer head
    │
    ▼
train.py                         # 2-phase training: frozen backbone → full fine-tune
    │
    ▼
checkpoints/best.pt              # saved by best val AUC
    │
    ▼
serve/app.py                     # FastAPI: POST /predict, GET /health, GET /metrics
    │
    ▼
Docker + Prometheus               # containerized, scraped every 15s
```

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

**despite a 1.5:1 class imbalance, queen recall is 0.91** — the
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
