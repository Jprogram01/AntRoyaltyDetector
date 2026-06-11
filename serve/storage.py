"""
Persistence for predictions + user feedback on Hugging Face Spaces.

Spaces have an ephemeral filesystem (wiped on restart/rebuild), so we can't just
append to a local file. Instead we use `huggingface_hub.CommitScheduler`, which
buffers writes to a local folder and periodically commits them to a HF **Dataset**
repo — persistent, free, and exactly where training data should live.

Two datasets (configurable via env):
  - PREDICTIONS_DATASET  metadata only (no images) — for monitoring / drift
  - FEEDBACK_DATASET     image + corrected label — the retraining corpus

If no HF_TOKEN is set (e.g. local dev), everything degrades to no-ops so the
app still runs without persistence.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

HF_TOKEN = os.getenv("HF_TOKEN")
PREDICTIONS_DATASET = os.getenv("PREDICTIONS_DATASET", "Jprogram01/ant-predictions")
FEEDBACK_DATASET = os.getenv("FEEDBACK_DATASET", "Jprogram01/ant-feedback")
COMMIT_EVERY_MIN = int(os.getenv("COMMIT_EVERY_MIN", "5"))

# Unique per-process suffix. Spaces have an ephemeral /tmp that's wiped on every
# restart; a FIXED metadata filename would be recreated empty each boot and the
# CommitScheduler would overwrite the remote copy, losing all prior labels. A
# per-session filename is append-only across restarts — files accumulate, nothing
# is clobbered. (Read the dataset by globbing data/metadata-*.jsonl.)
_SESSION = uuid.uuid4().hex[:12]
_PRED_DIR = Path("/tmp/ant_pred_log")
_FB_DIR = Path("/tmp/ant_feedback")
_PRED_FILE = _PRED_DIR / f"predictions-{_SESSION}.jsonl"
_FB_FILE = _FB_DIR / f"metadata-{_SESSION}.jsonl"

_pred_scheduler = None
_fb_scheduler = None
_pred_lock = threading.Lock()
_fb_lock = threading.Lock()
_enabled = False


def init() -> bool:
    """Start the commit schedulers. Returns True if persistence is active."""
    global _pred_scheduler, _fb_scheduler, _enabled
    if not HF_TOKEN:
        logger.warning("No HF_TOKEN — prediction/feedback persistence disabled")
        return False
    try:
        from huggingface_hub import CommitScheduler

        _PRED_DIR.mkdir(parents=True, exist_ok=True)
        (_FB_DIR / "images").mkdir(parents=True, exist_ok=True)

        _pred_scheduler = CommitScheduler(
            repo_id=PREDICTIONS_DATASET,
            repo_type="dataset",
            folder_path=_PRED_DIR,
            path_in_repo="data",
            every=COMMIT_EVERY_MIN,
            token=HF_TOKEN,
            private=True,
        )
        _fb_scheduler = CommitScheduler(
            repo_id=FEEDBACK_DATASET,
            repo_type="dataset",
            folder_path=_FB_DIR,
            path_in_repo="data",
            every=COMMIT_EVERY_MIN,
            token=HF_TOKEN,
            private=True,
        )
        _enabled = True
        logger.info(
            f"Persistence on → predictions:{PREDICTIONS_DATASET} "
            f"feedback:{FEEDBACK_DATASET} (commit every {COMMIT_EVERY_MIN}m)"
        )
    except Exception as exc:
        logger.error(f"Could not start persistence: {exc}")
        _enabled = False
    return _enabled


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_prediction(meta: dict) -> None:
    """Append a metadata-only row (NO image) for monitoring."""
    if not _enabled:
        return
    row = {"ts": _now(), **meta}
    try:
        with _pred_lock, _pred_scheduler.lock:
            with open(_PRED_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    except Exception as exc:
        logger.warning(f"log_prediction failed: {exc}")


def save_feedback(image_bytes: bytes, meta: dict) -> str:
    """
    Save the image + corrected label to the feedback dataset (retraining corpus).
    Only call this when the user has explicitly rated (consent). Returns the id.
    """
    fb_id = uuid.uuid4().hex[:12]
    if not _enabled:
        logger.info(f"feedback received (persistence off): {meta}")
        return fb_id
    try:
        img_path = _FB_DIR / "images" / f"{fb_id}.jpg"
        with _fb_lock, _fb_scheduler.lock:
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            row = {"ts": _now(), "id": fb_id, "image": f"images/{fb_id}.jpg", **meta}
            with open(_FB_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        logger.info(f"feedback saved: {fb_id} {meta}")
    except Exception as exc:
        logger.warning(f"save_feedback failed: {exc}")
    return fb_id
