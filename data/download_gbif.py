"""
Download caste-labeled ant images via the GBIF mirror of AntWeb.

Why GBIF instead of the AntWeb API directly:
  - AntWeb's v3.1 API has been returning empty bodies (backend down).
  - GBIF mirrors the full AntWeb dataset (California Academy of Sciences,
    datasetKey 13b70480-bd69-11dd-b15f-b8a03c50a862) with a stable, public,
    Cloudflare-free occurrence/search API.
  - AntWeb stores caste in the Darwin Core `sex` term, so a queen specimen
    comes back as sex="queen" and a worker as sex="worker". The GBIF full-text
    `q=` parameter filters on this directly.

The image binaries still live on AntWeb's CDN (www.antweb.org/images/...),
which IS behind Cloudflare — so image downloads need a `cf_clearance` cookie
copied from a browser session. Pass it via --cf-clearance or the
ANTWEB_CF_CLEARANCE env var. (Metadata needs no cookie; only images do.)

Usage:
    python -m data.download_gbif --queens 1500 --workers 1500 \
        --cf-clearance "<cookie value from browser>"
"""

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

GBIF_SEARCH = "https://api.gbif.org/v1/occurrence/search"
ANTWEB_DATASET = "13b70480-bd69-11dd-b15f-b8a03c50a862"
DEFAULT_RAW_DIR = Path(__file__).parent / "raw"
DEFAULT_THREADS = 16
PAGE_SIZE = 300  # GBIF max is 300 per page

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)


GBIF_VERBATIM = "https://api.gbif.org/v1/occurrence/{key}/verbatim"
DWC_SEX = "http://rs.tdwg.org/dwc/terms/sex"

# AntWeb image filenames carry a shot-type token: <code>_<t>_<n>_high.jpg
#   p = profile, d = dorsal, h = head, l = LABEL (a photo of the label card,
#   not the ant). We only ever want real specimen views, preferred p > d > h,
#   and must never download labels.
SHOT_PREFERENCE = ("p", "d", "h")
_SHOT_RE = re.compile(r"_([pdhl])_\d")


def _pick_image_url(media: list[dict]) -> tuple[str, str] | None:
    """
    From a record's media list, return (url, shot_type) for the best real
    specimen view, or None if the record only has label/other images.
    """
    by_shot: dict[str, str] = {}
    for m in media:
        url = m.get("identifier")
        if not url:
            continue
        match = _SHOT_RE.search(url)
        shot = match.group(1) if match else None
        if shot in SHOT_PREFERENCE and shot not in by_shot:
            by_shot[shot] = url
    for shot in SHOT_PREFERENCE:
        if shot in by_shot:
            return by_shot[shot], shot
    return None


def _gbif_search_page(caste: str, offset: int, retries: int = 5) -> dict:
    """GET one page of GBIF occurrence search, with exponential backoff so a
    transient network blip or GBIF throttle doesn't abort a long collection."""
    params = {
        "datasetKey": ANTWEB_DATASET,
        "mediaType": "StillImage",
        "q": caste,
        "limit": PAGE_SIZE,
        "offset": offset,
    }
    for attempt in range(retries):
        try:
            # (connect, read) timeout — read timeout caps slow-trickle stalls
            # so a half-closed connection fails fast instead of hanging minutes.
            resp = requests.get(GBIF_SEARCH, params=params, timeout=(10, 20))
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            wait = min(2 ** attempt, 30)
            logger.warning(
                f"GBIF page offset={offset} attempt {attempt + 1}/{retries} "
                f"failed ({exc}); retrying in {wait}s"
            )
            time.sleep(wait)
    raise RuntimeError(f"GBIF search failed after {retries} retries at offset {offset}")


def _verbatim_sex(key: int) -> str:
    """
    Fetch the TRUE caste for a GBIF occurrence. AntWeb maps caste into the
    Darwin Core `sex` term; GBIF's interpreted `sex` drops it (not a valid DwC
    enum), so the only reliable source is the verbatim record.
    """
    try:
        v = requests.get(GBIF_VERBATIM.format(key=key), timeout=20).json()
        return (v.get(DWC_SEX) or "").strip().lower()
    except Exception:
        return ""


def collect_records(
    caste: str,
    target_count: int,
    verify_verbatim: bool = True,
    exclude_codes: set[str] | None = None,
) -> list[dict]:
    """
    Page through GBIF occurrence search for AntWeb specimens of `caste`,
    returning up to `target_count` image-bearing records of that caste.

    Two modes:
      - verify_verbatim=True (default): confirm each candidate against the
        verbatim DwC `sex` field. Accurate but slow; use for the queen class.
      - verify_verbatim=False + exclude_codes=<full queen census>: skip the
        per-record calls (fast, robust at scale) and instead drop any specimen
        whose code is a known queen. Used for the majority worker class.

    Critical: GBIF full-text `q=<caste>` is only a candidate filter — it also
    matches workers whose record text mentions "queen" (e.g. nest series), so
    raw `q=` results are heavily cross-contaminated (~half, deeper in the
    ranking). We confirm each candidate against the verbatim DwC `sex` field
    and drop any mismatch. Each returned dict: specimen_code, scientific_name,
    shot_type, url.
    """
    results: list[dict] = []
    seen_codes: set[str] = set()
    offset = 0
    dropped = 0
    pbar = tqdm(total=target_count, desc=f"GBIF confirm {caste}")

    while len(results) < target_count:
        data = _gbif_search_page(caste, offset)
        page = data.get("results", [])
        if not page:
            logger.warning(f"GBIF exhausted for {caste} at offset {offset}")
            break

        # Build candidates from this page — only those with a real specimen
        # view (profile/dorsal/head). Records that are label-only are skipped.
        candidates = []
        for rec in page:
            picked = _pick_image_url(rec.get("media", []))
            if not picked:
                continue
            url, shot = picked
            candidates.append((rec, url, shot))

        if verify_verbatim:
            # Verbatim-confirm caste concurrently (strict contamination fix).
            # Accurate but slow — GBIF throttles after thousands of calls, so
            # this is reserved for the scarce/contamination-prone queen class.
            with ThreadPoolExecutor(max_workers=16) as pool:
                sexes = list(pool.map(lambda c: _verbatim_sex(c[0]["key"]), candidates))
        else:
            # Fast path (majority class): skip per-record verbatim. Cross-caste
            # contamination is removed via `exclude_codes` — pass the complete
            # queen census so any queen leaking into q=worker is dropped by code.
            sexes = [caste] * len(candidates)

        for (rec, url, shot), sex in zip(candidates, sexes):
            if len(results) >= target_count:
                break
            if sex != caste:          # <-- strict: reject cross-contamination
                dropped += 1
                continue
            code = rec.get("catalogNumber") or rec.get("occurrenceID", "unknown")
            code = str(code).replace("/", "_").replace(":", "_")
            if code in seen_codes:
                continue
            if exclude_codes and code in exclude_codes:  # known other-caste specimen
                dropped += 1
                continue
            seen_codes.add(code)
            results.append(
                {
                    "specimen_code": code,
                    "scientific_name": rec.get("scientificName", "?"),
                    "shot_type": shot,
                    "url": url,
                }
            )
            pbar.update(1)

        offset += PAGE_SIZE
        if offset >= data.get("count", 0):
            break

    pbar.close()
    logger.info(
        f"{caste}: kept {len(results)}, dropped {dropped} cross-contaminated "
        f"(wrong verbatim caste)"
    )
    return results


def _download_one(rec, dest_dir, session, skip_existing, pbar, counters, lock):
    fname = f"{rec['specimen_code']}_{rec['shot_type']}.jpg"
    out_path = dest_dir / fname
    if skip_existing and out_path.exists():
        with lock:
            counters["skip"] += 1
        pbar.update(1)
        return
    try:
        r = session.get(rec["url"], timeout=30, stream=True)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if not ctype.startswith("image/"):
            # Cloudflare HTML challenge slips back as 200 — treat as failure
            raise ValueError(f"non-image response ({ctype})")
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        with lock:
            counters["ok"] += 1
    except Exception as exc:
        with lock:
            counters["fail"] += 1
        if counters["fail"] <= 5:
            logger.warning(f"Failed {rec['url']}: {exc}")
    finally:
        pbar.update(1)


def download_images(records, dest_dir, cf_clearance, skip_existing, threads):
    dest_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    if cf_clearance:
        session.cookies.set("cf_clearance", cf_clearance, domain=".antweb.org")

    counters = {"ok": 0, "skip": 0, "fail": 0}
    lock = threading.Lock()
    with tqdm(total=len(records), desc=f"Downloading → {dest_dir.name}") as pbar:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = [
                pool.submit(
                    _download_one, rec, dest_dir, session,
                    skip_existing, pbar, counters, lock,
                )
                for rec in records
            ]
            for f in as_completed(futures):
                f.result()

    if counters["fail"]:
        logger.warning(
            f"{counters['fail']} downloads failed "
            f"(likely Cloudflare — refresh cf_clearance if all failed)"
        )
    return counters["ok"], counters["skip"]


def build_manifest(raw_dir: Path) -> None:
    manifest = {}
    for caste in ("queen", "worker"):
        d = raw_dir / caste
        if not d.exists():
            continue
        files = sorted(d.glob("*.jpg"))
        manifest[caste] = [str(f.relative_to(raw_dir.parent)) for f in files]
        logger.info(f"{caste}: {len(files)} images")
    out = raw_dir.parent / "manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest → {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Download AntWeb caste images via GBIF")
    p.add_argument("--queens", type=int, default=1500)
    p.add_argument("--workers", type=int, default=1500)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--no-skip", action="store_true")
    p.add_argument(
        "--workers-only", action="store_true",
        help="Skip queen collection; reuse the queen census already on disk",
    )
    p.add_argument(
        "--cf-clearance",
        default=os.getenv("ANTWEB_CF_CLEARANCE", ""),
        help="cf_clearance cookie from a browser (or set ANTWEB_CF_CLEARANCE)",
    )
    args = p.parse_args()

    if not args.cf_clearance:
        logger.warning(
            "No cf_clearance cookie — image downloads will likely 403. "
            "Pass --cf-clearance or set ANTWEB_CF_CLEARANCE."
        )

    # Queens first (strict verbatim — scarce, contamination-prone class).
    # The resulting queen census is then used to clean the worker pull.
    if args.workers_only:
        logger.info("--workers-only: skipping queen collection, reusing disk census")
        queen_records = []
    else:
        queen_records = collect_records("queen", args.queens, verify_verbatim=True)
        logger.info(f"Collected {len(queen_records)} queen image URLs from GBIF")
        ok, skipped = download_images(
            queen_records, args.raw_dir / "queen", args.cf_clearance,
            skip_existing=not args.no_skip, threads=args.threads,
        )
        logger.info(f"queen: {ok} downloaded, {skipped} skipped")

    # Build the full queen census (codes already on disk + just-collected) to
    # exclude from the worker pull — this removes queen contamination from
    # q=worker without slow per-record verbatim calls on the majority class.
    queen_codes = {p.stem.rsplit("_", 1)[0] for p in (args.raw_dir / "queen").glob("*.jpg")}
    queen_codes |= {r["specimen_code"] for r in queen_records}
    logger.info(f"Queen census for worker exclusion: {len(queen_codes)} codes")

    worker_records = collect_records(
        "worker", args.workers, verify_verbatim=False, exclude_codes=queen_codes
    )
    logger.info(f"Collected {len(worker_records)} worker image URLs from GBIF")
    ok, skipped = download_images(
        worker_records, args.raw_dir / "worker", args.cf_clearance,
        skip_existing=not args.no_skip, threads=args.threads,
    )
    logger.info(f"worker: {ok} downloaded, {skipped} skipped")

    build_manifest(args.raw_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
