"""
Download caste-labeled ant images from AntWeb API v3.
Pulls queens and workers, balances the dataset, and organises into
  data/raw/{queen,worker}/
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

ANTWEB_API = "https://antweb.org/v3.1"
CASTES = {"queen": "queen", "worker": "worker"}
DEFAULT_RAW_DIR = Path(__file__).parent / "raw"
API_SLEEP = 0.3       # between paginated API calls (single-threaded)
DEFAULT_THREADS = 16  # parallel image downloads


def fetch_specimens(
    caste: str,
    limit: int = 500,
    offset: int = 0,
) -> list[dict]:
    """Return a page of specimen records for the given caste."""
    params = {
        "caste": caste,
        "limit": limit,
        "offset": offset,
        "hasImage": "true",
    }
    resp = requests.get(f"{ANTWEB_API}/specimens", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("specimens", [])


def collect_image_urls(
    caste: str,
    target_count: int,
) -> list[tuple[str, str]]:
    """
    Walk the paginated AntWeb API and collect (url, filename) pairs
    until we have `target_count` images for `caste`.
    """
    results: list[tuple[str, str]] = []
    offset = 0
    page_size = 500
    pbar = tqdm(total=target_count, desc=f"Scanning {caste} records")

    while len(results) < target_count:
        specimens = fetch_specimens(caste, limit=page_size, offset=offset)
        if not specimens:
            logger.warning(f"No more {caste} specimens at offset {offset}")
            break

        for spec in specimens:
            if len(results) >= target_count:
                break
            images = spec.get("images", {})
            # Prefer profile view; fall back to head, then dorsal
            for view in ("profile", "head", "dorsal"):
                url = images.get(view, {}).get("high_resolution")
                if url:
                    spec_code = spec.get("specimen_code", f"spec_{offset}")
                    fname = f"{spec_code}_{view}.jpg".replace("/", "_")
                    results.append((url, fname))
                    pbar.update(1)
                    break

        offset += page_size
        time.sleep(API_SLEEP)

    pbar.close()
    return results


def _download_one(
    url: str,
    out_path: Path,
    skip_existing: bool,
    pbar: tqdm,
    counters: dict,
    lock: threading.Lock,
) -> None:
    if skip_existing and out_path.exists():
        with lock:
            counters["skip"] += 1
        pbar.update(1)
        return
    try:
        resp = requests.get(url, timeout=20, stream=True)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        with lock:
            counters["ok"] += 1
    except Exception as exc:
        logger.warning(f"Failed {url}: {exc}")
        with lock:
            counters["fail"] += 1
    finally:
        pbar.update(1)


def download_images(
    url_list: list[tuple[str, str]],
    dest_dir: Path,
    skip_existing: bool = True,
    threads: int = DEFAULT_THREADS,
) -> tuple[int, int]:
    """Download images to dest_dir in parallel. Returns (success_count, skip_count)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    counters = {"ok": 0, "skip": 0, "fail": 0}
    lock = threading.Lock()

    with tqdm(total=len(url_list), desc=f"Downloading → {dest_dir.name}") as pbar:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = [
                pool.submit(
                    _download_one,
                    url,
                    dest_dir / fname,
                    skip_existing,
                    pbar,
                    counters,
                    lock,
                )
                for url, fname in url_list
            ]
            for f in as_completed(futures):
                f.result()  # re-raise unexpected exceptions

    if counters["fail"]:
        logger.warning(f"{counters['fail']} downloads failed")
    return counters["ok"], counters["skip"]


def build_manifest(raw_dir: Path) -> None:
    """Write data/manifest.json with per-image metadata for reproducibility."""
    manifest = {}
    for caste in CASTES:
        caste_dir = raw_dir / caste
        if not caste_dir.exists():
            continue
        files = sorted(caste_dir.glob("*.jpg"))
        manifest[caste] = [str(f.relative_to(raw_dir.parent)) for f in files]
        logger.info(f"{caste}: {len(files)} images")

    out = raw_dir.parent / "manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AntWeb caste images")
    parser.add_argument(
        "--queens", type=int, default=2000, help="Target queen image count"
    )
    parser.add_argument(
        "--workers", type=int, default=2000, help="Target worker image count"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Output root"
    )
    parser.add_argument(
        "--no-skip", action="store_true", help="Re-download existing files"
    )
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_THREADS,
        help="Parallel download threads (default: 16)"
    )
    args = parser.parse_args()

    logger.info(
        f"Targeting {args.queens} queens, {args.workers} workers → {args.raw_dir} "
        f"({args.threads} threads)"
    )

    targets = {"queen": args.queens, "worker": args.workers}
    for caste, count in targets.items():
        logger.info(f"--- {caste.upper()} ---")
        urls = collect_image_urls(caste, count)
        ok, skipped = download_images(
            urls, args.raw_dir / caste,
            skip_existing=not args.no_skip,
            threads=args.threads,
        )
        logger.info(f"{caste}: {ok} downloaded, {skipped} skipped")

    build_manifest(args.raw_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
