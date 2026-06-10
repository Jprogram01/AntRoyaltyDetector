"""
Download the Hymenoptera ants-vs-bees dataset (PyTorch transfer learning classic).
Used as a baseline to validate the training loop before AntWeb data is available.

The dataset is hosted by PyTorch and contains ~240 train / ~150 val images
split into ants/ and bees/ subdirectories.

After download we re-map ants → worker so it drops straight into AntCasteDataset
by symlinking data/raw/worker → hymenoptera/train/ants (and val equivalents).
"""

import shutil
import zipfile
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

HYMENOPTERA_URL = (
    "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
)
DEFAULT_DEST = Path(__file__).parent / "hymenoptera_data"


def download_zip(url: str, dest_zip: Path) -> None:
    if dest_zip.exists():
        logger.info(f"Already downloaded: {dest_zip}")
        return
    logger.info(f"Downloading {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_zip, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="hymenoptera_data.zip"
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            pbar.update(len(chunk))


def extract(dest_zip: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        logger.info(f"Already extracted: {dest_dir}")
        return
    logger.info(f"Extracting → {dest_dir}")
    with zipfile.ZipFile(dest_zip) as zf:
        zf.extractall(dest_dir.parent)
    logger.info("Extraction complete.")


def link_into_raw(hymenoptera_dir: Path, raw_dir: Path) -> None:
    """
    Copy ants images into data/raw/worker and bees into data/raw/bee
    so AntCasteDataset can be pointed at data/raw for smoke-test training.
    Only ants (workers) are used for the queen/worker task — bees give us
    a sanity-check negative class during baseline validation.
    """
    mapping = {
        "ants": "worker",
        "bees": "bee",   # not a caste label — baseline sanity only
    }
    for split in ("train", "val"):
        for src_name, dst_name in mapping.items():
            src = hymenoptera_dir / split / src_name
            dst = raw_dir / dst_name
            if not src.exists():
                logger.warning(f"Source not found: {src}")
                continue
            dst.mkdir(parents=True, exist_ok=True)
            copied = 0
            for img in src.glob("*.jpg"):
                target = dst / f"hymenoptera_{split}_{img.name}"
                if not target.exists():
                    shutil.copy2(img, target)
                    copied += 1
            logger.info(f"{split}/{src_name} → raw/{dst_name}: {copied} copied")


def main() -> None:
    dest_zip = DEFAULT_DEST.parent / "hymenoptera_data.zip"
    download_zip(HYMENOPTERA_URL, dest_zip)
    extract(dest_zip, DEFAULT_DEST)

    raw_dir = Path(__file__).parent / "raw"
    link_into_raw(DEFAULT_DEST, raw_dir)
    logger.info("Done. Run: python train.py --raw-dir data/raw --baseline")


if __name__ == "__main__":
    main()
