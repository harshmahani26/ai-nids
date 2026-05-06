"""Cross-platform dataset downloader using urllib.

Downloads NSL-KDD and UNSW-NB15 from public mirrors to ``data/raw/``.
Skips files already present so re-runs are cheap.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from pathlib import Path

from src.config import CONFIG

log = logging.getLogger(__name__)


NSLKDD_FILES = ("KDDTrain+.txt", "KDDTest+.txt")

# UNSW-NB15: HuggingFace mirror keeps the original string-typed proto/service/state
# columns and the multi-class attack_cat label that CatBoost needs natively.
UNSW_TRAIN_URL = "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/train.csv"
UNSW_TEST_URL = "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/test.csv"


def _download(url: str, dest: Path, *, force: bool = False) -> Path:
    if dest.exists() and not force:
        log.info("skip (exists): %s", dest.name)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("download: %s -> %s", url, dest.name)

    req = urllib.request.Request(url, headers={"User-Agent": "ai-nids/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as f:
            while chunk := resp.read(64 * 1024):
                f.write(chunk)
    except urllib.error.URLError as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"failed to download {url}: {e}") from e
    return dest


def download_nslkdd(*, force: bool = False) -> list[Path]:
    """Fetch NSL-KDD KDDTrain+ and KDDTest+."""
    paths = []
    for fname in NSLKDD_FILES:
        url = f"{CONFIG.data.nslkdd_url_base}/{fname}"
        paths.append(_download(url, CONFIG.paths.data_raw / fname, force=force))
    return paths


def download_unsw(*, force: bool = False) -> list[Path]:
    """Fetch UNSW-NB15 official train/test CSVs (HuggingFace mirror)."""
    train_csv = CONFIG.paths.data_raw / CONFIG.data.unsw_train
    test_csv = CONFIG.paths.data_raw / CONFIG.data.unsw_test
    _download(UNSW_TRAIN_URL, train_csv, force=force)
    _download(UNSW_TEST_URL, test_csv, force=force)
    return [train_csv, test_csv]


def download_all(*, force: bool = False) -> dict[str, list[Path]]:
    """Download both datasets."""
    return {
        "nslkdd": download_nslkdd(force=force),
        "unsw": download_unsw(force=force),
    }


if __name__ == "__main__":
    from src.config import setup_logging

    setup_logging()
    download_all()
