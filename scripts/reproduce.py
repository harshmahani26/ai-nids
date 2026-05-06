"""One-command reproduction script.

Downloads the datasets, runs every tier on both NSL-KDD and UNSW-NB15, and
executes the dashboard notebook so its cell outputs end up in the file.
Intended for fresh clones::

    python scripts/reproduce.py

This will take a long time - allocate roughly 30-60 minutes on a modern
laptop.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import setup_logging  # noqa: E402
from src.data.download import download_all  # noqa: E402

log = logging.getLogger(__name__)


def run(cmd: list[str]) -> None:
    log.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def main() -> None:
    setup_logging()
    log.info("Downloading datasets")
    download_all()

    py = sys.executable
    for dataset in ("nslkdd", "unsw"):
        for tier in ("classical", "boosting", "deep", "hybrid", "stacking"):
            run([py, "-m", "src.train", "--dataset", dataset, "--tier", tier])

    nb = ROOT / "notebooks" / "nids_dashboard.ipynb"
    if nb.exists():
        run(
            [
                py,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=3600",
                str(nb),
            ]
        )

    log.info("done")


if __name__ == "__main__":
    main()
