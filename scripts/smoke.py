"""Quick post-install smoke check.

Verifies that imports work, datasets are available, and a tiny end-to-end
training run on synthetic data completes without error. Useful when triaging
a fresh clone.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import setup_logging  # noqa: E402

log = setup_logging(logging.INFO)


def check_imports() -> None:
    import lightgbm  # noqa: F401
    import shap  # noqa: F401
    import sklearn  # noqa: F401
    import torch  # noqa: F401
    import xgboost  # noqa: F401

    log.info("imports OK | torch.cuda=%s", torch.cuda.is_available())


def check_datasets() -> None:
    from src.config import CONFIG

    expected = [
        "KDDTrain+.txt",
        "KDDTest+.txt",
        "UNSW_NB15_training-set.csv",
        "UNSW_NB15_testing-set.csv",
    ]
    missing = [n for n in expected if not (CONFIG.paths.data_raw / n).exists()]
    if missing:
        log.warning("missing data files: %s. Run python -m src.data.download", missing)
    else:
        log.info("datasets present: %s", expected)


def check_tiny_training() -> None:
    import numpy as np
    from sklearn.datasets import make_classification

    from src.data.loader import DatasetMeta
    from src.data.preprocess import PreprocessedSplit

    X, y = make_classification(
        n_samples=400, n_features=10, n_classes=3, n_informative=6, random_state=0
    )
    Xtr, Xte = X[:300], X[300:]
    ytr, yte = y[:300], y[300:]
    meta = DatasetMeta(
        name="smoke",
        feature_cols=[f"f{i}" for i in range(10)],
        categorical_cols=[],
        numeric_cols=[f"f{i}" for i in range(10)],
        labels=["normal", "a", "b"],
        normal_index=0,
    )
    split = PreprocessedSplit(
        X_train=Xtr.astype(np.float64),
        y_train_multi=ytr.astype(np.int64),
        y_train_binary=(ytr != 0).astype(np.int64),
        X_test=Xte.astype(np.float64),
        y_test_multi=yte.astype(np.int64),
        y_test_binary=(yte != 0).astype(np.int64),
        feature_names=meta.feature_cols,
        meta=meta,
    )

    from src.models.deep import train_cnn

    cnn = train_cnn(split)
    preds = cnn.predict(split.X_test)
    assert preds.shape == (len(split.X_test),)
    log.info("tiny CNN training OK | accuracy=%.3f", float((preds == split.y_test_multi).mean()))


def main() -> None:
    check_imports()
    check_datasets()
    check_tiny_training()
    log.info("all checks passed")


if __name__ == "__main__":
    main()
