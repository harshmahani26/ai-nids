"""Smoke tests: each model trains on a tiny subsample without crashing."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.data.loader import DatasetMeta
from src.data.preprocess import PreprocessedSplit


@pytest.fixture
def tiny_split() -> PreprocessedSplit:
    X, y = make_classification(
        n_samples=600,
        n_features=12,
        n_informative=8,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=0,
    )
    Xtr, Xte = X[:480], X[480:]
    ytr, yte = y[:480], y[480:]
    meta = DatasetMeta(
        name="toy",
        feature_cols=[f"f{i}" for i in range(12)],
        categorical_cols=[],
        numeric_cols=[f"f{i}" for i in range(12)],
        labels=["normal", "attack_a", "attack_b"],
        normal_index=0,
    )
    return PreprocessedSplit(
        X_train=Xtr.astype(np.float64),
        y_train_multi=ytr.astype(np.int64),
        y_train_binary=(ytr != meta.normal_index).astype(np.int64),
        X_test=Xte.astype(np.float64),
        y_test_multi=yte.astype(np.int64),
        y_test_binary=(yte != meta.normal_index).astype(np.int64),
        feature_names=meta.feature_cols,
        meta=meta,
    )


def test_xgboost_trains(tiny_split) -> None:
    import xgboost as xgb

    clf = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=4,
        tree_method="hist",
        objective="multi:softprob",
        num_class=3,
        random_state=0,
        n_jobs=1,
        verbosity=0,
    )
    clf.fit(tiny_split.X_train, tiny_split.y_train_multi)
    preds = clf.predict(tiny_split.X_test)
    assert preds.shape == (len(tiny_split.X_test),)


def test_lightgbm_trains(tiny_split) -> None:
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(
        n_estimators=20,
        num_leaves=8,
        objective="multiclass",
        num_class=3,
        n_jobs=1,
        verbose=-1,
        random_state=0,
    )
    clf.fit(tiny_split.X_train, tiny_split.y_train_multi)
    preds = clf.predict(tiny_split.X_test)
    assert preds.shape == (len(tiny_split.X_test),)


def test_random_forest_trains(tiny_split) -> None:
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=20, max_depth=8, n_jobs=1, random_state=0)
    clf.fit(tiny_split.X_train, tiny_split.y_train_multi)
    assert clf.predict(tiny_split.X_test).shape == (len(tiny_split.X_test),)


def test_cnn_smoke(tiny_split) -> None:
    """Run a 1-epoch training pass to verify forward + backward shape correctness."""
    import torch

    from src.models.deep import CNN1D

    device = torch.device("cpu")
    model = CNN1D(n_features=tiny_split.X_train.shape[1], n_classes=3).to(device)
    x = torch.as_tensor(tiny_split.X_train[:32], dtype=torch.float32).to(device)
    y = torch.as_tensor(tiny_split.y_train_multi[:32], dtype=torch.long).to(device)
    out = model(x)
    assert out.shape == (32, 3)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
