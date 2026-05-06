"""Tests for preprocessing - especially the no-leakage invariant."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DatasetMeta
from src.data.preprocess import Preprocessor


@pytest.fixture
def toy_meta() -> DatasetMeta:
    return DatasetMeta(
        name="toy",
        feature_cols=["color", "length", "weight"],
        categorical_cols=["color"],
        numeric_cols=["length", "weight"],
        labels=["normal", "attack"],
        normal_index=0,
    )


@pytest.fixture
def train_test(toy_meta):
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "color": ["red", "blue", "red", "green"],
            "length": rng.normal(100, 10, size=4),
            "weight": rng.normal(50, 5, size=4),
        }
    )
    y_train = np.array([0, 1, 0, 1], dtype=np.int64)
    test = pd.DataFrame(
        {
            # 'yellow' is unseen at training time and must be handled
            "color": ["red", "yellow", "blue"],
            "length": rng.normal(100, 10, size=3),
            "weight": rng.normal(50, 5, size=3),
        }
    )
    y_test = np.array([0, 1, 0], dtype=np.int64)
    return train, y_train, test, y_test


def test_fit_transform_no_leakage(toy_meta, train_test) -> None:
    train_X, train_y, test_X, test_y = train_test
    pp = Preprocessor(toy_meta)
    Xtr, _, _ = pp.fit_transform(train_X, train_y)
    # After fit, mean ~0 and std ~1 over train (numeric features).
    np.testing.assert_allclose(Xtr[:, 1:].mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(Xtr[:, 1:].std(axis=0), 1, atol=1e-6)


def test_unseen_category_handled(toy_meta, train_test) -> None:
    train_X, train_y, test_X, test_y = train_test
    pp = Preprocessor(toy_meta)
    pp.fit_transform(train_X, train_y)
    Xte, _, _ = pp.transform(test_X, test_y)
    # The "yellow" row encodes to the sentinel (0) without raising.
    assert Xte.shape == (3, 3)
    assert np.isfinite(Xte).all()


def test_transform_requires_fit(toy_meta, train_test) -> None:
    _, _, test_X, test_y = train_test
    pp = Preprocessor(toy_meta)
    with pytest.raises(RuntimeError):
        pp.transform(test_X, test_y)


def test_binary_label_derivation(toy_meta, train_test) -> None:
    train_X, train_y, _, _ = train_test
    pp = Preprocessor(toy_meta)
    _, ymulti, ybin = pp.fit_transform(train_X, train_y)
    # binary == 1 wherever multi != normal_index
    assert (ybin == (ymulti != toy_meta.normal_index).astype(np.int64)).all()
