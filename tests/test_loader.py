"""Tests for dataset loaders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import CONFIG
from src.data.loader import (
    NSLKDD_LABELS,
    UNSW_LABELS,
    load_dataset,
    load_nslkdd,
    load_unsw,
)


def _data_present(filenames: list[str]) -> bool:
    return all((CONFIG.paths.data_raw / fn).exists() for fn in filenames)


nslkdd_required = pytest.mark.skipif(
    not _data_present(["KDDTrain+.txt", "KDDTest+.txt"]),
    reason="NSL-KDD raw files missing; run python -m src.data.download",
)
unsw_required = pytest.mark.skipif(
    not _data_present(["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]),
    reason="UNSW-NB15 raw files missing; run python -m src.data.download",
)


@nslkdd_required
def test_nslkdd_shapes() -> None:
    Xtr, ytr, Xte, yte, meta = load_nslkdd()
    assert isinstance(Xtr, pd.DataFrame)
    assert Xtr.shape[1] == 41
    assert Xte.shape[1] == 41
    assert len(ytr) == len(Xtr)
    assert len(yte) == len(Xte)
    assert meta.labels == list(NSLKDD_LABELS)
    assert meta.normal_index == 0


@nslkdd_required
def test_nslkdd_labels_in_range() -> None:
    _, ytr, _, yte, meta = load_nslkdd()
    assert ytr.min() >= 0 and ytr.max() < meta.num_classes
    assert yte.min() >= 0 and yte.max() < meta.num_classes


@unsw_required
def test_unsw_shapes_and_labels() -> None:
    Xtr, ytr, Xte, yte, meta = load_unsw()
    assert isinstance(Xtr, pd.DataFrame)
    assert "attack_cat" not in Xtr.columns
    assert "label" not in Xtr.columns
    assert "id" not in Xtr.columns
    assert meta.labels == list(UNSW_LABELS)
    assert meta.normal_index == 0
    assert ytr.dtype == np.int64


@nslkdd_required
def test_dispatch_helper() -> None:
    a = load_dataset("nslkdd")
    b = load_nslkdd()
    assert a[0].shape == b[0].shape
    with pytest.raises(ValueError):
        load_dataset("nope")
