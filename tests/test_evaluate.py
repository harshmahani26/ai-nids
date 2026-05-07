"""Tests for the evaluation module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.data.loader import DatasetMeta
from src.evaluate import evaluate_classifier, upsert_result


@pytest.fixture
def fake_meta() -> DatasetMeta:
    return DatasetMeta(
        name="fake",
        feature_cols=["a", "b"],
        categorical_cols=[],
        numeric_cols=["a", "b"],
        labels=["normal", "attack"],
        normal_index=0,
    )


def test_evaluate_returns_complete_result(fake_meta, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.evaluate.CONFIG", _patched_config(tmp_path))
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(100, 2))
    y_train = rng.integers(0, 2, size=100)
    model = DummyClassifier(strategy="stratified", random_state=0).fit(X_train, y_train)

    X_test = rng.normal(size=(50, 2))
    y_test_multi = rng.integers(0, 2, size=50).astype(np.int64)
    y_test_binary = (y_test_multi != fake_meta.normal_index).astype(np.int64)

    res = evaluate_classifier(
        name="dummy",
        model=model,
        X_test=X_test,
        y_test_multi=y_test_multi,
        y_test_binary=y_test_binary,
        meta=fake_meta,
        train_time_s=0.5,
    )
    assert 0.0 <= res.binary_accuracy <= 1.0
    assert 0.0 <= res.multi_accuracy <= 1.0
    assert res.train_time_s == 0.5
    assert res.dataset == "fake"
    assert res.model == "dummy"


def test_upsert_replaces_existing(fake_meta, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.evaluate.CONFIG", _patched_config(tmp_path))
    from src.evaluate import ModelResult

    r = ModelResult(
        dataset="fake",
        model="x",
        binary_accuracy=0.9,
        multi_accuracy=0.8,
        macro_f1=0.7,
        weighted_f1=0.85,
        macro_precision=0.7,
        macro_recall=0.7,
        roc_auc=0.95,
        pr_auc=0.92,
        inference_ms_per_record=0.1,
        train_time_s=1.0,
        model_size_mb=2.0,
        per_class={},
    )
    upsert_result(r)
    upsert_result(r.__class__(**{**r.__dict__, "binary_accuracy": 0.95}))

    store = json.loads((tmp_path / "results" / "metrics.json").read_text())
    assert store["fake"]["x"]["binary_accuracy"] == 0.95


def _patched_config(tmp: Path):
    """Replace CONFIG.paths with tmp paths for the test."""
    from src.config import Config, DataConfig, Paths, TrainConfig

    p = Paths(
        root=tmp,
        data_raw=tmp / "data" / "raw",
        data_processed=tmp / "data" / "processed",
        results=tmp / "results",
        confusion_matrices=tmp / "results" / "confusion_matrices",
        shap=tmp / "results" / "shap",
        lime=tmp / "results" / "lime_examples",
        checkpoints=tmp / "results" / "checkpoints",
        plots=tmp / "results" / "plots",
        metrics_json=tmp / "results" / "metrics.json",
    )
    p.ensure()
    return Config(paths=p, train=TrainConfig(), data=DataConfig())
