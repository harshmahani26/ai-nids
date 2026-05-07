"""Unified evaluation: metrics, plots, latency, model size.

Every tier writes into the same ``results/metrics.json`` keyed by
``(dataset, model_name)``. Re-runs are idempotent: a model entry is replaced
in place so partial runs accumulate cleanly.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import CONFIG
from src.data.loader import DatasetMeta

log = logging.getLogger(__name__)

# In-memory cache of P(attack) per (dataset, model) so a follow-up call from
# train.py can produce a calibration plot without re-running the evaluator.
_PROBA_CACHE: dict[tuple[str, str], np.ndarray] = {}
_BIN_LABELS_CACHE: dict[str, np.ndarray] = {}


class _ModelLike(Protocol):
    def predict(self, X) -> np.ndarray: ...


@dataclass
class ModelResult:
    """Metrics + diagnostics for a single (dataset, model) cell."""

    dataset: str
    model: str
    binary_accuracy: float
    multi_accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    roc_auc: float | None
    pr_auc: float | None
    inference_ms_per_record: float  # batched (1000 records in one predict call)
    train_time_s: float
    model_size_mb: float
    per_class: dict[str, Any]
    inference_ms_per_record_unbatched: float = 0.0  # per-record (one row at a time)
    extra: dict[str, Any] = field(default_factory=dict)


def _proba_attack(model, X: np.ndarray, normal_index: int) -> np.ndarray | None:
    """Return P(attack) using whatever proba/decision interface the model exposes."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return 1.0 - proba[:, normal_index]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Squash to [0,1] for AUC purposes.
        return 1.0 / (1.0 + np.exp(-scores))
    return None


def _measure_inference(model, X, n: int = 1000) -> tuple[float, float]:
    """Measure batched and unbatched inference latency, in ms per record.

    Parameters
    ----------
    model : sklearn-style estimator
        Must implement ``predict``.
    X : np.ndarray or pd.DataFrame
        Test features. The first ``min(n, len(X))`` rows are timed.
    n : int
        Sample size for the timing run.

    Returns
    -------
    (batched_ms_per_record, unbatched_ms_per_record)
        ``batched`` calls ``predict`` once on the whole sample. ``unbatched``
        calls ``predict`` once per row over a smaller subset (capped at 200) so
        the total wall time stays tractable for slow learners like SVM.
    """
    import pandas as pd

    n = min(n, len(X))
    sample = X[:n] if not isinstance(X, pd.DataFrame) else X.iloc[:n]
    # Warmup
    warm = sample[:32] if not isinstance(sample, pd.DataFrame) else sample.iloc[:32]
    model.predict(warm)

    # Batched
    t0 = time.perf_counter()
    model.predict(sample)
    batched_ms = (time.perf_counter() - t0) * 1000.0 / n

    # Unbatched: cap at 200 single-row predicts to bound wall time.
    m = min(200, n)
    one_row = (
        [sample.iloc[i : i + 1] for i in range(m)]
        if isinstance(sample, pd.DataFrame)
        else [sample[i : i + 1] for i in range(m)]
    )
    t0 = time.perf_counter()
    for r in one_row:
        model.predict(r)
    unbatched_ms = (time.perf_counter() - t0) * 1000.0 / m
    return batched_ms, unbatched_ms


def _model_size_mb(model: Any) -> float:
    """Estimate serialised model size in MB."""
    try:
        return len(pickle.dumps(model)) / (1024 * 1024)
    except Exception:
        return float("nan")


def evaluate_classifier(
    *,
    name: str,
    model,
    X_test: np.ndarray,
    y_test_multi: np.ndarray,
    y_test_binary: np.ndarray,
    meta: DatasetMeta,
    train_time_s: float,
    save_confusion: bool = True,
    extra: dict[str, Any] | None = None,
) -> ModelResult:
    """Compute every standard metric for one model on one dataset.

    Parameters
    ----------
    name : str
        Display name; used as the key in ``results/metrics.json``.
    model : sklearn-style estimator
        Must implement ``predict``. ``predict_proba`` / ``decision_function``
        are used opportunistically for ROC-AUC / PR-AUC.
    X_test : np.ndarray or pd.DataFrame
        Test features. CatBoost gets the raw DataFrame; every other model
        gets the encoded numpy matrix.
    y_test_multi, y_test_binary : np.ndarray
        Multi-class and 0/1 binary labels for the test set.
    meta : DatasetMeta
        Used for label names and ``normal_index``.
    train_time_s : float
        Wall time spent training this model.
    save_confusion : bool
        If True, save binary + multi-class confusion matrix PNGs.
    extra : dict | None
        Free-form per-model annotations (e.g. hybrid threshold).

    Returns
    -------
    ModelResult
        All metrics, latencies, model size, and per-class report.
    """
    y_pred_multi = model.predict(X_test)
    y_pred_binary = (y_pred_multi != meta.normal_index).astype(np.int64)

    multi_acc = accuracy_score(y_test_multi, y_pred_multi)
    binary_acc = accuracy_score(y_test_binary, y_pred_binary)
    macro_f1 = f1_score(y_test_multi, y_pred_multi, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test_multi, y_pred_multi, average="weighted", zero_division=0)
    macro_prec = precision_score(y_test_multi, y_pred_multi, average="macro", zero_division=0)
    macro_rec = recall_score(y_test_multi, y_pred_multi, average="macro", zero_division=0)

    proba_attack = _proba_attack(model, X_test, meta.normal_index)
    roc_auc = pr_auc = None
    if proba_attack is not None:
        try:
            roc_auc = float(roc_auc_score(y_test_binary, proba_attack))
            pr_auc = float(average_precision_score(y_test_binary, proba_attack))
        except Exception as e:  # pragma: no cover
            log.warning("AUC computation failed for %s: %s", name, e)
        # Cache so train.py can build a calibration plot for the best model.
        _PROBA_CACHE[(meta.name, name)] = np.asarray(proba_attack)
        _BIN_LABELS_CACHE[meta.name] = np.asarray(y_test_binary)

    per_class = classification_report(
        y_test_multi,
        y_pred_multi,
        labels=list(range(len(meta.labels))),
        target_names=meta.labels,
        output_dict=True,
        zero_division=0,
    )

    batched_ms, unbatched_ms = _measure_inference(model, X_test)
    size_mb = _model_size_mb(model)

    if save_confusion:
        _save_confusion_matrix(name, meta, y_test_multi, y_pred_multi, y_test_binary, y_pred_binary)

    result = ModelResult(
        dataset=meta.name,
        model=name,
        binary_accuracy=float(binary_acc),
        multi_accuracy=float(multi_acc),
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        macro_precision=float(macro_prec),
        macro_recall=float(macro_rec),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        inference_ms_per_record=float(batched_ms),
        inference_ms_per_record_unbatched=float(unbatched_ms),
        train_time_s=float(train_time_s),
        model_size_mb=float(size_mb),
        per_class=per_class,
        extra=extra or {},
    )
    log.info(
        "[%s/%s] bin=%.3f multi=%.3f macroF1=%.3f rocAUC=%s prAUC=%s "
        "lat_batched=%.3fms lat_single=%.3fms size=%.2fMB",
        meta.name,
        name,
        binary_acc,
        multi_acc,
        macro_f1,
        f"{roc_auc:.3f}" if roc_auc is not None else "n/a",
        f"{pr_auc:.3f}" if pr_auc is not None else "n/a",
        batched_ms,
        unbatched_ms,
        size_mb,
    )
    return result


def _save_confusion_matrix(
    name: str,
    meta: DatasetMeta,
    y_true_multi: np.ndarray,
    y_pred_multi: np.ndarray,
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
) -> None:
    CONFIG.paths.confusion_matrices.mkdir(parents=True, exist_ok=True)
    base = name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    # Multi-class
    cm = confusion_matrix(y_true_multi, y_pred_multi, labels=list(range(len(meta.labels))))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=meta.labels,
        yticklabels=meta.labels,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} - {meta.name} (multi-class)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(CONFIG.paths.confusion_matrices / f"{meta.name}_{base}_multi.png", dpi=120)
    plt.close(fig)

    # Binary
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} - {meta.name} (binary)")
    plt.tight_layout()
    fig.savefig(CONFIG.paths.confusion_matrices / f"{meta.name}_{base}_binary.png", dpi=120)
    plt.close(fig)


# ---------- Persistent metrics store --------------------------------------


def _load_store() -> dict:
    p = CONFIG.paths.metrics_json
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _save_store(store: dict) -> None:
    p = CONFIG.paths.metrics_json
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(store, indent=2))


def upsert_result(result: ModelResult) -> None:
    """Merge a result into ``results/metrics.json`` keyed by dataset + model."""
    store = _load_store()
    ds_bucket = store.setdefault(result.dataset, {})
    ds_bucket[result.model] = asdict(result)
    _save_store(store)


def all_results() -> dict:
    """Read all stored results."""
    return _load_store()


# ---------- Cross-tier comparison plots -----------------------------------


def plot_model_comparison(dataset_name: str) -> Path | None:
    """Bar chart comparing all models on a single dataset."""
    store = _load_store()
    bucket = store.get(dataset_name, {})
    if not bucket:
        return None
    models = list(bucket.keys())
    bin_acc = [bucket[m]["binary_accuracy"] * 100 for m in models]
    multi_acc = [bucket[m]["multi_accuracy"] * 100 for m in models]
    macro_f1 = [bucket[m]["macro_f1"] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 5))
    ax.bar(x - width, bin_acc, width, label="Binary Accuracy (%)")
    ax.bar(x, multi_acc, width, label="Multi-class Accuracy (%)")
    ax.bar(x + width, macro_f1, width, label="Macro F1 (x100)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score")
    ax.set_title(f"Model comparison on {dataset_name}")
    ax.legend()
    plt.tight_layout()
    out = CONFIG.paths.plots / f"comparison_{dataset_name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def best_model_for_calibration(dataset_name: str) -> str | None:
    """Return the model name with the highest macro F1 on this dataset."""
    bucket = _load_store().get(dataset_name, {})
    if not bucket:
        return None
    return max(bucket, key=lambda m: bucket[m].get("macro_f1", 0))


def save_calibration_for_best(dataset_name: str) -> Path | None:
    """Save a reliability diagram for the best-macro-F1 model on this dataset.

    Uses the cached per-record P(attack) values from the most recent
    :func:`evaluate_classifier` call within this Python process. Returns the
    saved path, or ``None`` if no probability information is available.
    """
    best = best_model_for_calibration(dataset_name)
    if best is None:
        return None
    proba = _PROBA_CACHE.get((dataset_name, best))
    y_bin = _BIN_LABELS_CACHE.get(dataset_name)
    if proba is None or y_bin is None:
        log.info(
            "calibration: no cached probabilities for (%s, %s); "
            "re-run with --tier all in one process to populate the cache.",
            dataset_name,
            best,
        )
        return None
    return calibration_plot(best, dataset_name, y_bin, proba)


def calibration_plot(
    name: str,
    dataset_name: str,
    y_true_binary: np.ndarray,
    proba_attack: np.ndarray,
    n_bins: int = 10,
) -> Path:
    """Reliability diagram: empirical vs predicted attack probability."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(proba_attack, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    centres, fracs = [], []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        centres.append(proba_attack[mask].mean())
        fracs.append(y_true_binary[mask].mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.plot(centres, fracs, "o-", label=name)
    ax.set_xlabel("Mean predicted P(attack)")
    ax.set_ylabel("Empirical attack rate")
    ax.set_title(f"Calibration: {name} ({dataset_name})")
    ax.legend()
    plt.tight_layout()
    out = CONFIG.paths.plots / f"calibration_{dataset_name}_{name.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
