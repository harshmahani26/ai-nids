"""SHAP-based explanations for tree boosting models.

Uses :class:`shap.TreeExplainer` for fast, exact attributions on XGBoost or
LightGBM. Saves a beeswarm plot, a global bar plot, and a per-class importance
heatmap to ``results/shap/``.

A waterfall plot for one chosen attack record is also produced; the notebook
stitches it together with prose for the SHAP walkthrough section.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.config import CONFIG
from src.data.loader import DatasetMeta

log = logging.getLogger(__name__)


def _is_tree_model(model: Any) -> bool:
    name = type(model).__name__
    return name in {"XGBClassifier", "LGBMClassifier", "RandomForestClassifier"}


def explain_tree_model(
    *,
    model: Any,
    X_test: np.ndarray,
    feature_names: list[str],
    meta: DatasetMeta,
    sample_size: int = 2_000,
    name: str = "model",
) -> dict[str, Path]:
    """Compute and save SHAP plots for a tree model.

    Returns a dict of ``{description: file path}``.
    """
    if not _is_tree_model(model):
        raise TypeError(f"SHAP TreeExplainer not supported for {type(model).__name__}")

    rng = np.random.RandomState(CONFIG.train.seed)
    n = min(sample_size, len(X_test))
    sample_idx = rng.choice(len(X_test), size=n, replace=False)
    Xs = X_test[sample_idx]

    log.info("SHAP TreeExplainer on %d samples for %s", n, name)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    out_dir = CONFIG.paths.shap
    out_dir.mkdir(parents=True, exist_ok=True)
    base = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    paths: dict[str, Path] = {}

    # XGBoost / LightGBM multi-class returns shape (n, F, C); RF returns list-of-class arrays.
    if isinstance(shap_values, list):
        # Stack into (C, n, F) -> (n, F, C) for uniform handling.
        sv = np.stack(shap_values, axis=0)
        sv = np.transpose(sv, (1, 2, 0))
    else:
        sv = np.asarray(shap_values)
    if sv.ndim == 2:
        sv = sv[:, :, np.newaxis]  # binary edge case

    # Global mean(|SHAP|) per feature, averaged across classes.
    abs_mean = np.mean(np.abs(sv), axis=(0, 2))
    order = np.argsort(abs_mean)[::-1][:20]

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in order[::-1]], abs_mean[order[::-1]])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP global importance: {name} ({meta.name})")
    plt.tight_layout()
    p = out_dir / f"{meta.name}_{base}_global_bar.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths["global_bar"] = p

    # Beeswarm summary on the *class with most spread* (typically the dominant attack).
    class_mean_abs = np.mean(np.abs(sv), axis=(0, 1))
    primary_class = int(np.argmax(class_mean_abs))
    sv_primary = sv[..., primary_class]
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(
        sv_primary,
        Xs,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.title(f"SHAP beeswarm for class '{meta.labels[primary_class]}' ({name}, {meta.name})")
    plt.tight_layout()
    p = out_dir / f"{meta.name}_{base}_beeswarm.png"
    plt.savefig(p, dpi=130)
    plt.close()
    paths["beeswarm"] = p

    # Per-class importance heatmap
    per_class_abs = np.mean(np.abs(sv), axis=0)  # (F, C)
    top_features = np.argsort(per_class_abs.mean(axis=1))[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(per_class_abs[top_features, :], aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([feature_names[i] for i in top_features])
    ax.set_xticks(range(len(meta.labels)))
    ax.set_xticklabels(meta.labels, rotation=30, ha="right")
    fig.colorbar(im, label="Mean |SHAP|")
    ax.set_title(f"Per-class SHAP importance: {name} ({meta.name})")
    plt.tight_layout()
    p = out_dir / f"{meta.name}_{base}_per_class.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths["per_class"] = p

    log.info("SHAP saved: %s", ", ".join(str(p.name) for p in paths.values()))
    return paths


def waterfall_for_record(
    *,
    model: Any,
    X_record: np.ndarray,
    feature_names: list[str],
    meta: DatasetMeta,
    record_label: int,
    name: str = "model",
) -> Path:
    """Save a SHAP waterfall plot for a single record's predicted class.

    The notebook stitches this together with prose for the explainability
    walkthrough.
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_record.reshape(1, -1))
    if isinstance(sv, list):
        sv = np.stack(sv, axis=0)
        sv = np.transpose(sv, (1, 2, 0))
    sv = np.asarray(sv)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, tuple, np.ndarray)):
        base_for_class = float(np.asarray(base_value)[record_label])
    else:
        base_for_class = float(base_value)
    sv_class = sv[0, :, record_label] if sv.ndim == 3 else sv[0]

    expl = shap.Explanation(
        values=sv_class,
        base_values=base_for_class,
        data=X_record,
        feature_names=feature_names,
    )

    fig = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(expl, max_display=12, show=False)
    plt.title(f"SHAP waterfall: predicted class={meta.labels[record_label]} ({name})")
    plt.tight_layout()
    out = CONFIG.paths.shap / f"{meta.name}_{name.lower().replace(' ', '_')}_waterfall.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    log.info("SHAP waterfall saved: %s", out.name)
    return out
