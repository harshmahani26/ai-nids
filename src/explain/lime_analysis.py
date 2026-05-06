"""LIME explanations for deep models.

Generates per-instance explanations for a sample of correctly-classified
attacks and false negatives. Saves PNGs and HTML files into
``results/lime_examples/`` for the notebook walkthrough.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from src.config import CONFIG
from src.data.loader import DatasetMeta

log = logging.getLogger(__name__)


def explain_deep_model(
    *,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    meta: DatasetMeta,
    name: str = "deep",
    n_examples: int = 6,
) -> dict[str, Path]:
    """Save LIME explanations for ``n_examples`` records.

    Half are correctly-classified attacks, half are false negatives.
    """
    out_dir = CONFIG.paths.lime
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("LIME explainer on %s", name)
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=meta.labels,
        mode="classification",
        discretize_continuous=True,
        random_state=CONFIG.train.seed,
    )

    y_pred = model.predict(X_test)
    correct_attack = np.where((y_test != meta.normal_index) & (y_pred == y_test))[0]
    false_negative = np.where((y_test != meta.normal_index) & (y_pred == meta.normal_index))[0]

    rng = np.random.RandomState(CONFIG.train.seed)
    half = max(1, n_examples // 2)
    correct_pick = (
        rng.choice(correct_attack, size=min(half, len(correct_attack)), replace=False)
        if len(correct_attack)
        else np.array([], dtype=int)
    )
    fn_pick = (
        rng.choice(
            false_negative,
            size=min(n_examples - len(correct_pick), len(false_negative)),
            replace=False,
        )
        if len(false_negative)
        else np.array([], dtype=int)
    )

    paths: dict[str, Path] = {}
    for tag, indices in (("correct", correct_pick), ("false_negative", fn_pick)):
        for k, i in enumerate(indices):
            exp = explainer.explain_instance(
                X_test[i],
                model.predict_proba,
                num_features=10,
                top_labels=3,
            )
            base = f"{meta.name}_{name.lower().replace(' ', '_')}_{tag}_{k}"
            html_path = out_dir / f"{base}.html"
            html_path.write_text(exp.as_html(), encoding="utf-8")
            paths[f"{tag}_{k}_html"] = html_path

            # PNG for the predicted class.
            fig = exp.as_pyplot_figure(label=int(y_pred[i]))
            fig.suptitle(
                f"{name} | true={meta.labels[int(y_test[i])]}  pred={meta.labels[int(y_pred[i])]}"
            )
            plt.tight_layout()
            png_path = out_dir / f"{base}.png"
            fig.savefig(png_path, dpi=130, bbox_inches="tight")
            plt.close(fig)
            paths[f"{tag}_{k}_png"] = png_path

    log.info("LIME saved %d files for %s", len(paths), name)
    return paths
