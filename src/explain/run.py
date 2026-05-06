"""Run SHAP and LIME against trained models.

Designed to be called after the boosting and deep tiers have completed::

    python -m src.explain.run --dataset nslkdd
    python -m src.explain.run --dataset unsw

For SHAP we re-train a quick XGBoost on the spot (cheap, deterministic).
For LIME we re-train a 1D-CNN on the spot.
This keeps explainability self-contained instead of pickling models from train.py.
"""

from __future__ import annotations

import argparse
import logging

import lightgbm as lgb  # noqa: F401  (registers a backend with shap)
import numpy as np
import xgboost as xgb

from src.config import CONFIG, setup_logging
from src.data.loader import load_dataset
from src.data.preprocess import fit_split
from src.explain.lime_analysis import explain_deep_model
from src.explain.shap_analysis import explain_tree_model, waterfall_for_record
from src.models.deep import train_cnn

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SHAP + LIME for one dataset")
    parser.add_argument("--dataset", choices=("nslkdd", "unsw"), required=True)
    parser.add_argument("--shap-sample", type=int, default=2000)
    parser.add_argument("--lime-examples", type=int, default=4)
    args = parser.parse_args()

    setup_logging()
    CONFIG.paths.ensure()

    Xtr_df, ytr, Xte_df, yte, meta = load_dataset(args.dataset)
    split = fit_split(Xtr_df, ytr, Xte_df, yte, meta)

    # ---------- SHAP on a fresh XGBoost --------------------------------
    log.info("Training XGBoost for SHAP analysis on %s", args.dataset)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.1,
        tree_method="hist",
        random_state=CONFIG.train.seed,
        objective="multi:softprob",
        num_class=meta.num_classes,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_clf.fit(split.X_train, split.y_train_multi, verbose=False)

    explain_tree_model(
        model=xgb_clf,
        X_test=split.X_test,
        feature_names=split.feature_names,
        meta=meta,
        sample_size=args.shap_sample,
        name="xgboost",
    )

    # Pick one correctly-classified attack record for the waterfall plot.
    y_pred = xgb_clf.predict(split.X_test)
    correct_attack = np.where(
        (split.y_test_multi != meta.normal_index) & (y_pred == split.y_test_multi)
    )[0]
    if len(correct_attack):
        idx = int(correct_attack[len(correct_attack) // 2])
        waterfall_for_record(
            model=xgb_clf,
            X_record=split.X_test[idx],
            feature_names=split.feature_names,
            meta=meta,
            record_label=int(y_pred[idx]),
            name="xgboost",
        )

    # ---------- LIME on a fresh 1D-CNN --------------------------------
    log.info("Training 1D-CNN for LIME analysis on %s", args.dataset)
    cnn = train_cnn(split)
    explain_deep_model(
        model=cnn,
        X_train=split.X_train,
        X_test=split.X_test,
        y_test=split.y_test_multi,
        feature_names=split.feature_names,
        meta=meta,
        name="cnn",
        n_examples=args.lime_examples,
    )


if __name__ == "__main__":
    main()
