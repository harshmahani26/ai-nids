"""Top-level CLI orchestrator.

Usage::

    python -m src.train --dataset nslkdd --tier classical
    python -m src.train --dataset unsw    --tier all

Each tier writes its metrics into ``results/metrics.json`` and saves plots into
``results/``.
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from typing import Any

import numpy as np

from src.config import CONFIG, setup_logging
from src.data.loader import load_dataset
from src.data.preprocess import fit_split

log = logging.getLogger(__name__)

VALID_DATASETS = ("nslkdd", "unsw")
VALID_TIERS = ("classical", "boosting", "deep", "hybrid", "stacking", "all")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_classical(split, X_train_df=None, X_test_df=None) -> None:
    """Tier 1: RF, SVM, KNN, Voting Ensemble."""
    from src.evaluate import evaluate_classifier, upsert_result
    from src.models.classical import (
        build_voting_ensemble,
        tune_knn,
        tune_rf,
        tune_svm,
    )

    log.info("=== Tier 1: Classical baselines (%s) ===", split.meta.name)
    t0 = time.perf_counter()
    rf = tune_rf(split.X_train, split.y_train_multi)
    rf_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    svm = tune_svm(split.X_train, split.y_train_multi)
    svm_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    knn = tune_knn(split.X_train, split.y_train_multi)
    knn_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    voting = build_voting_ensemble(rf, svm, knn)
    voting.fit(split.X_train, split.y_train_multi)
    voting_t = time.perf_counter() - t0

    for name, model, t in [
        ("Random Forest", rf, rf_t),
        ("SVM (RBF)", svm, svm_t),
        ("KNN", knn, knn_t),
        ("Voting Ensemble", voting, voting_t),
    ]:
        result = evaluate_classifier(
            name=name,
            model=model,
            X_test=split.X_test,
            y_test_multi=split.y_test_multi,
            y_test_binary=split.y_test_binary,
            meta=split.meta,
            train_time_s=t,
        )
        upsert_result(result)


def run_boosting(split, X_train_df=None, X_test_df=None) -> None:
    """Tier 2: XGBoost, LightGBM, CatBoost (Optuna-tuned).

    Each model is evaluated and persisted *as soon as it finishes tuning* so a
    crash midway through a long run does not lose earlier work.
    """
    from src.evaluate import evaluate_classifier, upsert_result
    from src.models.boosting import tune_catboost, tune_lightgbm, tune_xgboost

    log.info("=== Tier 2: Gradient boosting (%s) ===", split.meta.name)

    t0 = time.perf_counter()
    xgb = tune_xgboost(split.X_train, split.y_train_multi, split.meta)
    upsert_result(evaluate_classifier(
        name="XGBoost", model=xgb, X_test=split.X_test,
        y_test_multi=split.y_test_multi, y_test_binary=split.y_test_binary,
        meta=split.meta, train_time_s=time.perf_counter() - t0,
    ))

    t0 = time.perf_counter()
    lgb = tune_lightgbm(split.X_train, split.y_train_multi, split.meta)
    upsert_result(evaluate_classifier(
        name="LightGBM", model=lgb, X_test=split.X_test,
        y_test_multi=split.y_test_multi, y_test_binary=split.y_test_binary,
        meta=split.meta, train_time_s=time.perf_counter() - t0,
    ))

    t0 = time.perf_counter()
    # CatBoost gets the raw DataFrame so it can use cat_features natively.
    cat = tune_catboost(X_train_df, split.y_train_multi, split.meta)
    upsert_result(evaluate_classifier(
        name="CatBoost", model=cat, X_test=X_test_df,
        y_test_multi=split.y_test_multi, y_test_binary=split.y_test_binary,
        meta=split.meta, train_time_s=time.perf_counter() - t0,
    ))


def run_deep(split) -> None:
    """Tier 3: 1D-CNN, LSTM, Autoencoder.

    Each model is evaluated and persisted as soon as it finishes training.
    """
    from src.evaluate import evaluate_classifier, upsert_result
    from src.models.deep import AutoencoderDetector, train_cnn, train_lstm

    log.info("=== Tier 3: Deep learning (%s) ===", split.meta.name)

    t0 = time.perf_counter()
    try:
        cnn = train_cnn(split)
        upsert_result(evaluate_classifier(
            name="1D-CNN", model=cnn, X_test=split.X_test,
            y_test_multi=split.y_test_multi, y_test_binary=split.y_test_binary,
            meta=split.meta, train_time_s=time.perf_counter() - t0,
        ))
    except Exception:
        log.exception("CNN failed")

    t0 = time.perf_counter()
    try:
        lstm = train_lstm(split)
        upsert_result(evaluate_classifier(
            name="LSTM", model=lstm, X_test=split.X_test,
            y_test_multi=split.y_test_multi, y_test_binary=split.y_test_binary,
            meta=split.meta, train_time_s=time.perf_counter() - t0,
        ))
    except Exception:
        log.exception("LSTM failed")

    t0 = time.perf_counter()
    try:
        ae = AutoencoderDetector().fit(split)
        upsert_result(evaluate_classifier(
            name="Autoencoder", model=ae, X_test=split.X_test,
            y_test_multi=split.y_test_multi, y_test_binary=split.y_test_binary,
            meta=split.meta, train_time_s=time.perf_counter() - t0,
            extra={"threshold": float(ae.threshold_), "approach": "reconstruction-error"},
        ))
    except Exception:
        log.exception("Autoencoder failed")


def run_hybrid(split) -> None:
    """Tier 4: RF prefilter + LSTM cascade."""
    from src.evaluate import evaluate_classifier, upsert_result
    from src.models.hybrid import HybridRFLSTM

    log.info("=== Tier 4: Hybrid pipeline (%s) ===", split.meta.name)
    t0 = time.perf_counter()
    hybrid = HybridRFLSTM().fit(split)
    train_t = time.perf_counter() - t0

    upsert_result(
        evaluate_classifier(
            name="Hybrid RF+LSTM",
            model=hybrid,
            X_test=split.X_test,
            y_test_multi=split.y_test_multi,
            y_test_binary=split.y_test_binary,
            meta=split.meta,
            train_time_s=train_t,
            extra={
                "threshold": float(hybrid.threshold_),
                "slow_path_fraction": float(hybrid.slow_path_fraction_),
            },
        )
    )


def run_stacking(split) -> None:
    """Tier 5: Stacking ensemble."""
    from src.evaluate import evaluate_classifier, upsert_result
    from src.models.ensemble import build_stacking

    log.info("=== Tier 5: Stacking ensemble (%s) ===", split.meta.name)
    t0 = time.perf_counter()
    stack = build_stacking(split)
    train_t = time.perf_counter() - t0
    upsert_result(
        evaluate_classifier(
            name="Stacking",
            model=stack,
            X_test=split.X_test,
            y_test_multi=split.y_test_multi,
            y_test_binary=split.y_test_binary,
            meta=split.meta,
            train_time_s=train_t,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-NIDS multi-tier trainer")
    parser.add_argument("--dataset", choices=VALID_DATASETS, required=True)
    parser.add_argument("--tier", choices=VALID_TIERS, default="all")
    parser.add_argument("--seed", type=int, default=CONFIG.train.seed)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    CONFIG.paths.ensure()
    _set_seed(args.seed)

    log.info("Loading dataset: %s", args.dataset)
    X_train_df, y_train, X_test_df, y_test, meta = load_dataset(args.dataset)
    split = fit_split(X_train_df, y_train, X_test_df, y_test, meta)

    tiers: dict[str, Any] = {
        "classical": lambda: run_classical(split, X_train_df, X_test_df),
        "boosting": lambda: run_boosting(split, X_train_df, X_test_df),
        "deep": lambda: run_deep(split),
        "hybrid": lambda: run_hybrid(split),
        "stacking": lambda: run_stacking(split),
    }
    if args.tier == "all":
        for name in ("classical", "boosting", "deep", "hybrid", "stacking"):
            tiers[name]()
    else:
        tiers[args.tier]()

    from src.evaluate import plot_model_comparison

    out = plot_model_comparison(args.dataset)
    if out:
        log.info("comparison plot: %s", out)


if __name__ == "__main__":
    main()
