"""Tier 2: Gradient boosting trees with Optuna tuning.

- XGBoost and LightGBM run on the encoded numeric matrix.
- CatBoost gets the *raw* DataFrame so it can use ``cat_features`` natively;
  this is the whole point of including CatBoost alongside the other two.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.config import CONFIG
from src.data.loader import DatasetMeta
from src.training.tune import make_study

log = logging.getLogger(__name__)


# ---------- helpers --------------------------------------------------------


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _holdout(
    X: np.ndarray, y: np.ndarray, *, val_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=val_size, stratify=y, random_state=CONFIG.train.seed)


def _holdout_df(
    X_df: pd.DataFrame, y: np.ndarray, *, val_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    return train_test_split(X_df, y, test_size=val_size, stratify=y, random_state=CONFIG.train.seed)


# ---------- XGBoost --------------------------------------------------------


def tune_xgboost(X: np.ndarray, y: np.ndarray, meta: DatasetMeta):
    """Optuna-tuned XGBoost multi-class classifier."""
    import xgboost as xgb

    X_tr, X_val, y_tr, y_val = _holdout(X, y)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "objective": "multi:softprob",
            "num_class": meta.num_classes,
            "tree_method": "hist",
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "random_state": CONFIG.train.seed,
            "n_jobs": -1,
            "verbosity": 0,
        }
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return _macro_f1(y_val, clf.predict(X_val))

    study = make_study("xgb")
    study.optimize(objective, n_trials=CONFIG.train.optuna_trials, show_progress_bar=False)
    log.info("XGB best: %s | macroF1=%.4f", study.best_params, study.best_value)

    final = xgb.XGBClassifier(
        **study.best_params,
        objective="multi:softprob",
        num_class=meta.num_classes,
        tree_method="hist",
        random_state=CONFIG.train.seed,
        n_jobs=-1,
        verbosity=0,
    )
    final.fit(X, y, verbose=False)
    return final


# ---------- LightGBM -------------------------------------------------------


def tune_lightgbm(X: np.ndarray, y: np.ndarray, meta: DatasetMeta):
    """Optuna-tuned LightGBM multi-class classifier."""
    import lightgbm as lgb

    X_tr, X_val, y_tr, y_val = _holdout(X, y)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "objective": "multiclass",
            "num_class": meta.num_classes,
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "random_state": CONFIG.train.seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            np.asarray(X_tr),
            y_tr,
            eval_set=[(np.asarray(X_val), y_val)],
            callbacks=[lgb.log_evaluation(0)],
        )
        return _macro_f1(y_val, clf.predict(np.asarray(X_val)))

    study = make_study("lgb")
    study.optimize(objective, n_trials=CONFIG.train.optuna_trials, show_progress_bar=False)
    log.info("LGB best: %s | macroF1=%.4f", study.best_params, study.best_value)

    final = lgb.LGBMClassifier(
        **study.best_params,
        objective="multiclass",
        num_class=meta.num_classes,
        random_state=CONFIG.train.seed,
        n_jobs=-1,
        verbose=-1,
    )
    final.fit(X, y, callbacks=[lgb.log_evaluation(0)])
    return final


# ---------- CatBoost (native categoricals) --------------------------------


def tune_catboost(X_df: pd.DataFrame, y: np.ndarray, meta: DatasetMeta):
    """Optuna-tuned CatBoost using native ``cat_features``.

    Receives the *raw* (un-encoded) DataFrame: this is the differentiator
    versus XGBoost/LightGBM, which see only post-encoding numerics.
    """
    from catboost import CatBoostClassifier, Pool

    cat_idx = [X_df.columns.get_loc(c) for c in meta.categorical_cols if c in X_df.columns]

    # CatBoost handles NaN natively but expects categoricals as strings.
    Xc = X_df.copy()
    for c in meta.categorical_cols:
        if c in Xc.columns:
            Xc[c] = Xc[c].astype(str)

    X_tr_df, X_val_df, y_tr, y_val = _holdout_df(Xc, y)
    train_pool = Pool(X_tr_df, y_tr, cat_features=cat_idx)
    val_pool = Pool(X_val_df, y_val, cat_features=cat_idx)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "iterations": trial.suggest_int("iterations", 200, 800, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1",
            "random_seed": CONFIG.train.seed,
            "verbose": 0,
            "allow_writing_files": False,
        }
        clf = CatBoostClassifier(**params)
        clf.fit(train_pool, eval_set=val_pool, verbose=0)
        preds = clf.predict(X_val_df).ravel().astype(int)
        return _macro_f1(y_val, preds)

    study = make_study("catboost")
    study.optimize(objective, n_trials=CONFIG.train.optuna_trials, show_progress_bar=False)
    log.info("CatBoost best: %s | macroF1=%.4f", study.best_params, study.best_value)

    final = CatBoostClassifier(
        **study.best_params,
        loss_function="MultiClass",
        random_seed=CONFIG.train.seed,
        verbose=0,
        allow_writing_files=False,
    )
    final_pool = Pool(Xc, y, cat_features=cat_idx)
    final.fit(final_pool, verbose=0)
    # Wrap so evaluate.py can call .predict on a raw DataFrame the same way it
    # does on numpy arrays from other models.
    return _CatBoostPredictWrapper(final, list(Xc.columns), meta.categorical_cols)


class _CatBoostPredictWrapper:
    """Thin wrapper so CatBoost can accept the same X_test_df we pass it.

    Ensures the categorical columns are stringified (CatBoost is strict).
    """

    def __init__(self, model, columns: list[str], cat_cols: list[str]) -> None:
        self.model = model
        self.columns = columns
        self.cat_cols = cat_cols

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X[self.columns].copy()
        for c in self.cat_cols:
            if c in Xc.columns:
                Xc[c] = Xc[c].astype(str)
        return Xc

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prep(X)).ravel().astype(np.int64)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._prep(X))
