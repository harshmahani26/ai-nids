"""Tier 5: Stacking ensemble.

Base learners (XGBoost, LightGBM, RandomForest) feed predictions through a
5-fold CV split into a Logistic Regression meta-learner. We use sklearn's
:class:`StackingClassifier` so cross-validation, calibration, and refitting
are handled correctly.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.config import CONFIG
from src.data.preprocess import PreprocessedSplit

log = logging.getLogger(__name__)


def build_stacking(split: PreprocessedSplit) -> StackingClassifier:
    """Train a stacking ensemble over tuned tree-based base learners."""
    # Use modest defaults rather than re-running Optuna; the boosting tier
    # already has tuned models, but rebuilding fresh keeps stacking
    # independent and reproducible.
    # Modest base learner sizes - StackingClassifier fits each 6 times
    # (5 OOF folds + 1 final), so the per-model size has to be tractable.
    base = [
        (
            "xgb",
            xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                tree_method="hist",
                random_state=CONFIG.train.seed,
                n_jobs=-1,
                verbosity=0,
                objective="multi:softprob",
                num_class=split.meta.num_classes,
            ),
        ),
        (
            "lgb",
            lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=32,
                learning_rate=0.1,
                random_state=CONFIG.train.seed,
                n_jobs=-1,
                verbose=-1,
                objective="multiclass",
                num_class=split.meta.num_classes,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                n_jobs=1,
                random_state=CONFIG.train.seed,
            ),
        ),
    ]
    meta = LogisticRegression(max_iter=2000, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=CONFIG.train.seed)

    log.info("Stacking: fitting base learners with 5-fold CV passthrough")
    stack = StackingClassifier(
        estimators=base,
        final_estimator=meta,
        cv=cv,
        stack_method="predict_proba",
        n_jobs=1,  # base learners already parallelise internally
        passthrough=False,
    )
    stack.fit(split.X_train, split.y_train_multi)
    return stack
