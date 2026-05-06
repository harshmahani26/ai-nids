"""Tier 1: Classical baselines (RF, SVM, KNN, soft-voting ensemble).

GridSearchCV with 5-fold CV is used for tuning. SVM grid search uses a stratified
subsample to keep wall time tractable; the best estimator is then refit on the
full training set.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.config import CONFIG

log = logging.getLogger(__name__)


def _gs(estimator, grid, scoring: str = "accuracy", n_jobs: int = -1) -> GridSearchCV:
    return GridSearchCV(
        estimator,
        grid,
        cv=CONFIG.train.cv_folds,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=0,
    )


def tune_rf(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Tune Random Forest via grid search; returns refit best estimator.

    Sets ``n_jobs=1`` inside RF and parallelises across CV folds via
    GridSearchCV. Avoids the Windows ``WinError 1450`` from nested joblib pools.
    """
    grid = {"n_estimators": [100, 200, 300], "max_depth": [10, 20, None]}
    gs = _gs(RandomForestClassifier(random_state=CONFIG.train.seed, n_jobs=1), grid)
    gs.fit(X, y)
    log.info("RF best: %s | cv_acc=%.4f", gs.best_params_, gs.best_score_)
    # Refit with parallel trees for fast inference.
    best = gs.best_estimator_
    best.set_params(n_jobs=-1)
    return best


def tune_svm(X: np.ndarray, y: np.ndarray) -> SVC:
    """Tune SVM (RBF) on a stratified subsample, then refit on full data."""
    grid = {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1]}
    n = min(CONFIG.train.svm_subsample, len(X))
    if n < len(X):
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=CONFIG.train.seed)
        idx, _ = next(sss.split(X, y))
        Xs, ys = X[idx], y[idx]
        log.info("SVM grid on stratified subsample: %d / %d", n, len(X))
    else:
        Xs, ys = X, y

    gs = _gs(SVC(kernel="rbf", probability=True, random_state=CONFIG.train.seed), grid)
    gs.fit(Xs, ys)
    log.info("SVM best: %s | cv_acc=%.4f", gs.best_params_, gs.best_score_)
    best = gs.best_estimator_
    if n < len(X):
        log.info("refitting SVM on full training set (%d rows)", len(X))
        best.fit(X, y)
    return best


def tune_knn(X: np.ndarray, y: np.ndarray) -> KNeighborsClassifier:
    """Tune KNN via grid search.

    Uses ``algorithm='kd_tree'`` so prediction time scales as O(log N) instead
    of O(N) brute force; this matters when the test set is large (UNSW-NB15
    has 175k test rows).
    """
    grid = {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
    gs = _gs(KNeighborsClassifier(n_jobs=1, algorithm="kd_tree"), grid)
    gs.fit(X, y)
    log.info("KNN best: %s | cv_acc=%.4f", gs.best_params_, gs.best_score_)
    best = gs.best_estimator_
    best.set_params(n_jobs=-1)
    return best


def build_voting_ensemble(rf, svm, knn) -> VotingClassifier:
    """Soft-voting ensemble across the three tuned classical models.

    ``n_jobs=1`` at the voting level: RF and KNN already parallelise internally,
    and Windows joblib does not handle nested loky pools well.
    """
    return VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("knn", knn)],
        voting="soft",
        n_jobs=1,
    )
