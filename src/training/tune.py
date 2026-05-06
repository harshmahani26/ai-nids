"""Optuna tuning helpers shared by boosting models."""

from __future__ import annotations

import logging

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

from src.config import CONFIG

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_study(name: str, *, direction: str = "maximize") -> optuna.Study:
    """In-memory study; deterministic via seeded sampler."""
    sampler = optuna.samplers.TPESampler(seed=CONFIG.train.seed)
    return optuna.create_study(study_name=name, direction=direction, sampler=sampler)


def cv_iter(X: np.ndarray, y: np.ndarray, n_splits: int | None = None):
    """Stratified k-fold splitter; reuses CONFIG.train.cv_folds by default."""
    n = n_splits or CONFIG.train.cv_folds
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=CONFIG.train.seed)
    return skf.split(X, y)
