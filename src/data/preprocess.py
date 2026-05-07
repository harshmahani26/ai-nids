"""Preprocessing utilities for NSL-KDD and UNSW-NB15.

The :class:`Preprocessor` is fit on training data only. It returns numpy arrays
suitable for sklearn / boosting / PyTorch consumption.

For models that handle categoricals natively (e.g. CatBoost), use the
``raw`` accessor and pass ``cat_features`` directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data.loader import DatasetMeta

log = logging.getLogger(__name__)


@dataclass
class PreprocessedSplit:
    """A fully-encoded train/test split."""

    X_train: np.ndarray
    y_train_multi: np.ndarray
    y_train_binary: np.ndarray
    X_test: np.ndarray
    y_test_multi: np.ndarray
    y_test_binary: np.ndarray
    feature_names: list[str]
    meta: DatasetMeta


class Preprocessor:
    """Label-encode categoricals (fit on train), standardise numerics.

    Unseen categorical values at transform time are mapped to a fixed sentinel
    integer (0). This is intentional: it means novel-attack rows get a plausible
    encoding without leaking test-set information back into training.
    """

    def __init__(self, meta: DatasetMeta) -> None:
        self.meta = meta
        self._encoders: dict[str, LabelEncoder] = {
            c: LabelEncoder() for c in meta.categorical_cols
        }
        self._scaler = StandardScaler()
        self._fitted = False

    def fit_transform(
        self, X: pd.DataFrame, y_multi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit encoders and scaler on the training data, then transform.

        Parameters
        ----------
        X : pd.DataFrame
            Raw features (categoricals as strings, numerics as floats).
        y_multi : np.ndarray
            Multi-class integer labels.

        Returns
        -------
        X_arr : np.ndarray
            Standardised numeric matrix; categorical columns first, then
            numeric, in the order declared by ``meta``.
        y_multi : np.ndarray
            Pass-through of the input labels.
        y_binary : np.ndarray
            Derived 0/1 labels: ``y_multi != normal_index``.
        """
        Xc = X.copy()
        for col in self.meta.categorical_cols:
            Xc[col] = self._encoders[col].fit_transform(Xc[col].astype(str))
        ordered = self.meta.categorical_cols + self.meta.numeric_cols
        arr = self._scaler.fit_transform(Xc[ordered].to_numpy(dtype=np.float64))
        self._fitted = True
        y_binary = (y_multi != self.meta.normal_index).astype(np.int64)
        return arr, y_multi, y_binary

    def transform(
        self, X: pd.DataFrame, y_multi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform new data using the fitted encoders and scaler.

        Unseen categorical values are mapped to the sentinel index ``0``,
        so novel-attack rows at test time do not error out.

        Parameters
        ----------
        X, y_multi : same as :meth:`fit_transform`.

        Returns
        -------
        X_arr, y_multi, y_binary
            Same structure as :meth:`fit_transform`.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit_transform`.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit before transform")
        Xc = X.copy()
        for col in self.meta.categorical_cols:
            le = self._encoders[col]
            classes = set(le.classes_)
            Xc[col] = Xc[col].astype(str).map(
                lambda v, le=le, classes=classes: le.transform([v])[0] if v in classes else 0
            )
        ordered = self.meta.categorical_cols + self.meta.numeric_cols
        arr = self._scaler.transform(Xc[ordered].to_numpy(dtype=np.float64))
        y_binary = (y_multi != self.meta.normal_index).astype(np.int64)
        return arr, y_multi, y_binary

    @property
    def feature_names(self) -> list[str]:
        return self.meta.categorical_cols + self.meta.numeric_cols


def fit_split(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    meta: DatasetMeta,
) -> PreprocessedSplit:
    """Convenience: fit on train, transform test, return both."""
    pp = Preprocessor(meta)
    Xtr, ytm, ytb = pp.fit_transform(X_train, y_train)
    Xte, yem, yeb = pp.transform(X_test, y_test)
    return PreprocessedSplit(
        X_train=Xtr,
        y_train_multi=ytm,
        y_train_binary=ytb,
        X_test=Xte,
        y_test_multi=yem,
        y_test_binary=yeb,
        feature_names=pp.feature_names,
        meta=meta,
    )
