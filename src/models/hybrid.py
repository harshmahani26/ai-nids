"""Tier 4: Hybrid pipeline (RF prefilter + LSTM deep stage).

Stage 1 is a Random Forest that produces a fast P(attack) score per record.
Records below the threshold use the RF's own multi-class prediction.
Records above go to the LSTM for fine-grained classification.

The threshold is chosen on a validation split to balance the slow-path fraction
against multi-class accuracy. The selected threshold and the resulting
slow-path fraction are reported in metrics.json so reviewers can see how much
traffic actually exercises the deep stage in practice.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.config import CONFIG
from src.data.preprocess import PreprocessedSplit
from src.models.deep import _LSTMClassifier, train_lstm

log = logging.getLogger(__name__)


class HybridRFLSTM:
    """Two-stage cascade: cheap RF prefilter feeds LSTM only when needed."""

    def __init__(self) -> None:
        self.rf: RandomForestClassifier | None = None
        self.lstm: _LSTMClassifier | None = None
        self.threshold_: float = 0.5
        self.slow_path_fraction_: float = 0.0
        self.normal_index_: int = 0

    def fit(self, split: PreprocessedSplit) -> HybridRFLSTM:
        self.normal_index_ = split.meta.normal_index

        # Hold out a validation set for threshold selection.
        idx = np.arange(len(split.X_train))
        tr_idx, val_idx = train_test_split(
            idx,
            test_size=0.2,
            stratify=split.y_train_multi,
            random_state=CONFIG.train.seed,
        )

        X_tr, X_val = split.X_train[tr_idx], split.X_train[val_idx]
        y_tr, y_val = split.y_train_multi[tr_idx], split.y_train_multi[val_idx]

        log.info("Hybrid stage 1: training RF prefilter")
        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=CONFIG.train.seed
        )
        self.rf.fit(X_tr, y_tr)

        log.info("Hybrid stage 2: training LSTM deep stage")
        # Reuse the standard LSTM trainer on the same training fold.
        sub_split = PreprocessedSplit(
            X_train=X_tr,
            y_train_multi=y_tr,
            y_train_binary=(y_tr != self.normal_index_).astype(np.int64),
            X_test=split.X_test,
            y_test_multi=split.y_test_multi,
            y_test_binary=split.y_test_binary,
            feature_names=split.feature_names,
            meta=split.meta,
        )
        self.lstm = train_lstm(sub_split)

        # Choose threshold on the held-out validation slice.
        rf_proba = self.rf.predict_proba(X_val)
        attack_p = 1.0 - rf_proba[:, self.normal_index_]
        rf_pred = self.rf.predict(X_val)
        lstm_pred = self.lstm.predict(X_val)

        best_acc, best_thr = -1.0, 0.5
        best_frac = 1.0
        for thr in CONFIG.train.hybrid_threshold_search:
            mask = attack_p >= thr
            preds = np.where(mask, lstm_pred, rf_pred)
            acc = accuracy_score(y_val, preds)
            frac = float(mask.mean())
            log.debug("thr=%.2f  acc=%.4f  slow_path=%.2f", thr, acc, frac)
            # Pick the threshold that maximises accuracy; tiebreak by lower slow path.
            if acc > best_acc + 1e-4 or (abs(acc - best_acc) < 1e-4 and frac < best_frac):
                best_acc, best_thr, best_frac = acc, float(thr), frac
        self.threshold_ = best_thr
        self.slow_path_fraction_ = best_frac
        log.info(
            "Hybrid threshold=%.2f val_acc=%.4f slow_path_fraction=%.2f",
            self.threshold_,
            best_acc,
            self.slow_path_fraction_,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.rf is not None and self.lstm is not None
        rf_proba = self.rf.predict_proba(X)
        attack_p = 1.0 - rf_proba[:, self.normal_index_]
        slow_mask = attack_p >= self.threshold_
        rf_pred = self.rf.predict(X)
        out = rf_pred.astype(np.int64).copy()
        if slow_mask.any():
            lstm_pred = self.lstm.predict(X[slow_mask])
            out[slow_mask] = lstm_pred
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.rf is not None and self.lstm is not None
        rf_proba = self.rf.predict_proba(X)
        attack_p = 1.0 - rf_proba[:, self.normal_index_]
        slow_mask = attack_p >= self.threshold_
        out = rf_proba.copy()
        if slow_mask.any():
            lstm_proba = self.lstm.predict_proba(X[slow_mask])
            # Align widths: both should be n_classes.
            if lstm_proba.shape[1] == out.shape[1]:
                out[slow_mask] = lstm_proba
        return out
