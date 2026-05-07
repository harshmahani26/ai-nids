"""Unified loaders for NSL-KDD and UNSW-NB15.

Returns ``(X_train_df, y_train_multi, X_test_df, y_test_multi, meta)``
where:

- ``X_*_df`` : pandas DataFrame with raw (un-encoded) features
- ``y_*_multi`` : 1-D ``numpy.ndarray[int]`` of multi-class labels
- ``meta`` : ``DatasetMeta`` describing column groups and label names

Downstream code can derive a binary label by ``y != normal_index``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG

log = logging.getLogger(__name__)


# ---------- NSL-KDD --------------------------------------------------------

NSLKDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

NSLKDD_CATEGORICAL = ["protocol_type", "service", "flag"]

NSLKDD_ATTACK_MAP = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "apache2": "dos", "udpstorm": "dos", "processtable": "dos",
    "worm": "dos", "mailbomb": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "snmpguess": "r2l", "snmpgetattack": "r2l", "httptunnel": "r2l",
    "sendmail": "r2l", "named": "r2l", "xlock": "r2l", "xsnoop": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}

NSLKDD_LABELS = ["normal", "dos", "probe", "r2l", "u2r"]


# ---------- UNSW-NB15 ------------------------------------------------------
# UNSW-NB15 official partition columns. Drop ``id`` and ``label`` (the binary
# 0/1 label is redundant given ``attack_cat``).

UNSW_DROP = ["id", "label"]
UNSW_LABEL_COL = "attack_cat"
UNSW_CATEGORICAL = ["proto", "service", "state"]
# Canonical category order (lower-cased). Listed alphabetically for stability.
UNSW_LABELS = [
    "normal",
    "analysis",
    "backdoor",
    "dos",
    "exploits",
    "fuzzers",
    "generic",
    "reconnaissance",
    "shellcode",
    "worms",
]


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    feature_cols: list[str]
    categorical_cols: list[str]
    numeric_cols: list[str]
    labels: list[str]
    normal_index: int

    @property
    def num_classes(self) -> int:
        return len(self.labels)


def _ensure_files(paths: list[Path]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"missing data files: {names}. Run: python -m src.data.download"
        )


# ---------- NSL-KDD loader -------------------------------------------------


def load_nslkdd() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, DatasetMeta]:
    """Load the NSL-KDD KDDTrain+ / KDDTest+ partition.

    Returns
    -------
    X_train : pd.DataFrame
        Raw (un-encoded) training features.
    y_train : np.ndarray
        Multi-class labels in ``{0, 1, 2, 3, 4}`` corresponding to
        ``["normal", "dos", "probe", "r2l", "u2r"]``.
    X_test : pd.DataFrame
        Raw test features.
    y_test : np.ndarray
        Multi-class test labels.
    meta : DatasetMeta
        Column groupings (categorical / numeric), label names, and the index of
        the ``"normal"`` class.

    Raises
    ------
    FileNotFoundError
        If the raw text files are missing under ``data/raw/``. Run
        ``python -m src.data.download`` first.
    """
    train_path = CONFIG.paths.data_raw / CONFIG.data.nslkdd_train
    test_path = CONFIG.paths.data_raw / CONFIG.data.nslkdd_test
    _ensure_files([train_path, test_path])

    def _read(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, header=None, names=NSLKDD_COLUMNS)
        # Drop the difficulty column (leakage) and the raw label after we map it.
        return df

    train_df = _read(train_path)
    test_df = _read(test_path)

    def _to_category(series: pd.Series) -> pd.Series:
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map(NSLKDD_ATTACK_MAP)
            .fillna("u2r")  # unseen subtypes -> u2r (per literature convention)
        )

    y_train = _to_category(train_df["label"]).map(NSLKDD_LABELS.index).to_numpy()
    y_test = _to_category(test_df["label"]).map(NSLKDD_LABELS.index).to_numpy()

    # Drop label and difficulty
    feature_cols = [c for c in NSLKDD_COLUMNS if c not in ("label", "difficulty")]
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    numeric_cols = [c for c in feature_cols if c not in NSLKDD_CATEGORICAL]
    meta = DatasetMeta(
        name="nslkdd",
        feature_cols=feature_cols,
        categorical_cols=list(NSLKDD_CATEGORICAL),
        numeric_cols=numeric_cols,
        labels=list(NSLKDD_LABELS),
        normal_index=NSLKDD_LABELS.index("normal"),
    )
    log.info(
        "nslkdd loaded: train=%s test=%s features=%d",
        X_train.shape, X_test.shape, len(feature_cols),
    )
    return X_train, y_train.astype(np.int64), X_test, y_test.astype(np.int64), meta


# ---------- UNSW-NB15 loader -----------------------------------------------


def load_unsw() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, DatasetMeta]:
    """Load the official UNSW-NB15 train / test partition.

    Returns
    -------
    X_train, y_train, X_test, y_test, meta
        Same structure as :func:`load_nslkdd`. Multi-class labels are in
        ``{0..9}`` over ``["normal", "analysis", "backdoor", "dos",
        "exploits", "fuzzers", "generic", "reconnaissance", "shellcode",
        "worms"]``. Categorical columns are ``["proto", "service", "state"]``,
        kept as raw strings so CatBoost can consume them natively.

    Raises
    ------
    FileNotFoundError
        If the raw CSVs are missing under ``data/raw/``.
    """
    train_path = CONFIG.paths.data_raw / CONFIG.data.unsw_train
    test_path = CONFIG.paths.data_raw / CONFIG.data.unsw_test
    _ensure_files([train_path, test_path])

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Drop redundant columns
    train_df = train_df.drop(columns=[c for c in UNSW_DROP if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in UNSW_DROP if c in test_df.columns])

    # Normalise label casing.
    train_df[UNSW_LABEL_COL] = train_df[UNSW_LABEL_COL].astype(str).str.strip().str.lower()
    test_df[UNSW_LABEL_COL] = test_df[UNSW_LABEL_COL].astype(str).str.strip().str.lower()

    # Map labels -> int. Unknown labels -> "normal" should never happen, but guard it.
    label_to_idx = {lab: i for i, lab in enumerate(UNSW_LABELS)}

    def _y(df: pd.DataFrame) -> np.ndarray:
        return df[UNSW_LABEL_COL].map(label_to_idx).fillna(0).astype(np.int64).to_numpy()

    y_train = _y(train_df)
    y_test = _y(test_df)

    X_train = train_df.drop(columns=[UNSW_LABEL_COL])
    X_test = test_df.drop(columns=[UNSW_LABEL_COL])

    feature_cols = list(X_train.columns)
    numeric_cols = [c for c in feature_cols if c not in UNSW_CATEGORICAL]

    # Lower-case categorical values for consistent encoding.
    for c in UNSW_CATEGORICAL:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype(str).str.strip().str.lower()
            X_test[c] = X_test[c].astype(str).str.strip().str.lower()

    meta = DatasetMeta(
        name="unsw",
        feature_cols=feature_cols,
        categorical_cols=[c for c in UNSW_CATEGORICAL if c in feature_cols],
        numeric_cols=numeric_cols,
        labels=list(UNSW_LABELS),
        normal_index=UNSW_LABELS.index("normal"),
    )
    log.info(
        "unsw loaded: train=%s test=%s features=%d",
        X_train.shape, X_test.shape, len(feature_cols),
    )
    return X_train, y_train, X_test, y_test, meta


def load_dataset(name: str):
    """Dispatch loader by name (``'nslkdd'`` or ``'unsw'``)."""
    match name.lower():
        case "nslkdd" | "nsl-kdd" | "nsl_kdd":
            return load_nslkdd()
        case "unsw" | "unsw-nb15" | "unsw_nb15":
            return load_unsw()
        case _:
            raise ValueError(f"unknown dataset: {name!r}")
