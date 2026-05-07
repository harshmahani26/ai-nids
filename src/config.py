"""Project-wide configuration as immutable dataclasses."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data_raw: Path = ROOT / "data" / "raw"
    data_processed: Path = ROOT / "data" / "processed"
    results: Path = ROOT / "results"
    confusion_matrices: Path = ROOT / "results" / "confusion_matrices"
    shap: Path = ROOT / "results" / "shap"
    lime: Path = ROOT / "results" / "lime_examples"
    checkpoints: Path = ROOT / "results" / "checkpoints"
    plots: Path = ROOT / "results" / "plots"
    metrics_json: Path = ROOT / "results" / "metrics.json"

    def ensure(self) -> None:
        """Create all output directories."""
        for p in (
            self.data_raw,
            self.data_processed,
            self.results,
            self.confusion_matrices,
            self.shap,
            self.lime,
            self.checkpoints,
            self.plots,
        ):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    cv_folds: int = 5
    optuna_trials: int = 20
    svm_subsample: int = 20_000
    deep_epochs: int = 15
    deep_batch_size: int = 1024
    deep_patience: int = 3
    deep_lr: float = 1e-3
    hybrid_threshold_search: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


@dataclass(frozen=True)
class DataConfig:
    nslkdd_train: str = "KDDTrain+.txt"
    nslkdd_test: str = "KDDTest+.txt"
    unsw_train: str = "UNSW_NB15_training-set.csv"
    unsw_test: str = "UNSW_NB15_testing-set.csv"
    nslkdd_url_base: str = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master"
    unsw_url_base: str = (
        "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2F&files="
    )


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


CONFIG = Config()


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger and return a project logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("ai-nids")
