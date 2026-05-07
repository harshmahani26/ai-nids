"""Tier 3: Deep learning baselines (1D-CNN, LSTM, Autoencoder).

All three exposing a sklearn-compatible ``predict`` so :mod:`src.evaluate` can
score them with the same code path as everything else.

LSTM windowing strategy
-----------------------
NSL-KDD and UNSW-NB15 lack a session/source-IP grouping column at the row level
suitable for hierarchical sequence construction. We approximate sequential
context by sliding fixed-length windows over the input matrix in original row
order. Each window is a ``(window, feature)`` tensor and the LSTM predicts the
class of the *last* row in the window. This is a standard reduction in IDS
literature when raw flow grouping is not available.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from src.config import CONFIG
from src.data.preprocess import PreprocessedSplit
from src.training.torch_trainer import (
    TrainHistory,
    device_summary,
    get_device,
    make_loader,
    seed_torch,
    train_loop,
)

log = logging.getLogger(__name__)


# ---------- 1D-CNN ---------------------------------------------------------


class CNN1D(nn.Module):
    def __init__(self, n_features: int, n_classes: int, channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels * 2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(channels * 2, n_classes)
        self.n_features = n_features
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) -> (B, 1, F)
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class _TorchClassifier:
    """sklearn-style wrapper around a trained nn.Module."""

    def __init__(self, model: nn.Module, device: torch.device, n_classes: int) -> None:
        self.model = model
        self.device = device
        self.n_classes = n_classes
        self.history: TrainHistory | None = None

    def _forward(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        loader = make_loader(X, batch_size=CONFIG.train.deep_batch_size, shuffle=False)
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                out = self.model(xb)
                outs.append(torch.softmax(out, dim=1).cpu().numpy())
        return np.concatenate(outs, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)


def train_cnn(split: PreprocessedSplit) -> _TorchClassifier:
    """Train and return a 1D-CNN classifier."""
    seed_torch(CONFIG.train.seed)
    device = get_device()
    log.info("CNN device: %s", device_summary())

    val_size = max(int(0.15 * len(split.X_train)), 1024)
    val_idx = np.random.RandomState(CONFIG.train.seed).permutation(len(split.X_train))[:val_size]
    val_mask = np.zeros(len(split.X_train), dtype=bool)
    val_mask[val_idx] = True
    Xtr, ytr = split.X_train[~val_mask], split.y_train_multi[~val_mask]
    Xv, yv = split.X_train[val_mask], split.y_train_multi[val_mask]

    train_loader = make_loader(Xtr, ytr, batch_size=CONFIG.train.deep_batch_size, shuffle=True)
    val_loader = make_loader(Xv, yv, batch_size=CONFIG.train.deep_batch_size, shuffle=False)

    model = CNN1D(n_features=split.X_train.shape[1], n_classes=split.meta.num_classes)
    history = train_loop(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        epochs=CONFIG.train.deep_epochs,
        lr=CONFIG.train.deep_lr,
        patience=CONFIG.train.deep_patience,
        device=device,
        checkpoint_path=CONFIG.paths.checkpoints / f"cnn_{split.meta.name}.pt",
        use_amp=device.type == "cuda",
        tb_run_name=f"cnn_{split.meta.name}",
    )
    log.info("CNN trained: best_epoch=%d val=%.4f", history.best_epoch, history.best_val_loss)
    clf = _TorchClassifier(model, device, split.meta.num_classes)
    clf.history = history
    return clf


# ---------- LSTM with rolling-window context -------------------------------


class LSTMNet(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # final-step representation
        return self.fc(self.dropout(out))


def _windowed(
    X: np.ndarray, y: np.ndarray | None, *, window: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build sliding windows of length ``window``. Pads with zeros at the start."""
    n, f = X.shape
    pad = np.zeros((window - 1, f), dtype=X.dtype)
    Xp = np.concatenate([pad, X], axis=0)
    Xw = np.lib.stride_tricks.sliding_window_view(Xp, window_shape=window, axis=0)
    # Result shape: (n, F, window). Reorder to (n, window, F).
    Xw = Xw.transpose(0, 2, 1).copy()
    return Xw, y


class _LSTMClassifier(_TorchClassifier):
    """Wraps the LSTM so predict() materialises the windows on demand."""

    def __init__(self, model: nn.Module, device: torch.device, n_classes: int, window: int) -> None:
        super().__init__(model, device, n_classes)
        self.window = window

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xw, _ = _windowed(X, None, window=self.window)
        return super().predict_proba(Xw)


def train_lstm(split: PreprocessedSplit, window: int = 8) -> _LSTMClassifier:
    """Train and return an LSTM classifier over rolling windows."""
    seed_torch(CONFIG.train.seed)
    device = get_device()
    log.info("LSTM device: %s window=%d", device_summary(), window)

    Xw, yw = _windowed(split.X_train, split.y_train_multi, window=window)
    assert yw is not None  # we passed y in, so the helper returned it

    val_size = max(int(0.15 * len(Xw)), 1024)
    val_idx = np.random.RandomState(CONFIG.train.seed).permutation(len(Xw))[:val_size]
    val_mask = np.zeros(len(Xw), dtype=bool)
    val_mask[val_idx] = True
    Xtr, ytr = Xw[~val_mask], yw[~val_mask]
    Xv, yv = Xw[val_mask], yw[val_mask]

    train_loader = make_loader(Xtr, ytr, batch_size=CONFIG.train.deep_batch_size, shuffle=True)
    val_loader = make_loader(Xv, yv, batch_size=CONFIG.train.deep_batch_size, shuffle=False)

    model = LSTMNet(n_features=split.X_train.shape[1], n_classes=split.meta.num_classes)
    history = train_loop(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        epochs=CONFIG.train.deep_epochs,
        lr=CONFIG.train.deep_lr,
        patience=CONFIG.train.deep_patience,
        device=device,
        checkpoint_path=CONFIG.paths.checkpoints / f"lstm_{split.meta.name}.pt",
        use_amp=device.type == "cuda",
        tb_run_name=f"lstm_{split.meta.name}",
    )
    log.info("LSTM trained: best_epoch=%d val=%.4f", history.best_epoch, history.best_val_loss)
    clf = _LSTMClassifier(model, device, split.meta.num_classes, window=window)
    clf.history = history
    return clf


# ---------- Autoencoder anomaly detector -----------------------------------


class _AutoencoderModule(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, hidden),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderDetector:
    """Train on Normal-only traffic; flag rows with high reconstruction error.

    Threshold is chosen on a validation split to maximise binary F1.
    Multi-class predictions are degenerate (only ``normal_index`` vs not)
    because an AE has no notion of attack subtype.
    """

    def __init__(self, hidden: int = 32) -> None:
        self.hidden = hidden
        self.model: _AutoencoderModule | None = None
        self.device: torch.device | None = None
        self.threshold_: float = 0.0
        self.normal_index_: int = 0
        self.history: TrainHistory | None = None

    def _scores(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None and self.device is not None
        self.model.eval()
        loader = make_loader(X, batch_size=CONFIG.train.deep_batch_size, shuffle=False)
        out: list[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                rec = self.model(xb)
                err = ((xb - rec) ** 2).mean(dim=1)
                out.append(err.cpu().numpy())
        return np.concatenate(out, axis=0)

    def fit(self, split: PreprocessedSplit) -> AutoencoderDetector:
        from sklearn.metrics import f1_score

        seed_torch(CONFIG.train.seed)
        self.device = get_device()
        self.normal_index_ = split.meta.normal_index
        log.info("Autoencoder device: %s", device_summary())

        normal_mask = split.y_train_multi == self.normal_index_
        Xn = split.X_train[normal_mask]
        log.info("AE training on %d normal-only samples", len(Xn))
        # 90/10 split for early stopping.
        n_val = max(int(0.1 * len(Xn)), 512)
        rng = np.random.RandomState(CONFIG.train.seed)
        idx = rng.permutation(len(Xn))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xn_tr, Xn_val = Xn[tr_idx], Xn[val_idx]

        # AE loss is MSE(reconstruction, input) so we keep our own training
        # loop locally rather than reusing train_loop (which expects classifier
        # outputs and integer targets).
        self.model = _AutoencoderModule(n_features=split.X_train.shape[1], hidden=self.hidden)
        self._fit_local(Xn_tr, Xn_val)

        # Choose threshold on a held-out mixed validation split.
        rng = np.random.RandomState(CONFIG.train.seed)
        sample_idx = rng.permutation(len(split.X_train))[: max(20_000, len(split.X_train) // 10)]
        Xv = split.X_train[sample_idx]
        yv_bin = (split.y_train_multi[sample_idx] != self.normal_index_).astype(np.int64)
        scores = self._scores(Xv)

        candidates = np.quantile(scores, np.linspace(0.5, 0.99, 50))
        best_f1, best_thr = -1.0, float(np.median(scores))
        for thr in candidates:
            preds = (scores > thr).astype(np.int64)
            f1 = f1_score(yv_bin, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        self.threshold_ = best_thr
        log.info("AE threshold=%.6f val_binary_f1=%.4f", best_thr, best_f1)
        return self

    def _fit_local(self, X_tr: np.ndarray, X_val: np.ndarray) -> None:
        assert self.model is not None and self.device is not None
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=CONFIG.train.deep_lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        Xtr_t = torch.as_tensor(X_tr, dtype=torch.float32)
        Xv_t = torch.as_tensor(X_val, dtype=torch.float32)
        bs = CONFIG.train.deep_batch_size

        best_val = float("inf")
        best_epoch = 0
        bad = 0
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_state = None
        for epoch in range(1, CONFIG.train.deep_epochs + 1):
            self.model.train()
            perm = torch.randperm(len(Xtr_t))
            running = 0.0
            for i in range(0, len(perm), bs):
                idx = perm[i : i + bs]
                xb = Xtr_t[idx].to(self.device)
                opt.zero_grad(set_to_none=True)
                rec = self.model(xb)
                loss = criterion(rec, xb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
            train_losses.append(running / len(Xtr_t))

            self.model.eval()
            with torch.no_grad():
                v_running = 0.0
                for i in range(0, len(Xv_t), bs):
                    xb = Xv_t[i : i + bs].to(self.device)
                    rec = self.model(xb)
                    v_running += criterion(rec, xb).item() * xb.size(0)
                v = v_running / len(Xv_t)
                val_losses.append(v)
            if v < best_val - 1e-6:
                best_val = v
                best_epoch = epoch
                bad = 0
                best_state = {
                    k: t.detach().cpu().clone() for k, t in self.model.state_dict().items()
                }
            else:
                bad += 1
            if bad >= CONFIG.train.deep_patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.history = TrainHistory(
            train_losses=train_losses,
            val_losses=val_losses,
            best_val_loss=best_val,
            best_epoch=best_epoch,
            epochs_run=len(train_losses),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._scores(X)
        is_attack = scores > self.threshold_
        # Map: normal -> normal_index_, attack -> next-most-frequent class.
        # We don't know subtypes, so put all attacks in class 1 (or
        # whatever's not the normal index).
        attack_class = 1 if self.normal_index_ != 1 else 2
        preds = np.full(len(X), self.normal_index_, dtype=np.int64)
        preds[is_attack] = attack_class
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return proba derived from reconstruction error vs threshold."""
        from src.data.loader import DatasetMeta  # noqa: F401  (type-only import marker)

        scores = self._scores(X)
        # min-max in a way that makes large scores -> high attack proba.
        s_min, s_max = scores.min(), scores.max()
        attack_p = np.zeros_like(scores) if s_max <= s_min else (scores - s_min) / (s_max - s_min)
        # We need a proba matrix shaped (N, n_classes). The model only knows
        # "normal vs attack", so distribute attack mass into class 1
        # (or 2 if normal is 1).
        n_classes = max(self.normal_index_ + 1, 2)
        return self._build_proba_matrix(attack_p, n_classes)

    def _build_proba_matrix(self, attack_p: np.ndarray, n_classes: int) -> np.ndarray:
        # We don't know the meta here so default to 2 classes -> if caller wants
        # multi-class, evaluate.evaluate_classifier handles either width.
        # Caller may set the number of meta classes via wrap_for_meta.
        n_classes = max(n_classes, 2)
        out = np.zeros((len(attack_p), n_classes), dtype=np.float64)
        out[:, self.normal_index_] = 1.0 - attack_p
        attack_class = 1 if self.normal_index_ != 1 else 0
        # Spread attack_p over a single dummy attack class.
        if attack_class == self.normal_index_:
            attack_class = (self.normal_index_ + 1) % n_classes
        out[:, attack_class] = attack_p
        return out
