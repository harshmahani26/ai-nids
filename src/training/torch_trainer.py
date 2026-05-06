"""Generic PyTorch training loop with early stopping and reproducibility."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainHistory:
    train_losses: list[float]
    val_losses: list[float]
    best_val_loss: float
    best_epoch: int
    epochs_run: int


def make_loader(
    X: np.ndarray, y: np.ndarray | None = None, *, batch_size: int, shuffle: bool
) -> DataLoader:
    X_t = torch.as_tensor(X, dtype=torch.float32)
    if y is None:
        ds = TensorDataset(X_t)
    else:
        y_t = torch.as_tensor(y, dtype=torch.long)
        ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_loop(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    epochs: int,
    lr: float,
    patience: int,
    device: torch.device,
    checkpoint_path: Path | None = None,
    weight_decay: float = 1e-5,
    use_amp: bool = False,
) -> TrainHistory:
    """Train ``model`` with Adam + cosine LR + early stopping; returns history."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            xb = batch[0].to(device, non_blocking=True)
            yb = batch[1].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(xb)
                loss = criterion(out, yb)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_losses.append(running / max(n, 1))

        # Validation
        model.eval()
        v_running = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                xb = batch[0].to(device, non_blocking=True)
                yb = batch[1].to(device, non_blocking=True)
                out = model(xb)
                loss = criterion(out, yb)
                v_running += loss.item() * xb.size(0)
                vn += xb.size(0)
        v = v_running / max(vn, 1)
        val_losses.append(v)
        scheduler.step()

        if v < best_val - 1e-6:
            best_val = v
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            bad_epochs += 1

        log.debug(
            "epoch %d/%d  train=%.4f  val=%.4f  best=%.4f",
            epoch,
            epochs,
            train_losses[-1],
            v,
            best_val,
        )
        if bad_epochs >= patience:
            log.info("early stop at epoch %d (best=%d, val=%.4f)", epoch, best_epoch, best_val)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainHistory(
        train_losses=train_losses,
        val_losses=val_losses,
        best_val_loss=best_val,
        best_epoch=best_epoch,
        epochs_run=len(train_losses),
    )


def device_summary() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "cpu"
