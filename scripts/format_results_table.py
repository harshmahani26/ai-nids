"""Render the README results table from results/metrics.json.

Usage::

    python scripts/format_results_table.py > /tmp/table.md

Pipe the output into the README between the result-table markers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def fmt_f(v: float | None, digits: int = 3) -> str:
    return f"{v:.{digits}f}" if v is not None else "n/a"


def render() -> str:
    with (ROOT / "results" / "metrics.json").open() as f:
        store = json.load(f)

    if not store:
        return "_No metrics yet - run `python -m src.train --dataset nslkdd --tier all`._"

    # Each row is one (dataset, model) pair.
    headers = [
        "Dataset",
        "Model",
        "Binary Acc",
        "Multi Acc",
        "Macro F1",
        "Weighted F1",
        "ROC-AUC",
        "PR-AUC",
        "Inf batched (ms)",
        "Inf single (ms)",
        "Train (s)",
        "Size (MB)",
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")

    # Display order: nslkdd first, then unsw; classical -> boosting -> deep -> hybrid -> stacking.
    tier_order = [
        "Random Forest",
        "SVM (RBF)",
        "KNN",
        "Voting Ensemble",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "1D-CNN",
        "LSTM",
        "Autoencoder",
        "Hybrid RF+LSTM",
        "Stacking",
    ]

    for ds in ("nslkdd", "unsw"):
        bucket = store.get(ds, {})
        # Sort by tier order, then alphabetically.
        for model in sorted(
            bucket, key=lambda m: (tier_order.index(m) if m in tier_order else 999, m)
        ):
            v = bucket[model]
            lines.append(
                "| "
                + " | ".join(
                    [
                        ds.upper(),
                        model,
                        fmt_pct(v["binary_accuracy"]),
                        fmt_pct(v["multi_accuracy"]),
                        fmt_f(v["macro_f1"]),
                        fmt_f(v["weighted_f1"]),
                        fmt_f(v.get("roc_auc")),
                        fmt_f(v.get("pr_auc")),
                        fmt_f(v["inference_ms_per_record"]),
                        fmt_f(v.get("inference_ms_per_record_unbatched", 0.0)),
                        fmt_f(v["train_time_s"], 1),
                        fmt_f(v["model_size_mb"], 1),
                    ]
                )
                + " |"
            )

    return "\n".join(lines)


if __name__ == "__main__":
    print(render())
