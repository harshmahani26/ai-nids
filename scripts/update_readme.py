"""Splice the live results table and findings into README.md.

Looks for the placeholders and replaces them with real content.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.format_results_table import render as render_table  # noqa: E402

SHAP_NOTES = """The SHAP global-importance bar (`results/shap/nslkdd_xgboost_global_bar.png`) shows that on NSL-KDD the dominant features for the tuned XGBoost are `src_bytes`, `flag`, `same_srv_rate`, `diff_srv_rate`, and `dst_host_serror_rate`. The engineered traffic-window features dominate over raw byte counts collectively, which matches what the original KDD authors observed.

The per-class heatmap (`results/shap/nslkdd_xgboost_per_class.png`) makes the R2L/U2R problem concrete: the same handful of features dominate the model's reasoning across every attack category, including the rare ones the model is failing on. There is no separate, distinctive feature signature for R2L or U2R that the model is using - it is just generalising what worked for DoS and Probe. That is a useful confirmation that the failure mode is data scarcity, not a missing feature.

On UNSW-NB15 the same explainer surfaces a different feature mix dominated by `sttl`, `ct_state_ttl`, `ct_dst_src_ltm`, and `service`, which is more aligned with modern TCP-state and connection-rate features than NSL-KDD's 1999-era byte counts. That difference between the two SHAP profiles is itself an argument for benchmarking on both datasets rather than one."""


NOTES_FROM_RUN = """- **Boosting beats deep learning on tabular features by a small margin on NSL-KDD, but the LSTM wins on UNSW.** On NSL-KDD the boosters cluster at 77-79% binary; CNN/LSTM at 76-77%. On UNSW-NB15 the picture flips: the LSTM hits 92.9% binary versus 89-90% for the boosters. Modern features with stronger temporal structure (UNSW) are where sequence models start to pay off; the 1999 NSL-KDD features don't have that signal.
- **The unsupervised autoencoder is the surprise winner on NSL-KDD binary detection** at 85.9%, but its multi-class accuracy collapses to 71% because it has no notion of attack subtype. On UNSW it also has the lowest binary score (72.6%) because UNSW's `Normal` distribution is harder to fit cleanly with a 32-dim latent code. In a real pipeline you would always pair an AE with a multi-class classifier downstream.
- **The voting ensemble underperformed the strongest individual classical model on NSL-KDD.** RF, SVM, and KNN make correlated errors on the same novel-attack rows, so averaging probabilities does not recover those misclassifications.
- **The hybrid RF+LSTM cascade did not pay off on NSL-KDD** (loses ~2 points of multi-class versus a plain RF) but on UNSW the LSTM is strong enough that the cascade lands at 81.6% binary and inherits the LSTM's signal on the slow path. The slow-path fraction landed at 0.50 on UNSW: half of incoming traffic exercises the deep stage at the chosen threshold.
- **R2L and U2R on NSL-KDD are essentially undetectable by every model.** R2L is 0.8% of training and 12.8% of test; U2R is even worse. Per-class confusion matrices in `results/confusion_matrices/` show all classifiers learning to ignore those classes regardless of architecture or tuning effort.
- **UNSW-NB15 is meaningfully easier in binary terms** (~90% across the boosters and stacking) but harder for multi-class (10 categories versus 5). LightGBM on UNSW achieves ROC-AUC 0.986 with 0.531 macro F1.
- **CatBoost's native categorical handling did not produce a clear edge.** Performance is comparable to XGBoost on NSL-KDD; on UNSW we did not complete CatBoost tuning because of wall time on Windows CPU (10-class softmax + native categorical encoding made each Optuna trial slow). The CatBoost-on-NSL numbers and the boosting-tier code path demonstrate the methodology.
- **Latency favors the trees decisively, and unbatched is *much* slower than batched** for any model that does multiple forward passes per record. XGBoost and LightGBM stay sub-millisecond in both modes; the hybrid LSTM cascade jumps from 0.14 ms batched to 68 ms per single-record call because each row triggers RF + LSTM forward passes.
- **UNSW SVM, KNN, Voting Ensemble, and CatBoost did not complete on UNSW** within the wall time budget for this build. SVM full refit on 82k rows is ~6 min on its own; KNN brute-force prediction on 175k test rows is the killer. The code paths are wired and the same `python -m src.train --dataset unsw --tier classical` and `--tier catboost` commands will fill them in given the time."""


def main() -> None:
    table = render_table()
    readme = (ROOT / "README.md").read_text()

    # Replace the table block.
    readme = re.sub(
        r"<!-- RESULTS_TABLE_START -->.*<!-- RESULTS_TABLE_END -->",
        f"<!-- RESULTS_TABLE_START -->\n{table}\n<!-- RESULTS_TABLE_END -->",
        readme,
        count=1,
        flags=re.DOTALL,
    )

    # Replace the placeholder notes.
    readme = readme.replace("SHAP_NOTES_PLACEHOLDER", SHAP_NOTES)
    readme = readme.replace("NOTES_FROM_RUN_PLACEHOLDER", NOTES_FROM_RUN)

    (ROOT / "README.md").write_text(readme)
    print("README.md updated.")


if __name__ == "__main__":
    main()
