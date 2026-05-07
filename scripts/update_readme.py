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

The per-class heatmap (`results/shap/nslkdd_xgboost_per_class.png`) makes the R2L/U2R problem concrete: the same handful of features dominate the model's reasoning across every attack category, including the rare ones the model is failing on. There is no separate, distinctive feature signature for R2L or U2R that the model is using - it is just generalising what worked for DoS and Probe. That is a useful confirmation that the failure mode is data scarcity, not a missing feature."""


NOTES_FROM_RUN = """- **Boosting beats deep learning on tabular features, by a small margin.** XGBoost, LightGBM, and CatBoost cluster around 77 to 79% binary accuracy on KDDTest+; CNN and LSTM land at 76 to 77%. The stacking ensemble matches the best individual booster and is much smaller on disk. This matches the published consensus that deep learning does not categorically beat boosting on tabular data.
- **The unsupervised autoencoder is the surprise winner on NSL-KDD binary detection** at 85.9%, but its multi-class accuracy collapses (71%) because it has no notion of attack subtype. In a real pipeline you would always pair it with a multi-class classifier downstream.
- **The voting ensemble underperformed the strongest individual classical model.** RF, SVM, and KNN make correlated errors on the same novel-attack rows, so averaging probabilities does not recover those misclassifications.
- **The hybrid RF+LSTM cascade did not pay off on NSL-KDD.** The threshold-tuned configuration sends 46% of validation traffic to the LSTM but loses ~2 percentage points of multi-class accuracy versus a plain RF. The cost-benefit only flips if the LSTM brings a real accuracy edge, which it doesn't on these features.
- **R2L and U2R are essentially undetectable by every model.** R2L is 0.8% of training and 12.8% of test; U2R is even worse. Per-class confusion matrices in `results/confusion_matrices/` show all classifiers learning to ignore those classes regardless of architecture or tuning effort.
- **UNSW-NB15 is meaningfully easier in binary terms** (89% binary versus NSL-KDD's 76 to 79% best) but harder for multi-class (10 attack categories vs 5). XGBoost on UNSW achieves ROC-AUC 0.987 on binary detection.
- **CatBoost's native categorical handling did not produce a clear edge.** Performance is comparable to XGBoost on both datasets after tuning. The benefit shows up in code simplicity (no encoding step) rather than accuracy.
- **Latency favors the trees decisively.** XGBoost and CatBoost are sub-millisecond per record in batched inference; the voting ensemble at 0.32 ms/record is 100x slower because soft voting requires three full forward passes."""


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
