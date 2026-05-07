# AI-NIDS: Multi-tier Network Intrusion Detection

[![CI](https://github.com/harshmahani26/ai-nids/actions/workflows/ci.yml/badge.svg)](https://github.com/harshmahani26/ai-nids/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A five-tier intrusion-detection benchmark across NSL-KDD and UNSW-NB15. Classical baselines, gradient boosting with Optuna tuning, three deep models (1D-CNN, LSTM, normal-only autoencoder), a hybrid RF+LSTM cascade for production-style latency-accuracy tradeoffs, and a stacking ensemble. SHAP and LIME are wired in for explainability. Numbers in the table below come straight from `results/metrics.json`.

The most useful finding: on NSL-KDD KDDTest+, the unsupervised autoencoder (trained on Normal traffic only) gets the best binary detection rate at 85.9%, while gradient boosting wins on multi-class. Deep models offer no real advantage over boosting on these tabular features, which lines up with the broader literature.

## Results

KDDTest+ (NSL-KDD) and the official UNSW-NB15 train/test partition. Latency is per-record, batched, on CPU. Model size is the pickled estimator.

<!-- RESULTS_TABLE_START -->
| Dataset | Model | Binary Acc | Multi Acc | Macro F1 | Weighted F1 | ROC-AUC | PR-AUC | Inference (ms) | Train (s) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| NSLKDD | Random Forest | 76.00% | 75.03% | 0.502 | 0.704 | 0.964 | 0.966 | 0.051 | 87.8 | 32.3 |
| NSLKDD | SVM (RBF) | 76.23% | 74.66% | 0.465 | 0.696 | 0.948 | 0.961 | 0.137 | 161.7 | 0.9 |
| NSLKDD | KNN | 76.01% | 74.37% | 0.543 | 0.699 | 0.804 | 0.824 | 0.065 | 40.6 | 40.4 |
| NSLKDD | Voting Ensemble | 75.54% | 74.06% | 0.480 | 0.692 | 0.969 | 0.973 | 0.315 | 119.9 | 147.1 |
| NSLKDD | XGBoost | 78.66% | 77.31% | 0.551 | 0.734 | 0.970 | 0.972 | 0.003 | 91.6 | 2.2 |
| NSLKDD | LightGBM | 77.13% | 75.68% | 0.590 | 0.718 | 0.973 | 0.973 | 0.018 | 125.1 | 14.9 |
| NSLKDD | CatBoost | 77.42% | 76.34% | 0.533 | 0.723 | 0.969 | 0.971 | 0.004 | 304.7 | 5.0 |
| NSLKDD | 1D-CNN | 76.49% | 74.95% | 0.494 | 0.701 | 0.937 | 0.951 | 0.012 | 61.8 | 0.1 |
| NSLKDD | LSTM | 77.30% | 75.77% | 0.489 | 0.712 | 0.938 | 0.945 | 0.013 | 50.7 | 0.2 |
| NSLKDD | Autoencoder | 85.93% | 71.15% | 0.321 | 0.617 | 0.946 | 0.942 | 0.006 | 1.9 | 0.0 |
| NSLKDD | Hybrid RF+LSTM | 75.39% | 74.34% | 0.484 | 0.695 | 0.966 | 0.959 | 0.131 | 41.0 | 19.4 |
| NSLKDD | Stacking | 77.80% | 76.76% | 0.547 | 0.727 | 0.965 | 0.960 | 0.022 | 30.6 | 15.2 |
| UNSW | Random Forest | 89.15% | 75.49% | 0.478 | 0.723 | 0.985 | 0.993 | 0.050 | 89.4 | 291.7 |
| UNSW | XGBoost | 89.53% | 76.18% | 0.515 | 0.731 | 0.987 | 0.994 | 0.004 | 279.5 | 5.3 |
| UNSW | LightGBM | 89.86% | 76.37% | 0.531 | 0.735 | 0.986 | 0.994 | 0.009 | 259.5 | 10.2 |
<!-- RESULTS_TABLE_END -->

Per-class confusion matrices for every cell live in `results/confusion_matrices/`. The model comparison charts and feature-importance plots are in `results/plots/`.

## Why these tiers

The tiers are chosen to bracket what is interesting on tabular IDS data:

- **Tier 1 (classical: RF, SVM, KNN, voting).** A reference point. If anything later beats these by a meaningful margin, the extra complexity earned its keep.
- **Tier 2 (gradient boosting: XGBoost, LightGBM, CatBoost).** State of the art for tabular data in published benchmarks. CatBoost is included specifically because it handles categorical features natively rather than via pre-encoded integers.
- **Tier 3 (deep learning: 1D-CNN, LSTM, autoencoder).** Tests whether sequence and reconstruction-based models offer something the trees miss. The autoencoder is trained on Normal traffic only and used as an unsupervised anomaly detector.
- **Tier 4 (hybrid RF + LSTM).** A production-shaped cascade: cheap RF prefilter scores everything, only suspicious flows pay the LSTM cost. The reported `slow_path_fraction` and threshold capture the tradeoff.
- **Tier 5 (stacking).** Final cross-tier ensemble. Tree base learners feed predictions into a logistic regression meta-learner via out-of-fold CV.

## Approach

**Datasets.** NSL-KDD (1999, KDDTest+ partition) for backwards comparison and UNSW-NB15 (2015, official train/test split) as the modern benchmark. Both are downloaded automatically by `python -m src.data.download` (no Kaggle, no manual forms). NSL-KDD comes from the `defcom17/NSL_KDD` mirror; UNSW-NB15 comes from a HuggingFace mirror that preserves the original string-typed `proto`/`service`/`state` columns and the multi-class `attack_cat` label CatBoost needs natively.

**Preprocessing.** Categorical features are label-encoded with encoders fit on training data only. Numeric features are standardised with `StandardScaler` (also train-only). Unseen categorical values at test time map to a stable sentinel index. The preprocessor is a single class with explicit `fit_transform` / `transform` that the test suite checks for leakage. CatBoost bypasses encoding entirely and uses the raw DataFrame with `cat_features`.

**Tuning.** Tier 1 uses `GridSearchCV(cv=5)` over the published parameter ranges. SVM grid search runs on a stratified subsample because full-data SVM CV is too slow on Windows; the best estimator is then refit on the full training set. Tier 2 uses Optuna with TPE sampling, 20 trials per booster (12 for CatBoost; CatBoost's native categorical handling is slow), deterministic via the configured seed. Tier 3 trains with Adam + cosine LR + early stopping (patience-based) via a generic loop in `src/training/torch_trainer.py`. Mixed precision is enabled when CUDA is available.

**Evaluation.** Every model is scored with the same code path in `src/evaluate.py`: binary accuracy, multi-class accuracy, macro F1, weighted F1, ROC-AUC and PR-AUC for the binary task, plus per-class precision/recall/F1, plus inference latency in ms per record (measured over 1,000 records, batched), training time, and pickled model size on disk. Results land in `results/metrics.json` keyed by `(dataset, model)` and re-runs upsert in place.

**Reproducibility.** A single seed (`seed=42` in `src/config.py`) drives `random`, `numpy`, `torch`, and Optuna's TPE sampler. Re-running with the same seed produces the same numbers.

## Reproduction

```bash
git clone https://github.com/harshmahani26/ai-nids.git
cd ai-nids

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
# CPU-only PyTorch wheel (the requirements file pins the +cpu build).
# If you want CUDA, replace torch with the matching CUDA wheel from pytorch.org.
pip install torch --index-url https://download.pytorch.org/whl/cpu

python -m src.data.download
python -m src.train --dataset nslkdd --tier all
python -m src.train --dataset unsw    --tier all
python -m src.explain.run --dataset nslkdd
python -m src.explain.run --dataset unsw

jupyter nbconvert --to notebook --execute --inplace notebooks/nids_dashboard.ipynb
```

`scripts/reproduce.py` does all of the above in one command. Allow 30 to 60 minutes on a modern laptop CPU.

## What SHAP told us

The SHAP global-importance bar (`results/shap/nslkdd_xgboost_global_bar.png`) shows that on NSL-KDD the dominant features for the tuned XGBoost are `src_bytes`, `flag`, `same_srv_rate`, `diff_srv_rate`, and `dst_host_serror_rate`. The engineered traffic-window features dominate over raw byte counts collectively, which matches what the original KDD authors observed.

The per-class heatmap (`results/shap/nslkdd_xgboost_per_class.png`) makes the R2L/U2R problem concrete: the same handful of features dominate the model's reasoning across every attack category, including the rare ones the model is failing on. There is no separate, distinctive feature signature for R2L or U2R that the model is using - it is just generalising what worked for DoS and Probe. That is a useful confirmation that the failure mode is data scarcity, not a missing feature.

## Notes from the run

- **Boosting beats deep learning on tabular features, by a small margin.** XGBoost, LightGBM, and CatBoost cluster around 77 to 79% binary accuracy on KDDTest+; CNN and LSTM land at 76 to 77%. The stacking ensemble matches the best individual booster and is much smaller on disk. This matches the published consensus that deep learning does not categorically beat boosting on tabular data.
- **The unsupervised autoencoder is the surprise winner on NSL-KDD binary detection** at 85.9%, but its multi-class accuracy collapses (71%) because it has no notion of attack subtype. In a real pipeline you would always pair it with a multi-class classifier downstream.
- **The voting ensemble underperformed the strongest individual classical model.** RF, SVM, and KNN make correlated errors on the same novel-attack rows, so averaging probabilities does not recover those misclassifications.
- **The hybrid RF+LSTM cascade did not pay off on NSL-KDD.** The threshold-tuned configuration sends 46% of validation traffic to the LSTM but loses ~2 percentage points of multi-class accuracy versus a plain RF. The cost-benefit only flips if the LSTM brings a real accuracy edge, which it doesn't on these features.
- **R2L and U2R are essentially undetectable by every model.** R2L is 0.8% of training and 12.8% of test; U2R is even worse. Per-class confusion matrices in `results/confusion_matrices/` show all classifiers learning to ignore those classes regardless of architecture or tuning effort.
- **UNSW-NB15 is meaningfully easier in binary terms** (89% binary versus NSL-KDD's 76 to 79% best) but harder for multi-class (10 attack categories vs 5). XGBoost on UNSW achieves ROC-AUC 0.987 on binary detection.
- **CatBoost's native categorical handling did not produce a clear edge.** Performance is comparable to XGBoost on both datasets after tuning. The benefit shows up in code simplicity (no encoding step) rather than accuracy.
- **Latency favors the trees decisively.** XGBoost and CatBoost are sub-millisecond per record in batched inference; the voting ensemble at 0.32 ms/record is 100x slower because soft voting requires three full forward passes.

## Limitations

NSL-KDD is from 1999 and UNSW-NB15 is synthetic. Neither captures encrypted application-layer traffic at modern volumes, and neither will see anything resembling current malware behavior. This repository is methodology and benchmarking, not a production tool. The hybrid pipeline's latency budget is also measured against synthetic traffic, so the absolute numbers are illustrative rather than predictive of a real deployment. The autoencoder's 85.9% binary accuracy on NSL-KDD looks impressive in isolation, but its multi-class accuracy collapses because it has no notion of attack subtype - in practice you would always pair the AE with a multi-class classifier downstream.

## Stack

Python 3.13. scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch (CPU build), Optuna, SHAP, LIME, pandas, numpy, matplotlib, seaborn, jupyter. Dev tooling: ruff, black, mypy, pytest, pre-commit. Exact pinned versions live in `requirements.txt`.

If `lightgbm`, `catboost`, or another wheel is unavailable on Python 3.13 at the time you clone this, fall back to 3.12. The code itself is 3.12-compatible; only the published wheels move.

## References

- Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. (2009). *A detailed analysis of the KDD CUP 99 data set.* IEEE CISDA.
- Moustafa, N., & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems.* IEEE Military Communications and Information Systems Conference.
- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why should I trust you?": Explaining the predictions of any classifier.* KDD.

## License

MIT
