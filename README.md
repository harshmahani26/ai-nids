# AI-NIDS: Multi-tier Network Intrusion Detection

[![CI](https://github.com/harshmahani26/ai-nids/actions/workflows/ci.yml/badge.svg)](https://github.com/harshmahani26/ai-nids/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A five-tier intrusion-detection benchmark across NSL-KDD and UNSW-NB15. Classical baselines, gradient boosting with Optuna tuning, deep learning (1D-CNN, LSTM, autoencoder), a hybrid RF+LSTM cascade for production-style latency-accuracy tradeoffs, and a stacking ensemble. SHAP and LIME are wired in for explainability. Numbers in the table below come straight from `results/metrics.json`.

## Results

Numbers will be inserted here after the full reproduction run finishes. Until then, the canonical store is `results/metrics.json` and the executed notebook in `notebooks/nids_dashboard.ipynb`.

## Why these tiers

The tiers are chosen to bracket what is interesting on tabular IDS data:

- **Tier 1 (classical: RF, SVM, KNN, voting).** A reference point. If anything later beats these by a meaningful margin, the extra complexity earned its keep.
- **Tier 2 (gradient boosting: XGBoost, LightGBM, CatBoost).** State of the art for tabular data in published benchmarks. CatBoost is included specifically because it handles categorical features natively rather than via pre-encoded integers.
- **Tier 3 (deep learning: 1D-CNN, LSTM, autoencoder).** Tests whether sequence and reconstruction-based models offer something the trees miss. The autoencoder is trained on Normal traffic only and used as an unsupervised anomaly detector.
- **Tier 4 (hybrid RF + LSTM).** A production-shaped cascade: cheap RF prefilter scores everything, only suspicious flows pay the LSTM cost. The reported `slow_path_fraction` captures the tradeoff.
- **Tier 5 (stacking).** Final cross-tier ensemble. Tree base learners feed predictions into a logistic regression meta-learner via 5-fold out-of-fold CV.

## Approach

**Datasets.** NSL-KDD (1999) for backwards comparison and UNSW-NB15 (2015) as the modern benchmark. Both are downloaded automatically by `python -m src.data.download` (no Kaggle, no manual forms).

**Preprocessing.** Categorical features are label-encoded with encoders fit on training data only. Numeric features are standardised with `StandardScaler` (also train-only). Unseen categorical values at test time map to a stable sentinel index. The preprocessor is a single class with explicit `fit_transform` / `transform` that the test suite checks for leakage. CatBoost bypasses encoding entirely and uses the raw DataFrame with `cat_features`.

**Tuning.** Tier 1 uses `GridSearchCV(cv=5)` over the published parameter ranges. Tier 2 uses Optuna with TPE sampling, 25 trials per model, deterministic via the configured seed. Tier 3 trains with Adam + cosine LR + early stopping (patience-based) via a generic loop in `src/training/torch_trainer.py`. Mixed precision is enabled when CUDA is available.

**Evaluation.** Every model is scored with the same code path in `src/evaluate.py`: binary accuracy, multi-class accuracy, macro F1, weighted F1, ROC-AUC and PR-AUC for the binary task, plus per-class precision/recall/F1, plus inference latency in ms per record (measured over 1,000 records), training time, and pickled model size on disk. Results land in `results/metrics.json` keyed by `(dataset, model)`.

**Reproducibility.** A single seed (configured in `src/config.py`) drives `random`, `numpy`, `torch`, and Optuna. Re-running with the same seed produces the same numbers.

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
# CPU-only PyTorch wheel if your default index does not have one.
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

Filled in with the actual finding once the SHAP run completes.

## Notes from the run

Bullets are populated after the reproduction finishes so they reflect the actual numbers, not placeholders.

## Limitations

NSL-KDD is from 1999 and UNSW-NB15 is synthetic. Neither captures encrypted application-layer traffic at modern volumes, and neither will see anything resembling current malware behavior. This repository is methodology and benchmarking, not a production tool. The hybrid pipeline's latency budget is also measured against synthetic traffic, so the absolute numbers are illustrative rather than predictive of a real deployment.

## Stack

Python 3.13 (3.12 also supported). scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch (CPU build), Optuna, SHAP, LIME, pandas, numpy, matplotlib, seaborn, jupyter. Dev tooling: ruff, black, mypy, pytest, pre-commit. Exact pinned versions live in `requirements.txt`.

## References

- Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. (2009). *A detailed analysis of the KDD CUP 99 data set.* IEEE CISDA.
- Moustafa, N., & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems.* IEEE Military Communications and Information Systems Conference.
- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.

## License

MIT
