# Encoding Shared — Ridge Regression Utilities

Provides cross-validation, ridge fitting, and compression utilities reused across encoding analyses.

## Modules

| Module | Purpose |
|--------|---------|
| `fold_evaluator.py` | Leave-one-run-out cross-validation, standardization, and banded ridge regression with himalaya; imported by `encoding.py`, `incremental_av.py`, `roi_av_profile.py` |
| `compression.py` | Dimensionality reduction (PCA, random projection, cluster means, within-cluster PC1) with fit/test split; imported by `incremental_av.py` for compression-efficiency comparisons |
| `encoding_utils.py` | fMRI and embedding array construction, run-level indexing, model loading; shared by all encoding scripts |
