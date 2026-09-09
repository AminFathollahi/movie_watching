# RSA Shared — Representational Similarity Utilities

Provides RDM construction, embedding processing, residualization, and model registry shared across RSA analyses.

## Modules

| Module | Purpose |
|--------|---------|
| `rsa_utils.py` | fMRI extraction, RDM computation (Spearman/Pearson), embedding preprocessing, pairwise distance matrices |
| `residuals.py` | Linear and projection residualization of embeddings for confound control; imported by residualized_maps.py and run_extended_analyses.sh |
| `model_registry.py` | Central registry of model names, embedding paths, and availability checks; shared by searchlight.py, glasser.py, encoding.py |
