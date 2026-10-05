# encoding/shared

Library code behind `encoding/variance_partition.py`, imported as `encoding.shared.*` from the repository root. The method is described in `encoding/README.md`.

## `splits.py`

Inputs, normalization and cross-validation folds.

| Name | Purpose |
|---|---|
| `build_parser()` | Command-line options of `variance_partition.py` |
| `sample_metadata(timing, bin_sec, skip_sec)` | One row per window (clip, run, window index, onset); a clip of duration D has floor((D − bin) / skip) + 1 windows |
| `load_inputs(args)` | Loads A, V and J, builds the response matrix, checks that embedding and response rows and run boundaries agree, and applies per-run (`run`) or per-clip (`clip`) scaling to responses and embeddings over the same rows |
| `make_folds(data, split)` | Training and held-out masks for `fixed`, `loro` and `loco` |
| `check_scalings(args)` | Refuses `loco` with per-run scaling and feature scalings without a matching response scaling |
| `scale_groups(values, groups, scaling)` | Demeans or z-scores each group of rows on its own statistics |
| `scale_on_training_rows(values, run_ids, train_mask, test_mask, scaling)` | Scales each run with the statistics of its training rows in the fold and applies them to its held-out rows (`--response-scaling train`) |
| `training_components(values, train_mask, n_components)` | Projection on the first principal components of the training rows (`--n-components`) |
| `fit_folds(data, args, subsets)` | Iterates over folds, applies fold-wise scaling and projection, and calls `evaluate_split` |

## `fold_evaluator.py`

Banded ridge, scoring and the partition for one split. Inputs arrive already scaled.

| Name | Purpose |
|---|---|
| `fit_group_ridge(...)` | himalaya `GroupRidgeCV` (random search, inner leave-one-run-out); the training mean of the responses is the intercept |
| `evaluate_split(...)` | Fits every requested subset of {A, V, J} on one split; refuses overlapping training and held-out rows or fewer than two training runs |
| `partition_variance(r2)` | The seven inclusion–exclusion regions from the seven subset R² maps; they sum to R²(AVJ) |
| `_r2_per_target`, `_pearson_per_target` | Per-grayordinate R² (negatives kept, NaN for constant targets) and Pearson r |

## `encoding_utils.py`

| Name | Purpose |
|---|---|
| `build_fmri_arrays(...)` | Bins the continuous four-run CIFTI into windows with the response delay and, unless `zscore=False`, z-scores each grayordinate within each run, training and held-out clips separately |
| `spm_hrf`, `apply_hrf_to_segment` | SPM canonical hemodynamic response function and its convolution with embedding columns (`--hrf`) |
| `save_cifti_maps`, `load_cifti_maps` | Named-map `.dscalar.nii` input and output on the template's grayordinate axis |

## Tests

`tests/test_fold_evaluator.py`, `tests/test_no_leakage.py`, `tests/test_encoding.py`, `tests/test_variance_partition.py`.
