# encoding/shared

Library code behind `encoding/variance_partition.py`. Import as `encoding.shared.*` with the repository root on `sys.path` (`variance_partition.py` does this itself). There are no command-line entry points here; the argument parser (`build_parser`) is used by `variance_partition.py`.

## Modules

### `fold_evaluator.py`

Banded-ridge fitting, feature standardization, scoring and the variance partition for one train/test split.

| Name | Purpose |
|------|---------|
| `ALL_SUBSETS`, `BANDS`, `FEATURE_SCALINGS` | `("a","v","j","av","aj","vj","avj")`, `("a","v","j")`, `("zscore","center")` |
| `standardize_bands(train_bands, test_bands, scale=True)` | Per band, subtract the training mean and, when `scale` is true, divide by the training standard deviation (features with deviation below 1e-12 are left unscaled). Applies the training statistics to the test band. Returns standardized train and test bands and the statistics |
| `run_splitter(groups)` | `PredefinedSplit` that leaves one run out; needs at least two runs |
| `fit_group_ridge(train_bands, y_train, test_bands, train_runs, alphas, n_iter, backend, random_state)` | himalaya `GroupRidgeCV(groups="input", solver="random_search", fit_intercept=False)` with leave-one-run-out inner cross-validation. Subtracts the training mean of the responses before the fit and adds it back to the prediction. `backend="torch_cuda"` falls back to `"torch"` when CUDA is unavailable |
| `evaluate_split(audio, video, joint, targets, run_ids, train_mask, test_mask, alphas, label, subsets, n_iter, backend, model_random_state, feature_scaling)` | Standardizes the bands with training rows only, fits every requested subset, and returns a `FoldResult` (`test_indices`, `y_test`, `predictions` and `r2` keyed by subset, `provenance`, `arrays` with `{subset}_best_alphas` and `{subset}_deltas`). Raises if train and test rows overlap, the test set is empty, or fewer than two training runs exist |
| `partition_variance(r2)` | From the seven held-out R² maps (keys of `ALL_SUBSETS`) returns `unique_a`, `unique_v`, `unique_j`, `shared_av_only`, `shared_aj_only`, `shared_vj_only`, `shared_avj` by inclusion-exclusion; the regions sum to R²(avj) and negative values are kept |
| `_r2_per_target(y_true, y_pred)` | Per-column R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)², with ȳ the mean of the rows passed in; NaN where Σ(y − ȳ)² ≤ 1e-12 |
| `_pearson_per_target(y_true, y_pred)` | Per-column Pearson correlation; NaN where the denominator is 1e-12 or smaller |

### `splits.py`

Inputs and split logic shared by the seven-model fits.

| Name | Purpose |
|------|---------|
| `build_parser()` | Argument parser with all fit options (inputs, `--split`, `--feature-scaling`, `--model`, `--audio-model`, `--video-model`, `--subsets`, `--tag`, `--output-name`, `--joint-model-template`, timing, `--hrf`, `--exclude-video-ids`, penalty grid, `--n-iter`, `--backend`, `--model-random-state`); defaults are listed in `encoding/README.md` |
| `REPEATED_VALIDATION_CLIPS` | `("video5","video9","video14","video18")` |
| `sample_metadata(timing, bin_sec, skip_sec)` | One row per window: `row_index`, `video_id`, `run_id`, `window_index`, `window_onset_sec`; a clip of duration D has floor((D − bin) / skip) + 1 windows when D is at least the bin length |
| `apply_hrf_by_clip(embeddings, metadata, bin_sec)` | Convolves embeddings with the SPM hemodynamic response function within each clip (used with `--hrf`) |
| `reject_global_scramble(*names)` | Raises for model names containing `avscramble` |
| `load_inputs(args)` | Loads the A, V and J embeddings of the requested models (`{embeddings-dir}/{model}/bin{B}s_skip{S}s/{model}_{a,v,av}.npy`), checks their row counts against the timing table, builds the response matrix with `build_fmri_arrays`, checks that embedding and response run boundaries agree, and returns a dict: `a`, `v`, `joint` (per held-out run, or `default`), `targets`, `metadata` (sorted by run), `keep` (training-row mask), `excluded`, `joint_names`, `audio_model`, `video_model` |
| `make_folds(data, split)` | `fixed`: one fold, train = kept rows, test = excluded rows. `runwise`: one fold per run, train = kept rows of the other runs, test = kept rows of that run |
| `fit_folds(data, args, subsets)` | Generator over folds yielding `(label, test_mask, FoldResult)` from `evaluate_split`; selects the per-run J when `--joint-model-template` is used |

### `encoding_utils.py`

| Name | Purpose |
|------|---------|
| `build_fmri_arrays(cifti_path, run_trs_path, timing_df, test_video_ids, bin_sec, tr, delay_sec=0.0, skip_sec=None)` | Loads the continuous four-run CIFTI, extracts each clip's windows (window start = round((onset − run start + delay) / tr) plus the window index times round(skip / tr)), averages repetition times within each window, z-scores each grayordinate within each run (training windows with their own mean and standard deviation; held-out clips of that run with their own), and splits by `video_id`. Returns `Y_train`, `Y_test` and the training row index at which each run begins |
| `spm_hrf(tr, oversampling=16)` | SPM canonical hemodynamic response function sampled at `tr`, normalized to sum to 1 |
| `apply_hrf_to_segment(segment, hrf_kernel)` | Convolves each feature column with the kernel, trimmed to the original length |
| `save_cifti_maps(maps, template_path, output_path)` | Writes a dictionary of named maps as a `.dscalar.nii`, one named map per entry, using the grayordinate axis of the template |

## Tests

```bash
python -m pytest tests/test_fold_evaluator.py tests/test_no_leakage.py tests/test_encoding.py tests/test_variance_partition.py
```

`test_fold_evaluator.py` covers `evaluate_split`-level scoring, `partition_variance` and `make_folds`; `test_no_leakage.py` covers training-only statistics and fits; `test_encoding.py` covers `build_fmri_arrays`; `test_variance_partition.py` covers input loading, output names and the partition maps.
