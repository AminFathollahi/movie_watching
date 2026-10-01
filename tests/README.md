# Tests

Unit tests for the shared math, naming and file-format code. They use synthetic arrays or temporary directories; none needs the HCP data, the embeddings or a GPU (banded ridge tests use the `numpy` backend). The one exception is an optional check in `test_sheet_rsa.py` that reads a cached coordinate file under `outputs/rsa/cca_seed_sheet_rsa/` and returns early when it is absent.

## Run

```bash
conda activate movie
pytest tests -v                          # all tests, from the repository root
pytest tests/test_fold_evaluator.py -v   # one file
pytest tests -k leakage                  # by name
```

Some tests load scripts by file path (`importlib.util.spec_from_file_location`) and others import packages (`rsa`, `encoding`, `cluster`, ...), so run pytest from the repository root.

## Files

### Preprocessing and shared utilities

| File | Guards |
|---|---|
| `test_preprocess.py` | `preprocess_individual.py`: global signal regression, percent signal change, Savitzky-Golay filter parameters, delay-shifted TR indices |
| `test_encoding.py` | `encoding/shared/encoding_utils.py::build_fmri_arrays`: array shapes, run onsets, bin totals (nibabel mocked) |

### Encoding

| File | Guards |
|---|---|
| `test_fold_evaluator.py` | Fold masks for both split variants, R-squared with negative values and constant targets, variance partition regions that sum to the full model, recovery of a purely joint signal, per-target Pearson r |
| `test_variance_partition.py` | Separate audio, video and joint embedding sources, tag placement in output names, per-clip R-squared, single-feature screening (`--subsets`), the ten partition maps including `j_minus_av` |
| `test_no_leakage.py` | Fitted quantities unchanged when test features or test responses are perturbed, training-only standardization and centering, detection of residuals fitted before the split, rejection of global scramble models |

### CKA

| File | Guards |
|---|---|
| `test_cka.py` | `cka/`: likelihood identities of the CKA statistics, whitener, cross-validated Gram matrix, log-likelihoods, window indexing against `preprocess_fmri`, both searchlight passes against a direct per-vertex computation |

### RSA

| File | Guards |
|---|---|
| `test_rsa.py` | `rsa/shared/rsa_utils.py`: RDM construction (correlation and cosine), RDM correlation (Spearman and Pearson, lower triangle), fMRI binning, z-scoring and delay |
| `test_model_norm.py` | Model normalization variants `zscore` and `center` (per-run), the command-line default `center`, output names carrying the tag after the method, fMRI always z-scored |
| `test_searchlight_distance.py` | `rsa/searchlight.py` `--distance`: the single-vertex and batched kernels against a direct `pdist` and Spearman computation for correlation and squared Euclidean distance |
| `test_perm_searchlight.py` | `rsa/perm_searchlight.py`: paired null shares one permutation and matches the single-seed null |
| `test_channel_subset_searchlight.py` | Column selection for cluster-channel searchlights and length-mismatch rejection |
| `test_sheet_rsa.py` | `rsa/shared/sheet_rsa.py`: cortical-sheet tower boundaries, nearest-neighbor search on a lattice, within-tower random neighbors, per-tower summaries, robust color limits |
| `test_av_derived_maps.py` | Audiovisual derived-map comparisons, baseline mapping to each model's own encoders, per-map resume without overwriting |
| `test_max_uni.py` | Migration of legacy `max_uni` artifact names (files, maps, summary keys) |
| `test_residualized_maps.py` | Residualized-embedding roster, CAV-MAE audio and video blocks, constant-geometry RDM, per-scalar resume |
| `test_av_scramble_permutation_inference.py` | Permutation inference from scramble reruns: signed significance maps, FDR maps, max-statistic critical value with the pseudocount |
| `test_paired_bootstrap_resampling.py` | Joint (paired) block resampling in `corrected_2factor_bootstrap` versus independent resampling |
| `test_av_pairing.py` | `notebooks/feature_extraction/av_pairing.py`: factorial contrast cancels additive components, fold-confined pairing is deterministic and rejects a single-clip pool |
| `test_channel_class_rsa.py` | Input validation and inference bundle of the retired `rsa/channel_class_rsa.py` |

### Reports

| File | Guards |
|---|---|
| `test_measure_comparison.py` | `reports/measure_comparison/compare_measures.py`: neighbourhood homogeneity against a direct mean pairwise correlation, ring-diagnostic bins, map file names |

### CF modeling and connectivity

| File | Guards |
|---|---|
| `test_cf_math.py` | `cf_modeling/shared/ridge_utils.py`: projection onto eigenfunctions, null R-squared, leave-one-run-out splits, pipeline structure, integration score |
| `test_lboe_naming.py` | Eigenfunction-count naming (`cf_naming.py`), per-subject output root, artifact migration |
| `test_roi_mean_raw_connectivity.py` | Zero-order Pearson correlation of every vertex with an ROI-mean time series |
| `test_roi_mean_partial_connectivity.py` | Partial correlation controlling the other ROI mean, run-wise standardization, batched mask means |
| `test_bivariate_cifti.py` | Two-axis color quantization for split-CF bivariate exports |
| `test_partial_bivariate_cifti.py` | Four-view partial-correlation exports with a shared scale, map-order check |
| `test_raw_corr_bivariate_cifti.py` | Four-view zero-order correlation exports, preprocessing label in new map names |
| `test_cca_island_variants.py` | Retired `cf_modeling/deprecated/run_cca_islands.py`: threshold variants, naming, merge saddle, support rules |
| `test_channel_cca_analysis.py` | Retired `cf_modeling/deprecated/channel_cca_analysis.py`: profile correlation, partial correlation, block bootstrap, figures |
| `test_migrate_cca_artifacts.py` | Retired `cf_modeling/deprecated/migrate_cca_artifacts.py`: additive plan, dry run, in-place rewrite |

### Clustering

| File | Guards |
|---|---|
| `test_vertex_clustering.py` | `cluster/vertex_clustering.py`: every reducer and clusterer runs, configuration tags, `.dlabel.nii` round trip, stimulus and noise masks, small end-to-end run |
| `test_vertex_model_selection.py` | `cluster/vertex_model_selection.py`: embedding quality, reducer and cluster grids, elbow dimension, selection score |

### Visualization and subcortical

| File | Guards |
|---|---|
| `test_prepare_wb_view_cortical.py` | `viz/prepare_wb_view_cortical.py` lists group-level results only, never per-subject or cache directories |
| `test_subcortical_visualization.py` | Anatomical grouping, bilateral output units, result discovery, result identity in names, vertex-count self-check |
