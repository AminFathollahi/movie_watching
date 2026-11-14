# Tests

Synthetic-data unit tests; no HCP data, embeddings or GPU needed. Run from the repository root:

```bash
python -m pytest tests
python -m pytest deprecated/rsa/tests deprecated/cka/tests
```

The tests of the RSA scripts that moved to `deprecated/rsa/` are in `deprecated/rsa/tests/`: `test_av_derived_maps.py`, `test_av_scramble_permutation_inference.py`, `test_channel_subset_searchlight.py`, `test_max_uni.py`, `test_paired_bootstrap_resampling.py`, `test_perm_searchlight.py`, `test_residualized_maps.py`, `test_sheet_rsa.py`, `test_rsa_utils.py` (RDM construction and comparison), `test_rdm_searchlight.py` (Euclidean, Mahalanobis, crossnobis and Kendall searchlights against direct computation), `test_subcortical_searchlight.py` (one searchlight pass over the subcortical neighbor columns equals the per-structure passes), `test_load_subject_maps.py`, `test_model_norm_names.py` (output names of the correlation-distance searchlight) and `test_rdm_searchlight_subjects.py` (per-subject Euclidean maps).

The tests of the CKA code and of the RSA, CKA and encoding measure comparison that moved to `deprecated/cka/` are in `deprecated/cka/tests/`: `test_cka.py`, `test_collect_spj.py` and `test_measure_comparison.py`. RSA and CKA are retired and not part of the current analyses.

| File | Guards |
|---|---|
| `test_preprocess.py` | Signal cleaning in `preprocessing/preprocess_individual.py` |
| `test_encoding.py` | Response binning for encoding |
| `test_fold_evaluator.py` | R², Pearson r, folds, variance partition |
| `test_variance_partition.py` | Embedding sources, output names, screening, incremental fits, collection of the partition quantities |
| `test_no_leakage.py` | Fits independent of held-out rows; training-row scaling and components; refused scalings |
| `test_binning.py` | Response binning (`shared/binning.py`) |
| `test_subcortical_neighbors.py` | Subcortical neighborhoods stay within their structure |
| `test_labels.py` | Per-second and per-window label columns (speech, face, body, face-or-body) |
| `test_time_resolved.py` | TFCE, permutation procedure and collection of the time-resolved maps |
| `test_runners.py` | Work lists of the encoding runners, the time-resolved skip rule, and the runners' skip, retry, out-of-memory hook, GPU lock, disk limit, single instance, stop and resume with fake jobs |
| `test_model_norm.py` | Per-run embedding normalization and its naming (`shared/naming.py`) |
| `test_searchlight_distance.py` | Correlation-distance searchlight kernels against direct computation |
| `test_av_pairing.py` | Crossed-pair interaction contrast and fold-confined pairing |
| `test_cf_math.py` | Connective-field ridge utilities |
| `test_lboe_naming.py` | Eigenfunction-count naming |
| `test_roi_mean_partial_connectivity.py` | ROI-mean partial correlation maps |
| `test_bivariate_cifti.py` | Bivariate color quantization |
| `test_partial_bivariate_cifti.py` | ROI-mean partial-correlation bivariate export |
| `test_cca_island_variants.py` | `cf_modeling/deprecated/run_cca_islands.py` |
| `test_channel_cca_analysis.py` | `cf_modeling/deprecated/channel_cca_analysis.py` |
| `test_migrate_cca_artifacts.py` | `cf_modeling/deprecated/migrate_cca_artifacts.py` |
| `test_vertex_clustering.py` | Vertex clustering sweep |
| `test_vertex_model_selection.py` | Vertex clustering hyperparameter selection |
| `test_prepare_wb_view_cortical.py` | Workbench bundle contents |
| `test_subcortical_visualization.py` | Subcortical result discovery and grouping |
