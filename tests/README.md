# Tests

Synthetic-data unit tests; no HCP data, embeddings or GPU needed. Run from the repository root:

```bash
python -m pytest tests
```

| File | Guards |
|---|---|
| `test_preprocess.py` | Signal cleaning in `preprocess_individual.py` |
| `test_encoding.py` | Response binning for encoding |
| `test_fold_evaluator.py` | R², Pearson r, folds, variance partition |
| `test_variance_partition.py` | Embedding sources, output names, screening, incremental fits, partition maps |
| `test_no_leakage.py` | Fits independent of held-out rows; training-row scaling and components; refused scalings |
| `test_cka.py` | Window index, searchlight CKA, semi-partial, commonality, pooling of subject maps |
| `test_measure_comparison.py` | Homogeneity, ring bins and map paths of the measure comparison |
| `test_rsa.py` | RDM construction and comparison, response binning |
| `test_model_norm.py` | Per-run embedding normalization and its naming |
| `test_searchlight_distance.py` | Searchlight RDM kernels against direct computation |
| `test_perm_searchlight.py` | Shared permutation of the paired null |
| `test_channel_subset_searchlight.py` | Channel-subset column selection |
| `test_sheet_rsa.py` | Cortical-sheet tower geometry and neighbors |
| `test_av_derived_maps.py` | Audiovisual derived-map comparisons |
| `test_max_uni.py` | `max_uni` artifact renaming |
| `test_residualized_maps.py` | Residualized-embedding models |
| `test_av_scramble_permutation_inference.py` | Scramble permutation inference: signed maps, FDR, max statistic |
| `test_paired_bootstrap_resampling.py` | Paired block bootstrap |
| `test_av_pairing.py` | Crossed-pair interaction contrast and fold-confined pairing |
| `test_channel_class_rsa.py` | Input validation of `rsa/channel_class_rsa.py` |
| `test_cf_math.py` | Connective-field ridge utilities |
| `test_lboe_naming.py` | Eigenfunction-count naming |
| `test_roi_mean_raw_connectivity.py` | ROI-mean correlation maps |
| `test_roi_mean_partial_connectivity.py` | ROI-mean partial correlation maps |
| `test_bivariate_cifti.py` | Bivariate color quantization |
| `test_partial_bivariate_cifti.py` | Partial-correlation bivariate exports |
| `test_raw_corr_bivariate_cifti.py` | Correlation bivariate exports |
| `test_cca_island_variants.py` | `cf_modeling/deprecated/run_cca_islands.py` |
| `test_channel_cca_analysis.py` | `cf_modeling/deprecated/channel_cca_analysis.py` |
| `test_migrate_cca_artifacts.py` | `cf_modeling/deprecated/migrate_cca_artifacts.py` |
| `test_vertex_clustering.py` | Vertex clustering sweep |
| `test_vertex_model_selection.py` | Vertex clustering hyperparameter selection |
| `test_prepare_wb_view_cortical.py` | Workbench bundle contents |
| `test_subcortical_visualization.py` | Subcortical result discovery and grouping |
