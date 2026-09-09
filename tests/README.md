# Tests — Unit and Integration Tests

Automated test suite validating core functionality of RSA, encoding, CF modeling, and feature extraction pipelines.

## Test Files

| Test File | What It Guards |
|-----------|----------------|
| `test_encoding.py` | Ridge regression fitting, cross-validation, R² computation, per-vertex alpha selection |
| `test_fold_evaluator.py` | Ridge CV fold splitting, standardization, banded ridge fitting with himalaya |
| `test_compression.py` | Dimensionality reduction (PCA, random projection, clustering) fit/test pipeline |
| `test_incremental_av_checkpoints.py` | Incremental AV checkpoint saving/resuming and compression-efficiency workflow |
| `test_pairing_control.py` | Cross-clip mismatch control extraction and fold-confined pairing |
| `test_av_pairing.py` | Paired audio/video frame and bin extraction validation |
| `test_encoding_av_derived_maps.py` | AV conjunction, superadditivity, max-unimodal map computation |
| `test_roi_av_profile.py` | ROI-level variance decomposition and AV profile aggregation |
| `test_rsa.py` | RDM construction, Spearman/Pearson correlation, per-vertex searchlight RSA |
| `test_av_derived_maps.py` | RSA AV conjunction and superadditivity contrasts |
| `test_max_uni.py` | AV-minus-max-unimodal contrast and per-subject statistics |
| `test_residualized_maps.py` | Linear and projection residualization of embeddings |
| `test_partial_bivariate_cifti.py` | Bivariate CIFTI export for partial-correlation maps |
| `test_raw_corr_bivariate_cifti.py` | Bivariate CIFTI export for raw-correlation maps |
| `test_bivariate_cifti.py` | Core bivariate CIFTI construction and dual-colorbar generation |
| `test_lboe_naming.py` | LBOE artifact naming convention and count qualification |
| `test_roi_hierarchy.py` | Hierarchical clustering and parcellation of ROI connectivity patterns |
| `test_roi_mean_raw_connectivity.py` | Pearson correlation between vertices and ROI mean timecourses |
| `test_roi_mean_partial_connectivity.py` | Partial correlation controlling for bilateral ROI means |
| `test_cf_math.py` | Core CF model linear algebra and numerical correctness |
| `test_channel_cca_analysis.py` | Channel-level CF analysis for CCA regions |
| `test_channel_class_rsa.py` | Restricted RSA for channel significance classes (audio-only, video-only, both, neither) |
| `test_channel_modality_preference.py` | Channel modality preference scoring and labeling |
| `test_channel_stability_selection.py` | Channel cluster stability and model selection |
| `test_cca_island_variants.py` | CCA island extraction and threshold variants |
| `test_heldout_roi_alignment.py` | ROI alignment from held-out seeds and regional pairing |
| `test_migrate_cca_artifacts.py` | CCA artifact migration and naming consistency |
| `test_av_scramble_permutation_inference.py` | Empirical permutation null aggregation from scramble reruns |
| `test_prepare_wb_view_cortical.py` | Workbench cortical visualization setup and spec generation |
| `test_preprocess.py` | fMRI preprocessing (SG filter, PSC, GSR) correctness |
| `test_subcortical_visualization.py` | Subcortical mesh generation and overlay export for Workbench |
| `test_vertex_clustering.py` | Vertex-level clustering for ROI definition |
| `test_vertex_model_selection.py` | Per-vertex model selection (e.g., best encoding model per vertex) |
| `test_paired_bootstrap_resampling.py` | Bootstrap resampling for paired comparisons and confidence intervals |
