# Graph Report - .  (2026-07-15)

## Corpus Check
- 9 files · ~104,916 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1181 nodes · 2185 edges · 80 communities (67 shown, 13 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 66 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- K-Reliability RSA
- Encoding Pipeline Core
- CF Functional ROI Masking
- fMRI Preprocessing (GSR/PSC/SG)
- Localizer Map Labeling
- TopoOmni Extraction Core
- CF Geometry Extraction (Subsurface/LBOE)
- Partial RSA (banded projection)
- CF Ridge-Encoding Utils
- RDM-Diagonal Grouped Searchlight
- vicsompy Vendor Attribution
- Seed Connectivity Analysis
- RSA analysis.sh Orchestrator
- CF Model Fitting
- TFCE Group Stats (encoding)
- Multimodal CKA Decomposition
- Encoding Unit Tests
- Glasser Parcel RSA Utils
- Noise Ceiling Computation
- CF run_analysis.sh Orchestrator
- RSA Group Stats (TFCE/FDR)
- Delta-Rho Comparison
- RSA Overlap Scoring
- Encoding analysis.sh Orchestrator
- CAV-MAE-Sync Extraction
- RSA Timing/HRF Utils
- Glasser Parcellation Loader
- CIFTI I/O Core
- Integration Convergence Maps
- Crossnobis Searchlight
- fMRI Preprocess/Bin Utils
- RSA Border Drawing
- Repo Directory Layout
- Omni3B Extraction Docs/Plan
- Localizer Readout Comparison
- Scramble Paired Group Stats
- Encoding Group Stats (legacy)
- AV-Integration Localizer Plan Notes
- RSA Script Cross-References
- Cortex Map Plotting
- Encoding Env/Pipeline Docs
- Nemotron Extraction Docs
- Conda Env Version Pins
- CF ROI Mask Generation
- FDR Unit-Selection Design Notes
- AV Transformer Scramble Docs
- PE-AV Dummy-Modality Extraction
- PE-AV Scramble Extraction
- Scramble Diff-Map Rationale
- AudioCaption Env/Model
- CAV-MAE-Sync Env/Model
- CF Integration Maps
- Text Rewrite (LLM cleanup)
- Results Report / Timing Docs
- vicsompy Environment Setup
- AVTransformer Env Isolation Notes
- Whisper Transcription (2s bins)
- Surface Averaging Script
- Scramble Unimodal Copy Builder
- Nemotron Scramble Extraction
- Omni3B Scramble Extraction
- Omni3B Unimodal Extraction
- TopoOmni Sheet Coordinates
- Group-Avg Multiscale Script
- Hedger Data Concatenation
- CF Modeling Package Init
- Hedger Data Validation
- AV-Separability Brain-Map Runner
- Crossnobis Runner Script
- Sheet-Localizer Brain-Map Runner
- Sparse Matrix Import (misc)
- RDM-Diagonal Module Ref

## God Nodes (most connected - your core abstractions)
1. `get_bm_axis()` - 44 edges
2. `get_cortex_vertex_indices()` - 33 edges
3. `preprocess_fmri()` - 29 edges
4. `save_cifti_multimap()` - 24 edges
5. `process_model_embeddings()` - 23 edges
6. `save_cifti_map()` - 22 edges
7. `get_neighbors()` - 22 edges
8. `preprocess_subject()` - 21 edges
9. `merge_into_combined()` - 20 edges
10. `get_combined_map_names()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `movie conda env spec (encoding/environment.yml)` --semantically_similar_to--> `movie conda env spec (cf_modeling/environment.yml)`  [INFERRED] [semantically similar]
  encoding/environment.yml → cf_modeling/environment.yml
- `movie conda env spec (rsa/environment.yml)` --semantically_similar_to--> `movie conda env spec (cf_modeling/environment.yml)`  [INFERRED] [semantically similar]
  rsa/environment.yml → cf_modeling/environment.yml
- `topo_omni_extract_dummy_modality.py` --semantically_similar_to--> `pe_av_extract_dummy_modality.py`  [INFERRED] [semantically similar]
  plan.md → README.md
- `Z-scoring per run with training stats applied to test to prevent leakage (rationale)` --semantically_similar_to--> `Vectorized numpy Pearson r replacing per-vertex scipy.stats.pearsonr loop (rationale)`  [INFERRED] [semantically similar]
  rsa/README.md → encoding/README.md
- `test_fit_null_r2_random_shape_and_range()` --calls--> `fit_null_r2()`  [INFERRED]
  tests/test_cf_math.py → cf_modeling/shared/ridge_utils.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Isolated per-model conda environments to avoid torch/transformers version conflicts** — avtransformer_conda_env, topo_omni_conda_env, cav_mae_sync_conda_env, audiocaption_conda_env, movie_conda_env [INFERRED 0.85]
- **Dummy-modality extraction pattern (blank/silent placeholder to isolate true joint AV embeddings) reused across PE-AV, TopoOmni, planned Nemotron/Omni3B** — readme_pe_av_extract_dummy_modality_py, plan_topo_omni_extract_dummy_modality_py, plan_nemotron_extract_dummy_modality_py, plan_omni3b_extract_dummy_modality_py [INFERRED 0.85]
- **cf_modeling pipeline built on vicsompy (MssCf, Subsurface, generate_leave_one_run_out) with attribution to Hedger et al. 2025 and Gallant Lab** — cf_readme_mssCf, cf_readme_subsurface, cf_readme_generate_loro, cf_readme_hedger_paper, cf_modeling_vendor_hedger_cf_attribution_gallant_lab [EXTRACTED 1.00]

## Communities (80 total, 13 thin omitted)

### Community 0 - "K-Reliability RSA"
Cohesion: 0.06
Nodes (68): Cifti2Image, Namespace, compute_rsa_maps_for_ks(), generate_half_surface(), load_half_average(), main(), parse_args(), ndarray (+60 more)

### Community 1 - "Encoding Pipeline Core"
Cohesion: 0.07
Nodes (54): _cifti_path(), _compute_n_test(), _config_label(), _fdr_sigmap(), main(), parse_args(), DataFrame, ndarray (+46 more)

### Community 2 - "CF Functional ROI Masking"
Cohesion: 0.06
Nodes (41): _compute_mask(), _grayord_mask_to_sphere_hemispheres(), _load_r2_maps(), main(), parse_args(), ndarray, cf_modeling/03_functional_masks.py ===================================== Derive, Apply threshold within valid vertices; floor at R2_nc > 0.      Returns     ---- (+33 more)

### Community 3 - "fMRI Preprocessing (GSR/PSC/SG)"
Cohesion: 0.06
Nodes (52): apply_gsr(), apply_psc(), apply_sg_filter(), compute_group_average_from_disk(), extract_cortex(), get_run_path(), load_subjects(), main() (+44 more)

### Community 4 - "Localizer Map Labeling"
Cohesion: 0.10
Nodes (43): main(), rsa/label_av_separability_maps.py ===================================== Merges e, main(), rsa/label_sheet_localizer_maps.py ===================================== Merges e, base_name(), demo(), driver_tag(), model_tag() (+35 more)

### Community 5 - "TopoOmni Extraction Core"
Cohesion: 0.07
Nodes (37): ModuleList, _build_inputs_for_video(), find_all_segments(), _find_audio_path(), _load_audio(), main(), ndarray, Path (+29 more)

### Community 6 - "CF Geometry Extraction (Subsurface/LBOE)"
Cohesion: 0.06
Nodes (27): build_subsurface(), _load_csv_masks(), load_dlabel_masks(), main(), parse_args(), ndarray, cf_modeling/01_extract_geometry.py ===================================== Build v, Extends vicsompy's Subsurface with a sigma=1e-6 shift in the Laplacian.      Wit (+19 more)

### Community 7 - "Partial RSA (banded projection)"
Cohesion: 0.09
Nodes (35): NamedTuple, fit_banded_projection(), import_or_default(), ndarray, rsa/partial_rsa.py ======================= Vertex-wise Partial RSA via Banded Ri, Z-score a 1-D vector; return zeros if std == 0., Compute correlation-distance RDM and return z-scored lower triangle.      Parame, Find per-band alphas via himalaya RidgeCV, then build projection matrix C. (+27 more)

### Community 8 - "CF Ridge-Encoding Utils"
Cohesion: 0.07
Nodes (29): build_pipeline(), build_sphere_to_grayord_lut(), fit_null_r2(), project_onto_lboes(), ndarray, shared/ridge_utils.py ===================== Shared ridge regression utilities fo, Map full-sphere bilateral vertex indices → CIFTI grayordinate positions.      Th, Project BOLD surface data onto LBOEs for a list of Subsurface objects.      Repl (+21 more)

### Community 9 - "RDM-Diagonal Grouped Searchlight"
Cohesion: 0.11
Nodes (33): _build_group_structure(), _build_pair_indices(), build_segment_labels(), compute_grouped_parcel_rsa(), _get_neighbors(), _grouped_searchlight_vertex(), _load_glasser_parcels(), main() (+25 more)

### Community 10 - "vicsompy Vendor Attribution"
Cohesion: 0.07
Nodes (33): vendor/ DEPRECATED notice, Departures from vicsompy: CLI ROI args, dynamic LBOE cap, preprocessed CIFTI input, group/persubject modes (rationale), Gallant Lab (UC Berkeley), Attribution: Hedger et al. (2025), Mohammad Amin Fathollahi (pipeline author), Nicholas Hedger (vicsompy author), voxelwise_tutorials package, 00_make_roi_masks.py (+25 more)

### Community 11 - "Seed Connectivity Analysis"
Cohesion: 0.12
Nodes (29): _apply_window(), _border_to_surface_mask(), build_filtered_stim_mask(), build_official_rest_mask(), compute_seed_connectivity(), load_roi_grayord_mask(), main(), parse_args() (+21 more)

### Community 12 - "RSA analysis.sh Orchestrator"
Cohesion: 0.15
Nodes (22): BLIS_NUM_THREADS, _emb_exists(), log(), MKL_NUM_THREADS, NUMEXPR_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_avg() (+14 more)

### Community 13 - "CF Model Fitting"
Cohesion: 0.11
Nodes (20): load_subsurface(), main(), parse_args(), ndarray, cf_modeling/02_fit_cf_model.py ================================ Fit a multi-sour, Split (n_grayord, T_total) into per-run train/test with z-scoring.      Replicat, Load a Subsurface pkl from the cache built by 01_extract_geometry.py., Run the full CF modeling pipeline on pre-loaded grayordinate data.      Paramete (+12 more)

### Community 14 - "TFCE Group Stats (encoding)"
Cohesion: 0.17
Nodes (24): get_combined_map_names(), Save a 1-D cortical map as a CIFTI dscalar.nii.      Args:         data_1d: (n_g, Return the scalar map names in an existing combined CIFTI dscalar.      Returns, save_cifti_map(), _compute_fdr_maps(), _compute_tfce_sigmap(), _filter_clusters(), _load_surface() (+16 more)

### Community 15 - "Multimodal CKA Decomposition"
Cohesion: 0.15
Nodes (24): _center_kernel(), _cka_vertex_cpu(), compute_interaction_residual(), gram(), linear_cka(), load_glasser_parcels(), main(), parse_args() (+16 more)

### Community 16 - "Encoding Unit Tests"
Cohesion: 0.13
Nodes (22): _make_run_trs(), _make_timing_df(), _patch_fmri(), DataFrame, ndarray, tests/test_encoding.py ====================== Unit tests for encoding/shared/enc, Y_train + Y_test row count must equal total included bins., If all videos are test, Y_train should have 0 rows, Y_test covers all bins. (+14 more)

### Community 17 - "Glasser Parcel RSA Utils"
Cohesion: 0.15
Nodes (21): compute_parcel_rsa(), Compute RSA for every Glasser parcel., compute_rdm(), correlate_rdms(), Compute a representational dissimilarity matrix (RDM).      Args:         data:, Correlate the lower triangles of two RDMs.      Args:         rdm1: (n, n) float, tests/test_rsa.py ================= Unit tests for rsa/shared/rsa_utils.py.  All, Changing the upper triangle must not change the correlation. (+13 more)

### Community 18 - "Noise Ceiling Computation"
Cohesion: 0.16
Nodes (21): _compute_nc_cpu(), _compute_nc_gpu(), _dispatch_hemisphere(), _load_and_bin_disk(), _load_and_bin_streaming(), main(), parse_args(), _pearson_rows() (+13 more)

### Community 19 - "CF run_analysis.sh Orchestrator"
Cohesion: 0.21
Nodes (19): BLIS_NUM_THREADS, log(), MKL_NUM_THREADS, NUMEXPR_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, PYCORTEX_FILESTORE, PYTORCH_CUDA_ALLOC_CONF (+11 more)

### Community 20 - "RSA Group Stats (TFCE/FDR)"
Cohesion: 0.20
Nodes (20): _compute_fdr_maps(), _compute_tfce_sigmap(), _config_label(), _filter_clusters(), _load_surface(), main(), parse_args(), _patch_fdr_maps() (+12 more)

### Community 21 - "Delta-Rho Comparison"
Cohesion: 0.16
Nodes (17): _align_stacks(), _fdr_sigmap(), load_block_stack(), load_rho_stack(), main(), parse_args(), _parse_baseline_pairs(), ndarray (+9 more)

### Community 22 - "RSA Overlap Scoring"
Cohesion: 0.20
Nodes (18): _bm_axis_from_template(), _bootstrap_spearman(), collect_maps(), _config_dir(), _fdr_mask(), _get_grayordinate_indices(), _load_rsa_fullbrain(), main() (+10 more)

### Community 23 - "Encoding analysis.sh Orchestrator"
Cohesion: 0.22
Nodes (14): BLIS_NUM_THREADS, _emb_exists(), log(), MKL_NUM_THREADS, NUMEXPR_NUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, run_avg() (+6 more)

### Community 24 - "CAV-MAE-Sync Extraction"
Cohesion: 0.18
Nodes (14): extract(), gather_pairs(), load_audio(), load_model(), load_video(), Path, Tensor, extract_cav_mae_sync.py ======================= Standalone script to extract CAV (+6 more)

### Community 25 - "RSA Timing/HRF Utils"
Cohesion: 0.18
Nodes (16): _apply_hrf_to_segment(), assert_segment_timing(), correlate_rdms_rho_a(), get_run_bin_counts(), process_model_embeddings(), process_model_embeddings_with_hrf(), DataFrame, ndarray (+8 more)

### Community 26 - "Glasser Parcellation Loader"
Cohesion: 0.27
Nodes (14): _cifti_path(), _config_label(), main(), parse_args(), DataFrame, ndarray, Path, rsa/glasser.py ========================== Parcel-wise RSA using the Glasser MMP (+6 more)

### Community 27 - "CIFTI I/O Core"
Cohesion: 0.20
Nodes (12): load_cifti_data(), merge_into_combined(), ndarray, cifti_io.py ====================== Minimal CIFTI I/O utilities for scripts., Add or overwrite one scalar map in a combined CIFTI dscalar file.      If *combi, Load a CIFTI dtseries and return data as (n_vertices, T) float32.      Args:, Write a nibabel CIFTI image atomically (temp file → rename).      Prevents parti, _save_cifti_atomic() (+4 more)

### Community 28 - "Integration Convergence Maps"
Cohesion: 0.25
Nodes (12): get_bm_axis(), Return the BrainModelAxis from a CIFTI file.      Args:         cifti_path: str, Save multiple cortical maps as a multi-map CIFTI dscalar.nii.      Args:, save_cifti_multimap(), main(), parse_args(), rsa/integration_convergence.py ================================ Move 4 — Cross-a, _load_map() (+4 more)

### Community 29 - "Crossnobis Searchlight"
Cohesion: 0.23
Nodes (13): build_model_rdm(), _crossnobis_isub_vertex(), load_subjects(), main(), parse_args(), DataFrame, ndarray, rsa/crossnobis_searchlight.py ============================== Inter-subject cross (+5 more)

### Community 30 - "fMRI Preprocess/Bin Utils"
Cohesion: 0.18
Nodes (14): preprocess_fmri(), Bin and normalize fMRI data to match model embeddings.      Parameters     -----, _make_timing(), Helper: build a single-run timing DataFrame.      onset_sec values are GLOBAL (c, Output must be z-scored per run — mean≈0, std≈1 over bins., Multi-clip timing: output row count equals sum of floor(dur/bin) per clip., Two-run timing: each run is z-scored independently., Positive delay shifts the fMRI window forward, reducing available bins. (+6 more)

### Community 31 - "RSA Border Drawing"
Cohesion: 0.26
Nodes (12): GiftiImage, _build_adjacency(), _connected_components(), _filter_small_islands(), main(), parse_args(), ndarray, rsa/draw_rsa_borders.py ======================= Draw surface borders around high (+4 more)

### Community 32 - "Repo Directory Layout"
Cohesion: 0.21
Nodes (12): cf_modeling/ (Connective Field Modeling), connectivity/ (Seed-based Functional Connectivity), seed_connectivity.py, encoding/ (Ridge Encoding Models), movie conda environment, notebooks/feature_extraction/, notebooks/visualization/, make_average.sh (+4 more)

### Community 33 - "Omni3B Extraction Docs/Plan"
Cohesion: 0.18
Nodes (12): qwen-omni-utils, omni3b's existing _av.npy is (a+v)/2 from separate unimodal passes, not directly comparable (rationale), omni3b_extract_dummy_modality.py (planned, not yet built), omni3b_extract_scramble.py, omni3b_extract_unimodal.py, topo_omni_extract.py, topo_omni_extract_scramble.py, topo_omni_extract_unimodal.py (+4 more)

### Community 34 - "Localizer Readout Comparison"
Cohesion: 0.27
Nodes (11): load_glasser_parcels(), Extract Glasser parcel membership mapped to fMRI grayordinate indices., combined_path(), dice(), load_maps(), main(), parcel_name_for_index(), ndarray (+3 more)

### Community 35 - "Scramble Paired Group Stats"
Cohesion: 0.33
Nodes (11): _compute_fdr_maps(), _load_block_stack(), _load_subject_stack(), main(), _one_sample_test(), parse_args(), ndarray, Path (+3 more)

### Community 36 - "Encoding Group Stats (legacy)"
Cohesion: 0.38
Nodes (10): csr_matrix, _compute_fdr_maps(), _filter_clusters(), _load_surface(), main(), parse_args(), ndarray, rsa/group_stats.py ====================== Aggregate per-subject searchlight RSA (+2 more)

### Community 37 - "AV-Integration Localizer Plan Notes"
Cohesion: 0.20
Nodes (11): AV-integration / TopoOmni localizer next steps (plan), Avoid circularity: word density can't drive clustering when whisper_speech_proxy is the independent proxy (rationale), Island Moran's I (spatial compactness test), rsa/label_av_separability_maps.py, pe-av-small-16-frame_transcript_t, presentation_brief.md, rsa/run_av_separability_brain_maps.sh, spatial_stats.py (+3 more)

### Community 38 - "RSA Script Cross-References"
Cohesion: 0.22
Nodes (11): rsa/analysis.sh, rsa/glasser.py, GNU parallel, data/movie_timing.csv, rsa/multimodal_decomposition.py, rsa/partial_rsa.py, rsa/precompute_neighbors.py, rsa/run_spin_permutations.py (+3 more)

### Community 39 - "Cortex Map Plotting"
Cohesion: 0.29
Nodes (9): get_cortex_vertex_indices(), Return left and right cortical vertex index arrays from a BrainModelAxis.      A, Compute and save vertex-wise p-values + FDR maps for a group-average rho map., _save_significance_maps(), load_map(), main(), ndarray, viz/plot_cortex_map.py ======================= Render a grayordinate-space 1-D m (+1 more)

### Community 40 - "Encoding Env/Pipeline Docs"
Cohesion: 0.22
Nodes (10): encoding/analysis.sh, encoding/encoding.py, Encoding — Ridge Encoding Models pipeline, himalaya RidgeCV (torch_cuda backend), encoding/shared/encoding_utils.py, Vectorized numpy Pearson r replacing per-vertex scipy.stats.pearsonr loop (rationale), Y_in_cpu=True to avoid GPU OOM (rationale), himalaya (ridge regression library) (+2 more)

### Community 41 - "Nemotron Extraction Docs"
Cohesion: 0.28
Nodes (9): nemotron_extract_dummy_modality.py (planned, analogous to topo_omni version), nvidia/omni-embed-nemotron-3b, PE-AV trained on <=10s clips, cannot reliably embed longer segments (rationale), Pooling-formula bug fix: av_mask union → (mean(a)+mean(v))/2 equal weighting (rationale), src/eval/run/run_selectivity.py (topo-omni repo), topo_omni_extract_dummy_modality.py, topo_omni_extract.py, pe_av_extract_dummy_modality.py (+1 more)

### Community 42 - "Conda Env Version Pins"
Cohesion: 0.25
Nodes (8): himalaya 0.4.11, movie conda env spec (cf_modeling/environment.yml), numpy < 2.0 pin required by himalaya (rationale), pycortex 1.3.0, torch 2.11.0+cu128, movie conda env spec (encoding/environment.yml), movie conda env spec (rsa/environment.yml), RSA — Representational Similarity Analysis pipeline

### Community 43 - "CF ROI Mask Generation"
Cohesion: 0.36
Nodes (7): load_dlabel_masks(), main(), parse_args(), cf_modeling/00_make_roi_masks.py ================================== Generate per, Write {roi}_{L/R}_mask.csv files in vicsompy's expected format.      Format: sin, Extract Boolean vertex masks for each ROI from the Glasser dlabel.      Paramete, save_masks()

### Community 44 - "FDR Unit-Selection Design Notes"
Cohesion: 0.29
Nodes (8): topo-discover/calculate_selectivity.py (TopoOmni repo), cluster_reports_for_proxy/_score, src/eval/extract/extract_clusters.py (TopoOmni repo), scipy.stats.false_discovery_control (Benjamini-Hochberg), FDR-gated channel-selection mode as alternative to top-1% (matches TopoOmni's own convention) (rationale), FDR mode mask size non-uniform across clusters; decide honest-empty vs minimum-unit floor (rationale), top_units_for_cluster() (rsa/topoomni_sheet_localizer.py), rsa/topoomni_sheet_localizer.py

### Community 45 - "AV Transformer Scramble Docs"
Cohesion: 0.29
Nodes (7): avtransformer conda environment, build_scramble_unimodal_copies.py, nemotron_extract_scramble.py, nemotron_extract_unimodal.py, Omni-Embed-Nemotron-3B model, pe_av_embeddings.ipynb, text.ipynb (Whisper + Gemma + InternVL2.5)

### Community 46 - "PE-AV Dummy-Modality Extraction"
Cohesion: 0.38
Nodes (6): extract_embeddings(), find_chunk_pairs(), main(), Path, notebooks/feature_extraction/pe_av_extract_dummy_modality.py ===================, Identical to pe_av_extract_scramble.py's function -- must produce the     SAME 6

### Community 47 - "PE-AV Scramble Extraction"
Cohesion: 0.48
Nodes (6): extract_embeddings(), find_chunk_pairs(), main(), Path, notebooks/feature_extraction/pe_av_extract_scramble.py =========================, scramble_audio()

### Community 48 - "Scramble Diff-Map Rationale"
Cohesion: 0.29
Nodes (7): Group-average-only scramble comparison lacks subject-level variability/significance test (rationale), rsa/group_stats.py, pe_av_extract_scramble.py, rsa/scramble_diff_maps.py, rsa/searchlight.py, pe_av_extract_scramble.py, rsa/group_stats.py

### Community 49 - "AudioCaption Env/Model"
Cohesion: 0.47
Nodes (6): anthropic Python SDK, audiocaption conda environment, audiocaption conda env spec, msclap (CLAP-Cap), audiocaption.ipynb, Claude Haiku (LLM rewrite)

### Community 50 - "CAV-MAE-Sync Env/Model"
Cohesion: 0.40
Nodes (6): cav-mae-sync conda environment, cav-mae-sync conda env spec, YuanGongND/cav-mae GitHub repo, torch 1.13.1 (pinned, isolated), CAV-MAE-sync model, extract_cav_mae_sync.py

### Community 51 - "CF Integration Maps"
Cohesion: 0.40
Nodes (5): collect_maps(), main(), parse_args(), cf_modeling/integration_maps.py ================================= Derive integra, Load map_name.npy from all completed subject directories → (N, n_verts).

### Community 52 - "Text Rewrite (LLM cleanup)"
Cohesion: 0.60
Nodes (5): _format_no_llm(), _is_hallucinated_transcript(), main(), Fuse concept captions + Whisper transcripts into clean per-segment event descrip, rewrite_one()

### Community 53 - "Results Report / Timing Docs"
Cohesion: 0.40
Nodes (6): results_report.tex, data/HCP_7T_Movie_Clip_Timing.csv, data/movie_timing.csv, official_timing.py, results_report.tex, Timing reconciliation still open (rationale)

### Community 54 - "vicsompy Environment Setup"
Cohesion: 0.47
Nodes (6): Environment Setup Guide, vicsompy.modeling.MssCf, vicsompy cannot be pip-installed with modern PyTorch (rationale), PYCORTEX_FILESTORE env var, vicsompy repository (Vicarious_somatotopy), VICSOMPY_REPO env var

### Community 55 - "AVTransformer Env Isolation Notes"
Cohesion: 0.40
Nodes (5): cav-mae-sync must stay isolated due to old torch/timm pin conflicting with all other envs (rationale), decord, avtransformer conda env spec, qwen-vl-utils, topo_omni kept isolated from avtransformer due to newer transformers/torch pins (rationale)

### Community 56 - "Whisper Transcription (2s bins)"
Cohesion: 0.70
Nodes (3): average_surfaces(), log(), make_average.sh script

### Community 57 - "Surface Averaging Script"
Cohesion: 0.67
Nodes (3): find_chunk_pairs(), main(), notebooks/feature_extraction/whisper_transcribe_bin2s.py =======================

### Community 62 - "TopoOmni Sheet Coordinates"
Cohesion: 0.67
Nodes (3): load_positions() (coords.npy fallback), qwen2_5_omni.py (TopoOmni model source, unified_sheet logic), unified_sheet collapses to empty for clips <~16-17s (traced empirically) (rationale)

## Knowledge Gaps
- **67 isolated node(s):** `PYCORTEX_FILESTORE`, `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS` (+62 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `preprocess_subject()` connect `fMRI Preprocessing (GSR/PSC/SG)` to `K-Reliability RSA`, `Encoding Pipeline Core`, `Seed Connectivity Analysis`, `CF Model Fitting`, `Noise Ceiling Computation`, `Glasser Parcellation Loader`, `Crossnobis Searchlight`?**
  _High betweenness centrality (0.131) - this node is a cross-community bridge._
- **Why does `get_bm_axis()` connect `Integration Convergence Maps` to `K-Reliability RSA`, `Encoding Pipeline Core`, `Localizer Readout Comparison`, `Partial RSA (banded projection)`, `Cortex Map Plotting`, `RDM-Diagonal Grouped Searchlight`, `Seed Connectivity Analysis`, `TFCE Group Stats (encoding)`, `Multimodal CKA Decomposition`, `Noise Ceiling Computation`, `RSA Group Stats (TFCE/FDR)`, `Glasser Parcellation Loader`, `CIFTI I/O Core`, `Crossnobis Searchlight`, `RSA Border Drawing`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Why does `preprocess_fmri()` connect `fMRI Preprocess/Bin Utils` to `K-Reliability RSA`, `Partial RSA (banded projection)`, `RDM-Diagonal Grouped Searchlight`, `Multimodal CKA Decomposition`, `Noise Ceiling Computation`, `Delta-Rho Comparison`, `RSA Timing/HRF Utils`, `Glasser Parcellation Loader`, `Crossnobis Searchlight`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `preprocess_fmri()` (e.g. with `test_preprocess_fmri_bin_shape_multi_clip()` and `test_preprocess_fmri_delay()`) actually correct?**
  _`preprocess_fmri()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `cf_modeling/00_make_roi_masks.py ================================== Generate per`, `Extract Boolean vertex masks for each ROI from the Glasser dlabel.      Paramete`, `Write {roi}_{L/R}_mask.csv files in vicsompy's expected format.      Format: sin` to the rest of the system?**
  _410 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `K-Reliability RSA` be split into smaller, more focused modules?**
  _Cohesion score 0.057511737089201875 - nodes in this community are weakly interconnected._
- **Should `Encoding Pipeline Core` be split into smaller, more focused modules?**
  _Cohesion score 0.06957047791893527 - nodes in this community are weakly interconnected._