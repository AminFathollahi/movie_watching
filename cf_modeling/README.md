# CF Modeling — Connective Field Models

Fits banded ridge connective field models predicting cortical fMRI from Laplace-Beltrami Operator Eigenfunctions (LBOEs) of two Glasser ROI subsurfaces, then derives audiovisual integration and connectivity maps via null-corrected variance partitioning and partial correlation.

## Scripts

| Script | Purpose | Output Directory |
|--------|---------|------------------|
| `00_make_roi_masks.py` | Export Glasser ROI binary masks to CSV (optional; geometry scripts read dlabel directly) | `outputs/cf_modeling/masks` |
| `01_extract_geometry.py` | Build Subsurface objects and compute LBOEs (up to 100 per ROI, auto-capped for small ROIs) | `outputs/cf_modeling/{group_average\|per_subject}/{ROI_A}_{ROI_B}/subsurfaces` |
| `02_fit_cf_model.py` | Main CF model fitting: banded ridge regression on grayordinate targets (59k vertices) | `outputs/cf_modeling/{group_average\|per_subject}/{ROI_A}_{ROI_B}/prep` |
| `03_functional_masks.py` | Define functional ROI masks from connectivity patterns | outputs path varies |
| `integration_maps.py` | Audiovisual integration scores, modality balance, integration masks | `outputs/cf_modeling/{group_average\|per_subject}/{ROI_A}_{ROI_B}/cifti_maps` |
| `overlap.py` | RSA overlap with PE-AV searchlight RSA (group-average) or group statistics (per-subject) | `outputs/cf_modeling/{group_average\|per_subject}/{ROI_A}_{ROI_B}/results` |
| `roi_mean_raw_connectivity.py` | Pearson correlation between each vertex and ROI mean timecourse (no confound control) | `outputs/cf_modeling/connectivity` |
| `roi_mean_partial_connectivity.py` | Partial correlation controlling for the other ROI mean timecourse | `outputs/cf_modeling/connectivity` |
| `export_raw_corr_bivariate_cifti.py` | Export raw-correlation maps as dual-color Workbench labels (Figure 3a style) | `outputs/cf_modeling/connectivity` |
| `export_bivariate_cifti.py` | Export null-corrected CF R² maps as dual-color Workbench labels | `outputs/cf_modeling/{group_average\|per_subject}` |
| `export_partial_bivariate_cifti.py` | Export partial-correlation maps as dual-color Workbench labels | `outputs/cf_modeling/connectivity` |
| `roi_hierarchy.py` | Two-stage mediation/latency/integration-window directionality tests: upstream (a5, ffc -> cca_a, cca_p) and frontal (cca_a, cca_p -> Glasser frontal parcels) | `outputs/cf_modeling/roi_hierarchy/{upstream\|frontal}/group_average/{model}/{stage}/` |
| `frontal_gradient_map.py` | Winner-take-all whole-cortex gradient map comparing cca_a/cca_p connectivity (unlabelled where max r <= 0) to define the dominant source per grayordinate | `outputs/cf_modeling/roi_hierarchy/frontal_gradient_map/` |
| `channel_cca_analysis.py` | AV-channel-class restricted CF analysis (audio-only, video-only, both, neither) | `outputs/cf_modeling/channel_cca_preference` |
| `persubject_cca_maps.py` | Per-subject CF fits for CCA-defined anterior/posterior temporal regions | `outputs/cf_modeling/persubject_cca_1pct` |
| `persubject_cca_channel_1pct.py` | Per-subject channel-wise CF analysis for CCA 1% regions | `outputs/cf_modeling/persubject_cca_1pct/channel_analysis` |
| `run_cca_islands.py` | Threshold PE-AV/Nemotron/Topo-Omni RSA maps at 1% global → extract largest bilateral temporal components → pair by centroid | `outputs/cf_modeling/cca_masks` |
| `compare_lboe_sensitivity.py` | Grid search: LBOE count sensitivity for fixed ROI pairs | `outputs/cf_modeling/lboe_sensitivity` |
| `validate_hedger_data.py` | Validate reproducibility of Hedger et al. 2025 data/preprocessing pipeline | utility/validation |
| `concat_hedger_average.py` | Concatenate separate Hedger group-average CIFTIs into unified output | outputs path varies |
| `migrate_lboe_artifacts.py` | Rename artifacts to include LBOE count qualifier (e.g., `roi_name_lboe100`) | utility/migration |
| `migrate_cca_artifacts.py` | Rename CCA artifacts for consistency (old → `cca_a`/`cca_p`) | utility/migration |
| `cf_naming.py` | Shared utilities for consistent output file naming | utility module |

## Runners

`analysis.sh` modes:
- `bash analysis.sh all` — Full pipeline (geometry → group-average → per-subject)
- `bash analysis.sh geometry` — Build subsurfaces + LBOEs only
- `bash analysis.sh avg` — Group-average CF fitting
- `bash analysis.sh persubject [N_JOBS] [RESUME_SUBJECT]` — Per-subject fitting (sequential or parallel)

`run_analysis.sh` modes:
- `bash run_analysis.sh cca` — CCA-region CF pipeline (threshold PE-AV 1%, extract components, pair, fit)
- `bash run_analysis.sh cca_peav_0p104` — CCA variant with PE-AV threshold 0.104
- `bash run_analysis.sh cca_hemi_saddle_selected` — Hemisphere-adaptive saddle threshold for CCA components
- `bash run_analysis.sh cca_prepare_variants` — All threshold/peak variants (surface-connected AMPLE 70–90%)
- `bash run_analysis.sh roi_hierarchy` — Two-stage (upstream, frontal) mediation/latency/integration-window directionality tests plus the whole-cortex cca_a/cca_p connectivity gradient map (environment overrides available: `ROI_HIERARCHY_MODELS_OVERRIDE`, `ROI_HIERARCHY_STAGE`)

`run_channel_cca_analysis.sh` modes:
- `bash run_channel_cca_analysis.sh analyze` — Channel-level CF analysis (PE-AV, Nemotron, Topo-Omni)
- `bash run_channel_cca_analysis.sh topoomni` — Topo-Omni channel analysis only
- `bash run_channel_cca_analysis.sh nemotron-own-rois` — Nemotron with self-defined CCA masks

## Configuration

Key parameters in `analysis.sh`:
- `VICSOMPY_REPO`: path to vicsompy source (direct import, no pip install)
- `CX_SUB`: pycortex subject (`hcp_999999_draw_NH`)
- `SURF_TYPE`: surface type (`fiducial` = midthickness)
- `N_LBOE`: maximum LBOEs per ROI (default 100, auto-capped for small ROIs)
- `STREAM`: streaming mode (preprocess raw CIFTIs on-the-fly)
- `SG_FILTER`, `PSC`, `GSR`: signal preprocessing
- `BACKEND`: himalaya solver (`torch_cuda` / `torch` / `numpy`)
- `PERSUBJECT_PAIRS`, `AVG_PAIRS`: ROI pairs for analysis
