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
| `roi_mean_raw_connectivity.py` | Legacy script name; writes zero-order Pearson r between every vertex and an ROI-mean fMRI time series (not a CF map) | pair `cifti_maps/` |
| `roi_mean_partial_connectivity.py` | Partial Pearson r between every vertex and an ROI-mean time series, controlling the other ROI mean (not a CF map) | pair `cifti_maps/` |
| `export_raw_corr_bivariate_cifti.py` | Legacy script name; exports zero-order ROI-mean Pearson-r maps as dual-color Workbench labels | pair `cifti_maps/` |
| `export_bivariate_cifti.py` | Export split-CF R² maps as dual-color Workbench labels | `outputs/cf_modeling/{group_average\|per_subject}` |
| `export_partial_bivariate_cifti.py` | Export partial ROI-mean Pearson-r maps as dual-color Workbench labels | pair `cifti_maps/` |
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

## Output names and meanings

New output names use the calculation as their leading token; `raw` is not used
because it previously meant two unrelated things.  Within a fitted pair's
`prep/` (or subject) directory, the CF arrays are:

| Name | Meaning |
|------|---------|
| `cf_model_full_r2.npy` | Held-out R² of the full two-ROI LBOE CF model. |
| `cf_model_split_r2_{ROI}.npy` | Held-out R² assigned to that ROI's split-CF predictor block; it is neither raw BOLD nor a correlation. |
| `cf_model_roi_mean_null_r2_{ROI}.npy` | Held-out R² of the one-regressor null model made from that ROI's mean fMRI time series. |
| `cf_model_null_corrected_split_r2_{ROI}.npy` | Split-CF R² minus its ROI-mean null R². |
| `cf_model_joint_*_r2_geomean.npy` | Geometric mean across the two ROI split-CF maps, before or after null correction. |

The group-average path has a precise meaning: it fits one CF model to the
across-subject mean fMRI time series.  Its combined CIFTI is named
`cf_model_maps_{ROI_A}_{ROI_B}_fit_to_group_mean_timeseries.dscalar.nii`.
By contrast, the per-subject pipeline fits each person first and then writes
`*_mean_of_subject_maps.dscalar.nii` plus `per_subject_mean_cf_model_*.npy`.

`roi_mean_timeseries_zero_order_pearson_r_*` maps are ordinary, runwise
standardized Pearson correlations of a vertex and an ROI-mean time series;
they do not use CF modeling and control no confound.  The corresponding
`roi_mean_timeseries_partial_pearson_r_{A}_given_{P}_*` maps control the other
ROI mean.  Bivariate label exports repeat these full terms in their stems.

Each ROI-mean correlation is written twice, with an explicit preprocessing
suffix. `sg_psc_per_run_zscore` reads the `sg_psc` group average:
within each run it has Savitzky–Golay detrending followed by PSC scaling using
that run's pre-detrending mean, then this correlation analysis z-scores vertex
and ROI-mean series within run. `raw_per_run_zscore` reads the raw
continuous group average (no SG, PSC, or GSR); this correlation
analysis likewise z-scores both series within run. These are TR-level
correlations—not RSA's movie-segment/binning step—and neither variant changes
or is used by the fitted CF model.

CF readers accept legacy `R2_*`, `product_map*`, and `_avg` array names as a
read-only fallback. All new writes use the names above.

Within every pair directory, `cifti_maps/` is reserved for `.dscalar.nii` and
`.dlabel.nii` files plus the PNG legend that interprets a bivariate dlabel.
ROI-mean component `.npy` backups and JSON provenance live directly in the
pair directory.  Pycortex views of fitted LBOE CF models belong in that pair's
`figures/` directory; no figures are generated for ROI-mean correlations.

Pair-directory names identify both source ROIs and any selection variant.
`A5_lboe200_FFC_lboe200` is the Glasser A5–FFC CF fit with 200 LBOEs per
ROI/hemisphere. `cca_a_peav_1pct_lboe200_cca_p_peav_1pct_lboe200` is the CF
fit using the top 1% of the PE-AV group RSA map and 200 LBOEs per
ROI/hemisphere.

ROI-mean time-series correlations do not use LBOEs and therefore live in
separate, LBOE-free collections:

- `roi_mean_timeseries_connectivity_A5_glasser_FFC_glasser`
- `roi_mean_timeseries_connectivity_cca_a_peav_top_1pct_cca_p_peav_top_1pct`

Every artifact stem contains `zero_order_pearson_r` or `partial_pearson_r`,
the ROI definition, and one preprocessing tag:

- `sg_psc_per_run_zscore`
- `raw_per_run_zscore`

The zero-order and partial-correlation CLIs accept
`--preprocessing raw|sg_psc`. The default is `raw`; the CF runner explicitly
requests `sg_psc` for its standard branch and also runs the `raw` comparison.

Bivariate artifacts additionally contain the scope tag `bilateral`,
`within_hemisphere`, `L`, or `R`, followed by the bin-count tag such as
`32bin`.

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
