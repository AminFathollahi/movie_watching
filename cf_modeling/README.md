# cf_modeling: connective-field models

Connective-field (CF) models of cortical 7T movie-watching fMRI, following Hedger et al. (2025). For a pair of cortical regions of interest (ROIs; Glasser HCP-MMP1 parcels), the activity of every cortical grayordinate (108,441, medial wall excluded) is predicted from the spatial pattern of activity inside each ROI, expressed in a low-dimensional geometric basis. The model classes come from the `vicsompy` repository, which is imported from source and not pip-installed (see `NOTICE` and `SETUP.md`).

## Method

1. Basis (`01_extract_geometry.py`). For each ROI and hemisphere, the Laplace-Beltrami operator of the ROI's midthickness subsurface is built and the `k` eigenfunctions with eigenvalues closest to zero (the smoothest spatial patterns on the patch) are kept, with `k = min(N_LBOE, n_left - 2, n_right - 2)` and `n_left`, `n_right` the ROI's vertex counts.
2. Design matrix (`02_fit_cf_model.py`). With `D_{r,h}` the (ROI vertices x time) training data of ROI `r` in hemisphere `h` and `E_{r,h}` its (ROI vertices x `k`) eigenvector matrix, the predictors are `X_{r,h} = D_{r,h}^T E_{r,h}` (time x `k`). The design matrix is `[X_{A,L} | X_{A,R} | X_{B,L} | X_{B,R}]`: two bands, one per ROI, of `2k` columns each.
3. Banded ridge regression. Each band is standardized and turned into a linear kernel. himalaya `MultipleKernelRidgeCV` (solver `random_search`, `n_iter` draws, regularization grid `logspace(1, 20, 20)`) learns one regularization strength per band and per target grayordinate, chosen by leave-one-run-out cross-validation within the training data.
4. Train/test split. The last `--n-test-trs` (default 103) time points of each run are held out. Training and test segments are z-scored separately along time and concatenated across runs.
5. Held-out scores, per grayordinate, on the test segments:
   - full R^2: coefficient of determination of the two-ROI model.
   - split R^2 of ROI `r`: himalaya `r2_score_split`, the part of the held-out R^2 attributed to band `r`.
   - ROI-mean null R^2 of ROI `r`: R^2 of the least-squares fit `y ~ intercept + b * m_r(t)`, where `m_r(t)` is the mean time series over all vertices of ROI `r` in both hemispheres (the squared Pearson correlation between `m_r` and `y`).
   - null-corrected split R^2 of ROI `r`: split R^2 minus null R^2.
   - joint geometric mean of the two ROIs' split R^2 (or null-corrected split R^2): `sqrt(max(a, 0) * max(b, 0))`.
   - modality balance: `(a_nc - b_nc) / (|a_nc| + |b_nc| + 1e-8)` on the null-corrected values, in [-1, 1], positive where ROI A explains more.
6. Two estimates. `group_average` fits one model to the across-subject mean time series. `per_subject` fits each subject separately, then takes the vertex-wise `nanmean` of the subject maps (`integration_maps.py`) and a one-sample t-test across subjects (`overlap.py`).

## Pipeline

| Step | Script | Purpose |
|---|---|---|
| 0 | `00_make_roi_masks.py` | Optional. Writes `{roi}_{L,R}_mask.csv` (column `mask`, 59,292 rows) from the Glasser dlabel. Step 1 reads the dlabel directly and uses these files only for ROI names missing from it. |
| 1 | `01_extract_geometry.py` | Subsurfaces and eigenfunctions for the ROI pair. |
| 2 | `02_fit_cf_model.py` | Fit, held-out scores, null models. |
| 3 | `integration_maps.py` | Combine the maps into one CIFTI (`per_subject`: average the subject maps). |
| 3b | `export_bivariate_cifti.py` | Two-dimensional colour map (ROI A against ROI B split R^2) as a Workbench dlabel. |
| 4 | `overlap.py` | `per_subject` only: statistics across subjects. |
| optional | `03_functional_masks.py` | Data-driven ROI masks from a finished fit. |

Supporting files: `cf_naming.py` (output-name rules), `bivariate_cifti.py` (colour quantization and legends), `lib/`, `shared/`, and `vendor/` (verbatim vicsompy copies kept for attribution, not imported; see `vendor/DEPRECATED.md`).

## Usage

The environment is `movie` (see `SETUP.md`). The runner takes its input and output locations from the CONFIG block at the top of `run_analysis.sh`:

```bash
bash cf_modeling/run_analysis.sh [MODE] [BATCH_SIZE] [START_FROM]
```

| MODE | Action |
|---|---|
| `masks` | Step 0 for all ROIs in the pair lists. |
| `geometry` | Step 1 for all pairs (skipped if both `sub_*.pkl` caches exist). |
| `preprocess` | `preprocess_individual.py` (Savitzky-Golay filter, percent signal change, optional global signal regression per run) and the group average. |
| `avg` | For each `AVG_PAIRS` entry: steps 1, 2, 3, 3b. |
| `persubject` | For each `PERSUBJECT_PAIRS` entry: step 1; step 2 for every subject in `SUBJECTS_LIST` (GNU `parallel`, `BATCH_SIZE` jobs, sequential without it; `START_FROM` resumes at that subject ID); then steps 3, 3b, 4. |
| `all` (default) | `geometry`, `avg`, `persubject`. |

Main CONFIG variables: `N_LBOE` (default 100; the runner appends `_lboe{N}` to ROI names, so a pair directory is `A5_lboe100_FFC_lboe100`), `STREAM` (`true` preprocesses raw 7T CIFTIs on the fly, per-subject only; `false` reads `PREPROCESSED_INDIV_DIR`), `SG_FILTER`, `PSC`, `GSR` (default `true`, `true`, `false`; percent signal change uses the pre-filter run mean), `BACKEND` (`torch_cuda`, `torch` or `numpy`), `N_ITER` (20), `N_TARGETS_BATCH` (20000), and the pair lists `PERSUBJECT_PAIRS` and `AVG_PAIRS` (`"ROI_A:ROI_B"`). The group-average input is the pseudo-subject 999999 of Hedger et al. (2025), built with `concat_hedger_average.py`; the full-brain file is used for fitting and the maps are sliced to cortex. Setting `_CF_PRUNE_BETAS=true` deletes each subject's `betas_*.npy` after a successful fit.

Single steps (`--mode` is `group_average` or `per_subject`; see `--help` for all options):

```bash
python cf_modeling/01_extract_geometry.py --mode group_average --roi-a A5 --roi-b FFC --n-lboe 100

python cf_modeling/02_fit_cf_model.py --mode group_average --roi-a A5 --roi-b FFC \
    --preprocessed-dir DIR --fmri-suffix sg_psc --template-cifti FILE \
    [--fmri-fullbrain-path FILE] [--n-test-trs 103] [--backend torch_cuda] [--n-iter 20]
# per_subject: add --subject ID; --raw-dir DIR --sg-filter --psc [--gsr] (streaming)
# replaces --preprocessed-dir (mutually exclusive)

python cf_modeling/integration_maps.py --mode group_average --roi-a A5 --roi-b FFC --template-cifti FILE
python cf_modeling/export_bivariate_cifti.py --mode group_average --roi-a A5 --roi-b FFC \
    --template-cifti FILE [--bins 32] [--vmin 0] [--vmax 0.4]
python cf_modeling/overlap.py --mode per_subject --roi-a A5 --roi-b FFC --template-cifti FILE
```

`--roi-a` and `--roi-b` must be identical strings (including any `_lboeN` suffix) in every step, since they name the pair directory. `--template-cifti` is the cortex-only dtseries used for CIFTI headers. A fit is skipped when both null-corrected split R^2 files already exist.

`03_functional_masks.py` requires `--source-roi-a`, `--source-roi-b`, `--mode` and `--template-cifti`. For each ROI it keeps vertices with full R^2 above `--min-r2-full` (default 0) and null-corrected split R^2 above the `--threshold-q` percentile (default 50) of those vertices, floored at 0. Vertices selected for both ROIs are resolved by `--overlap-method` (`winner`: higher null-corrected R^2; `allow`; `exclude`). It reads `prep/` (`group_average`) or the `per_subject_mean_*` arrays in `group/` (`per_subject`) and writes `{name}_{L,R}_mask.csv`, `functional_masks_{A}_{B}.dscalar.nii` (mask A, mask B, valid vertices) and `.dlabel.nii` to `--masks-dir` (default `{output-base}/masks`).

## Outputs

Root `outputs/cf_modeling/` (`--output-base`), pair directory `{mode}/{ROI_A}_{ROI_B}/`:

| Path | Content |
|---|---|
| `subsurfaces/` | `sub_{roi}.pkl` and `{roi}_subsurface.pickle` (identical copies, `roi` lowercased), `lboe_provenance_{ROI_A}_{ROI_B}.json` (requested and actual eigenfunction counts, vertex counts) |
| `prep/` (`group_average`) or `subjects/{subject}/` (`per_subject`) | the arrays below, `betas_{ROI}.npy`, `train_scores.npy`, `test_scores.npy`, `best_alphas.npy`, and `band_sizes.npy` (`[2k_A, 2k_B]`) |
| `group/` (`per_subject`) | `per_subject_mean_*.npy`; `stats/` and `cifti_maps/` from `overlap.py` |
| `cifti_maps/` | `.dscalar.nii`, `.dlabel.nii` and legend PNGs. `integration_maps.py` writes the combined CIFTI here for `group_average` and to `group/cifti_maps/` for `per_subject` |
| `cf_model_maps_{roi_a}_{roi_b}_{estimate}.json` | provenance, beside the `cifti_maps/` directory of the combined CIFTI |

Arrays are `.npy` of shape `(108441,)`:

| File | Meaning |
|---|---|
| `cf_model_full_r2` | held-out R^2 of the two-ROI model |
| `cf_model_split_r2_{ROI}` | split R^2 of that ROI's band |
| `cf_model_roi_mean_null_r2_{ROI}` | null R^2 from that ROI's mean time series |
| `cf_model_null_corrected_split_r2_{ROI}` | split R^2 minus null R^2 |
| `cf_model_joint_split_r2_geomean`, `cf_model_joint_null_corrected_split_r2_geomean` | geometric mean over the two ROIs |

The combined CIFTI is `cf_model_maps_{roi_a}_{roi_b}_{estimate}.dscalar.nii` (ROI names lowercased; `estimate` is `fit_to_group_mean_timeseries` for `group_average` and `mean_of_subject_maps` for `per_subject`). It holds the nine arrays above and `cf_model_null_corrected_split_r2_balance` (modality balance).

The bivariate export writes `bivariate_cf_model_split_r2_{ROI_A}_{ROI_B}_{estimate}_{bins}bin.dlabel.nii` (each grayordinate coloured by the bin of ROI B on the horizontal axis and ROI A on the vertical axis of the pycortex `PU_RdBu_covar_alpha` colour map), `..._axes.dscalar.nii` (the two continuous maps) and `..._legend.png`.

`overlap.py` computes, per grayordinate across the `N` subjects, for the null-corrected split R^2 of ROI A, of ROI B, and the joint (not null-corrected) geometric-mean map ("integration"), in `group/`:
- `stats/t_*`, `d_*`, `p_*`: one-sample t statistic `t = mean / (sd / sqrt(N))` against 0, Cohen's d `= t / sqrt(N)`, and two-sided p.
- `stats/fdr_mask_{ROI}_nc.npy`: 1 where the Benjamini-Hochberg-adjusted p is below 0.05 (ROI A and ROI B maps only).
- `stats/stats_summary.json`: `N`, critical t (two-sided, 0.05, `N - 1` degrees of freedom), mean d, and counts of vertices passing the corrected and uncorrected thresholds.
- `cifti_maps/{name}.dscalar.nii`: t, d and the FDR masks.

## ROI-mean time-series correlation

Companion maps without the geometric basis. With `m_r(t)` the mean time series of ROI `r`, every vertex series and `m_r` are z-scored within each run and the Pearson r is the mean product over time.

| Script | Output |
|---|---|
| `roi_mean_raw_connectivity.py` | zero-order `r(vertex, m_A)` and `r(vertex, m_P)` |
| `roi_mean_partial_connectivity.py` | partial `r(vertex, m_A \| m_P) = (r_yA - r_yP r_AP) / sqrt((1 - r_yP^2)(1 - r_AP^2))` and the symmetric map |
| `export_raw_corr_bivariate_cifti.py`, `export_partial_bivariate_cifti.py` | four 2-D dlabels each (`bilateral`, `within_hemisphere`, `L`, `R`) |
| `make_glasser_roi_mask.py` | bilateral Glasser ROI mask `.dscalar.nii`, the format of `--mask-a` and `--mask-p` |

```bash
python cf_modeling/roi_mean_partial_connectivity.py --dtseries FILE --run-trs FILE.npy \
    --roi-a A5 --roi-p FFC --mask-a A5_mask.dscalar.nii --mask-p FFC_mask.dscalar.nii \
    --output-dir DIR/cifti_maps [--preprocessing raw|sg_psc]
```

`roi_mean_raw_connectivity.py` takes the same options. The exporters take `--partial-cifti` or `--zero-order-cifti`, `--roi-a`, `--roi-p`, `--output-dir` and the required `--preprocessing-label {raw,sg_psc}_per_run_zscore`. `--preprocessing raw` (default) reads the unfiltered continuous group average and `sg_psc` the Savitzky-Golay plus percent-signal-change group average; the label is part of every file name. The `bilateral` maps pool both hemispheres' ROI means, `within_hemisphere` correlates left vertices with the left ROI mean and right vertices with the right one, and `L` and `R` keep one hemisphere (other grayordinates NaN). Combined maps are `roi_mean_timeseries_{zero_order,partial}_pearson_r_{A}_{P}_{label}.dscalar.nii`, collected in `group_average/roi_mean_timeseries_connectivity_{A}_{P}/`. `run_analysis.sh` has no mode for these scripts.

## Utilities

| Script | Purpose |
|---|---|
| `concat_hedger_average.py` | Concatenates the four per-run `sg_psc` CIFTIs of pseudo-subject 999999 into the cortex-only and full-brain group-average dtseries and `run_trs.npy`. Locations are set at the top of the script. |
| `migrate_lboe_artifacts.py` | `python cf_modeling/migrate_lboe_artifacts.py OUTPUT_BASE [--default-lboe 200] [--apply]` renames fitted pair trees to the `_lboe{N}` convention; a dry run unless `--apply`. |
| `validate_hedger_data.py` | Fits 3b x V1 (200 eigenfunctions) on Hedger's pseudo-subject 999999 and prints mean R^2 and the fraction of positive grayordinates beside our group-average fit. |
| `viz.ipynb`, `functional_masks_cf.ipynb` | Flatmap figures of the CF outputs; functional-mask analysis for an A1 x V1 fit. |

`lib/`, `shared/` and `deprecated/` have their own READMEs.

## Tests

```bash
pytest tests/test_cf_math.py tests/test_lboe_naming.py tests/test_bivariate_cifti.py \
       tests/test_partial_bivariate_cifti.py tests/test_raw_corr_bivariate_cifti.py \
       tests/test_roi_mean_partial_connectivity.py tests/test_roi_mean_raw_connectivity.py -v
```
