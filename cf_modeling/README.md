# cf_modeling: connective-field models

Fits connective-field (CF) models of cortical 7T movie-watching fMRI. For a pair of cortical regions of interest (ROIs, Glasser HCP-MMP1 parcels), the activity of every cortical grayordinate is predicted from the spatial pattern of activity inside each ROI, expressed in a low-dimensional geometric basis. The method follows Hedger et al. (2025); the model classes come from the `vicsompy` source repository (imported directly, not pip-installed; see `NOTICE`).

## What is computed

1. **Basis (script 01).** For each ROI and hemisphere, the Laplace-Beltrami operator of the ROI's midthickness subsurface is built, and the `k` eigenfunctions with eigenvalues closest to zero (the Laplace-Beltrami operator eigenfunctions, LBOEs, i.e. the smoothest spatial patterns on that patch) are kept. `k = min(N_LBOE, n_left - 2, n_right - 2)`, where `n_left`, `n_right` are the ROI's vertex counts. A shift of 1e-6 is added in the sparse eigensolver for numerical stability.
2. **Design matrix (script 02).** With `D_{r,h}` the (ROI vertices x time) training data of ROI `r` in hemisphere `h` and `E_{r,h}` its (ROI vertices x `k`) eigenvector matrix, the predictor block is `X_{r,h} = D_{r,h}^T E_{r,h}` (time x `k`). The design matrix is `[X_{A,L} | X_{A,R} | X_{B,L} | X_{B,R}]`, i.e. two bands (one per ROI) of `2k` columns each.
3. **Banded ridge regression.** Each band is standardized and turned into a linear kernel; himalaya `MultipleKernelRidgeCV` (solver `random_search`, `n_iter` draws, regularization grid `logspace(1, 20, 20)`) learns one regularization strength per band and per target grayordinate. Hyperparameters are chosen by leave-one-run-out cross-validation inside the training data. Targets are all 59,412 cortical grayordinates.
4. **Train/test split.** In each run the last `--n-test-trs` (default 103) time points are held out; every run's training and test segments are z-scored separately along time, then concatenated across runs.
5. **Held-out scores.** On the test segments:
   - `full R^2`: coefficient of determination of the two-ROI model, per grayordinate.
   - `split R^2` for ROI `r`: himalaya `r2_score_split`, the part of the held-out R^2 attributed to band `r`.
   - `ROI-mean null R^2` for ROI `r`: R^2 of an ordinary least squares fit `y ~ intercept + b * m_r(t)` on the test time points, where `m_r(t)` is the mean time series over all vertices of ROI `r`, both hemispheres (equal to the squared Pearson correlation between `m_r` and `y`).
   - `null-corrected split R^2` for ROI `r`: `split R^2_r - null R^2_r`.
   - `joint geometric mean`: `sqrt(max(a, 0) * max(b, 0))` of the two ROIs' split R^2 (or null-corrected split R^2).
   - `modality balance` (null-corrected): `(a_nc - b_nc) / (|a_nc| + |b_nc| + 1e-8)`, in [-1, 1]; positive where ROI A explains more.
6. **Two estimates.** `group_average` fits one model to the across-subject mean time series. `per_subject` fits each subject separately and then takes the vertex-wise `nanmean` of the subject maps (`integration_maps.py`) and a one-sample t-test across subjects (`overlap.py`).

Splicing of lookup tables from vicsompy is not used.

## Pipeline

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `00_make_roi_masks.py` | Optional. Write `{roi}_{L,R}_mask.csv` (column `mask`, 59,292 rows) from the Glasser dlabel. Script 01 reads the dlabel directly and only uses these CSVs as a fallback for ROI names not in the dlabel. |
| 1 | `01_extract_geometry.py` | Build subsurfaces and LBOEs for the ROI pair. |
| 2 | `02_fit_cf_model.py` | Fit the model, score it on held-out time points, compute the null models. |
| 3 | `integration_maps.py` | Combine the maps into one CIFTI (`group_average`: read from `prep/`; `per_subject`: average the subject maps). |
| 3b | `export_bivariate_cifti.py` | Two-dimensional colour map (ROI A split R^2 vs ROI B split R^2) as a Workbench dlabel. |
| 4 | `overlap.py` | `per_subject` only. Across-subject statistics. |
| optional | `03_functional_masks.py` | Derive data-driven ROI masks from a finished fit. |

Supporting modules: `cf_naming.py` (output-name rules), `bivariate_cifti.py` (colour quantization and legends), `lib/`, `shared/`, `vendor/` (superseded verbatim copies of vicsompy code, kept for attribution; see `vendor/DEPRECATED.md`).

## Running

Environment: conda environment `movie` (`environment.yml`; himalaya 0.4.11, pycortex, nibabel). Inputs assumed by the runner (paths in its CONFIG block): the Glasser 59k dlabel, the pycortex filestore, preprocessed group-average and individual CIFTIs, and `subjects.txt`.

Runner: `bash cf_modeling/run_analysis.sh [MODE] [BATCH_SIZE] [START_FROM]`

| MODE | Action |
|------|--------|
| `masks` | Step 0 for all ROIs in the pair lists. |
| `geometry` | Step 1 for all pairs (skipped if both `sub_*.pkl` caches exist). |
| `preprocess` | Run `preprocess_individual.py` (Savitzky-Golay, percent signal change, optional global signal regression per run) and move the group average to `PREPROCESSED_DIR`. |
| `avg` (alias `groupaverage`) | For each `AVG_PAIRS` entry: steps 1, 2, 3, 3b. |
| `persubject` | For each `PERSUBJECT_PAIRS` entry: step 1, step 2 for every subject in `SUBJECTS_LIST` (GNU `parallel` with `BATCH_SIZE` jobs, sequential if absent), then steps 3, 3b, 4. `START_FROM` resumes from that subject ID. |
| `all` (default) | `geometry`, `avg`, `persubject`. |

Runner CONFIG variables (edit at the top of `run_analysis.sh`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `VICSOMPY_REPO` | `.../Vicarious_somatotopy` | vicsompy source checkout |
| `CX_SUB`, `SURF_TYPE` | `hcp_999999_draw_NH`, `fiducial` | pycortex subject and surface (fiducial is the midthickness) |
| `N_LBOE` | 100 | maximum LBOEs per ROI and hemisphere; the runner appends `_lboe{N}` to ROI names, so pair directories read `A5_lboe100_FFC_lboe100` |
| `STREAM` | `false` | `true`: preprocess raw 7T CIFTIs from `CIFTI_DIR` on the fly (per-subject only); `false`: read `PREPROCESSED_INDIV_DIR` |
| `SG_FILTER`, `PSC`, `GSR` | `true`, `true`, `false` | preprocessing flags; percent signal change uses the pre-filter run mean |
| `FMRI_GROUP_CIFTI`, `FMRI_GROUP_CIFTI_FULLBRAIN`, `FMRI_GROUP_RUN_TRS` | Hedger pseudo-subject 999999 (`hedger_sg_psc`) | group-average input; the full-brain file (170,494 grayordinates) is used for fitting and maps are sliced to cortex; build the files with `concat_hedger_average.py` |
| `BACKEND` | `torch_cuda` | himalaya backend: `torch_cuda`, `torch`, `numpy` (02 also accepts `cupy`); falls back to `torch` without CUDA |
| `N_ITER`, `N_TARGETS_BATCH` | 20, 20000 | random-search draws; targets per GPU batch |
| `PERSUBJECT_PAIRS`, `AVG_PAIRS` | arrays of `"ROI_A:ROI_B"` | pairs to analyse |
| `_CF_PRUNE_BETAS` (environment) | `false` | `true` deletes each subject's `betas_*.npy` after a successful fit |

Single-step commands (flags with defaults; `--mode` is `group_average` or `per_subject`):

```bash
python cf_modeling/01_extract_geometry.py --mode group_average --roi-a A5 --roi-b FFC \
    [--n-lboe 100] [--pycortex-store DIR] [--cx-sub hcp_999999_draw_NH] [--surf-type fiducial] \
    [--glasser-dlabel FILE] [--masks-dir DIR] [--output-base DIR] [--vicsompy-repo DIR]

python cf_modeling/02_fit_cf_model.py --mode group_average --roi-a A5 --roi-b FFC \
    --preprocessed-dir DIR --fmri-suffix sg_psc --template-cifti FILE \
    [--fmri-fullbrain-path FILE] [--n-test-trs 103] [--tr 1.0] [--backend torch_cuda] \
    [--n-iter 20] [--n-targets-batch 20000] [--output-base DIR]
# per_subject: add --subject ID; use --raw-dir DIR (streaming, with --sg-filter --psc [--gsr])
# instead of --preprocessed-dir. --preprocessed-dir and --raw-dir are mutually exclusive.

python cf_modeling/integration_maps.py --mode group_average --roi-a A5 --roi-b FFC \
    --template-cifti FILE [--min-subjects 1] [--output-base DIR] [--pycortex-store DIR]

python cf_modeling/export_bivariate_cifti.py --mode group_average --roi-a A5 --roi-b FFC \
    --template-cifti FILE [--bins 32] [--vmin 0] [--vmax 0.4] [--colormap-png FILE] [--output-dir DIR]

python cf_modeling/overlap.py --mode per_subject --roi-a A5 --roi-b FFC \
    --template-cifti FILE [--min-subjects 1]
```

`--roi-a/--roi-b` must be the same strings (including any `_lboeN` suffix) in every step, because they name the pair directory. `--template-cifti` is the 59k cortex-only dtseries used for CIFTI headers. A fit is skipped when both null-corrected split R^2 files already exist in its output directory.

`03_functional_masks.py` reads the null-corrected arrays from `prep/` (`--mode group_average`) or the `per_subject_mean_*` arrays written by `integration_maps.py` in `group/` (`--mode per_subject`). Arguments (required: `--source-roi-a`, `--source-roi-b`, `--mode`, `--template-cifti`; defaults: `--min-r2-full 0`, `--threshold-method percentile`, `--threshold-q 50`, `--threshold-abs 0`, `--overlap-method winner`, output names `{source}_fx`) keeps vertices with full R^2 above `--min-r2-full` and null-corrected split R^2 above the `--threshold-q` percentile of those vertices (floored at 0) for each ROI, resolves vertices selected by both (`winner`: higher null-corrected R^2; `allow`; `exclude`), and writes `{name}_{L,R}_mask.csv`, `functional_masks_{A}_{B}.dscalar.nii` (maps: mask A, mask B, valid vertices) and `.dlabel.nii` to `--masks-dir` (default `{output-base}/masks`).

## Outputs

Root `outputs/cf_modeling/` (`--output-base`). Pair directory `{mode}/{ROI_A}_{ROI_B}/`:

| Path | Content |
|------|---------|
| `subsurfaces/sub_{roi}.pkl`, `{roi}_subsurface.pickle` | Subsurface with eigenvectors (identical copies; `roi` lowercased) |
| `subsurfaces/lboe_provenance_{ROI_A}_{ROI_B}.json` | requested vs actual LBOE count, vertex counts |
| `prep/` (`group_average`) or `subjects/{subject}/` (`per_subject`) | the arrays below, plus `betas_{ROI}.npy` (ridge weights), `train_scores.npy`, `test_scores.npy`, `best_alphas.npy` (written by vicsompy) and `band_sizes.npy` (columns per band, `[2k_A, 2k_B]`) |
| `subjects/{subject}/pipeline.log` | runner log (per-subject) |
| `group/` (`per_subject`) | `per_subject_mean_*.npy` (subject means of each map below) |
| `group/stats/`, `group/cifti_maps/` (`per_subject`) | output of `overlap.py` |
| `cifti_maps/` | `.dscalar.nii`, `.dlabel.nii` and legend PNGs only. `integration_maps.py` writes the combined CIFTI here for `group_average` and to `group/cifti_maps/` for `per_subject`; `export_bivariate_cifti.py` writes to `cifti_maps/` in both modes |
| `cf_model_maps_{roi_a}_{roi_b}_{estimate}.json` | provenance, in the parent directory of the combined CIFTI's `cifti_maps/` |

Array names (`.npy`, shape `(59412,)`, one value per cortical grayordinate):

| File | Meaning |
|------|---------|
| `cf_model_full_r2` | held-out R^2 of the two-ROI model |
| `cf_model_split_r2_{ROI}` | split R^2 of that ROI's band |
| `cf_model_roi_mean_null_r2_{ROI}` | null R^2 from that ROI's mean time series |
| `cf_model_null_corrected_split_r2_{ROI}` | split R^2 minus null R^2 |
| `cf_model_joint_split_r2_geomean`, `cf_model_joint_null_corrected_split_r2_geomean` | geometric mean over the two ROIs |

Combined CIFTI (`integration_maps.py`): `cf_model_maps_{roi_a}_{roi_b}_{estimate}.dscalar.nii` with ROI names lowercased and `estimate` = `fit_to_group_mean_timeseries` (`group_average`) or `mean_of_subject_maps` (`per_subject`). It holds the nine arrays above plus the map `cf_model_null_corrected_split_r2_balance` (modality balance).

Bivariate export (`export_bivariate_cifti.py`): `bivariate_cf_model_split_r2_{ROI_A}_{ROI_B}_{estimate}_{bins}bin.dlabel.nii` (each grayordinate coloured by the bin of ROI B on the horizontal axis and ROI A on the vertical axis of the pycortex `PU_RdBu_covar_alpha` colour map), `..._axes.dscalar.nii` (the two continuous maps) and `..._legend.png`.

`overlap.py` (`per_subject/{pair}/group/`), per grayordinate across the `N` subjects, for three subject-level maps: null-corrected split R^2 of ROI A, of ROI B, and the joint (not null-corrected) geometric-mean map ("integration"):
- `stats/t_*`, `d_*`, `p_*`: one-sample t statistic `t = mean / (sd / sqrt(N))` against 0, Cohen's d `= t / sqrt(N)`, and two-sided p (names `t_{ROI}_nc`, `d_{ROI}_nc`, `p_{ROI}_nc`, `t_integration`, ...).
- `stats/fdr_mask_{ROI}_nc.npy`: 1 where the Benjamini-Hochberg-adjusted p is below 0.05 (ROI A and ROI B maps only).
- `stats/stats_summary.json`: `N`, critical t (two-sided, 0.05, `N-1` degrees of freedom), mean d, counts of vertices passing the corrected and uncorrected thresholds.
- `cifti_maps/{name}.dscalar.nii`: t, d and FDR masks.

Legacy array names `R2_*`, `product_map*`, `*_avg` are still read when the new names are absent; new files always use the names above.

## ROI-mean time-series correlation (not a CF model)

Companion maps that use no LBOEs. `m_r(t)` is the mean time series of ROI `r`; every vertex series and `m_r` are z-scored within each run (all time points kept), and the Pearson r is the mean product over time.

| Script | Output |
|--------|--------|
| `roi_mean_raw_connectivity.py` | zero-order `r(vertex, m_A)`, `r(vertex, m_P)` |
| `roi_mean_partial_connectivity.py` | partial `r(vertex, m_A \| m_P) = (r_yA - r_yP r_AP) / sqrt((1 - r_yP^2)(1 - r_AP^2))` and the symmetric map |
| `export_raw_corr_bivariate_cifti.py`, `export_partial_bivariate_cifti.py` | four 2-D dlabels each (`bilateral`, `within_hemisphere`, `L`, `R`) |
| `make_glasser_roi_mask.py` | bilateral Glasser ROI mask `.dscalar.nii`, the format of `--mask-a/--mask-p` below (`--glasser-dlabel`, `--template-cifti`, `--roi`, `--output-path`) |

```bash
python cf_modeling/roi_mean_partial_connectivity.py --dtseries FILE --run-trs FILE.npy \
    --roi-a A5 --roi-p FFC --mask-a A5_mask.dscalar.nii --mask-p FFC_mask.dscalar.nii \
    --output-dir DIR/cifti_maps [--preprocessing raw|sg_psc] [--supporting-dir DIR] [--batch-size 4096]
```

`roi_mean_raw_connectivity.py` takes the same flags. The exporters take `--partial-cifti` / `--zero-order-cifti`, `--roi-a`, `--roi-p`, `--output-dir`, `--preprocessing-label {raw,sg_psc}_per_run_zscore` (required), `--bins 32 --vmin 0 --vmax 0.4 --colormap-png`.

`--preprocessing raw` (default) reads the unfiltered continuous group average; `sg_psc` reads the Savitzky-Golay plus percent-signal-change group average. The label `{raw,sg_psc}_per_run_zscore` is part of every file name. The `bilateral` maps pool both hemispheres' ROI means; `within_hemisphere` correlates left vertices with the left ROI mean and right vertices with the right ROI mean; `L`/`R` keep one hemisphere (other grayordinates NaN). Stems: `roi_mean_timeseries_{zero_order,partial}_pearson_r_{A}[_given_{P}]_{scope}_{label}`; combined `.dscalar.nii` is `roi_mean_timeseries_{zero_order,partial}_pearson_r_{A}_{P}_{label}.dscalar.nii` (8 maps in each). The four bilateral/within-hemisphere maps are also written as `.npy`, with a `.json` provenance file, to `--supporting-dir` (default: parent of `--output-dir`). Pair collections live in `group_average/roi_mean_timeseries_connectivity_{A}_{P}/` (`cifti_maps/` for `.dscalar.nii`/`.dlabel.nii`/legend only). `run_analysis.sh` has no mode that runs these scripts; call them directly.

## Utilities

| Script | Purpose |
|--------|---------|
| `concat_hedger_average.py` | Concatenate Hedger's four per-run `sg_psc` CIFTIs of pseudo-subject 999999 into the cortex-only and full-brain group-average dtseries plus `run_trs.npy` (paths hard-coded in the script). No flags. |
| `migrate_lboe_artifacts.py` | `python cf_modeling/migrate_lboe_artifacts.py OUTPUT_BASE [--default-lboe 200] [--apply]` renames fitted pair trees to the `_lboe{N}` convention; dry run unless `--apply`; writes `lboe_naming_migration_manifest.json`. |
| `validate_hedger_data.py` | Concatenates the four `sg_psc` runs of Hedger's subject 999999, runs `01_extract_geometry.py` and `02_fit_cf_model.py` on them (3b × V1, 200 LBOEs, output under `validation_hedger999999/`), and prints mean R² and the fraction of positive grayordinates beside our group-average fit. |
| `viz.ipynb` | Flatmap figures of CF model outputs (pycortex). |
| `functional_masks_cf.ipynb` | Notebook for the functional-mask CF analysis (auditory and visual cortex masks from an A1 x V1 fit). |

## lib/, shared/, deprecated/

See `lib/README.md`, `shared/README.md`, `deprecated/README.md`. The CCA-ROI connective-field work is in `deprecated/`.

## Tests

From the repository root, with the `movie` environment:

```bash
pytest tests/test_cf_math.py tests/test_lboe_naming.py tests/test_bivariate_cifti.py \
       tests/test_partial_bivariate_cifti.py tests/test_raw_corr_bivariate_cifti.py \
       tests/test_roi_mean_partial_connectivity.py tests/test_roi_mean_raw_connectivity.py -v
```

They use synthetic data: ridge/LBOE-projection/null-R^2 math (`shared/ridge_utils.py`), the `_lboeN` naming and per-subject root rules, bivariate quantization and exporters, and the two ROI-mean correlation scripts.
