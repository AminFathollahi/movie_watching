# RSA: representational similarity analysis

Compares cortical activity in the HCP 7T movie-watching fMRI data with the representational geometry of model embeddings (audio, video and joint audio-video models). RSA is computed as a vertex-wise searchlight on the cortical surface or per Glasser parcel. Further scripts provide group inference, partial and residual-embedding RSA, audio-video integration controls, noise ceilings and an analysis of the Topo-Omni cortical sheet.

Run from the repository root; environment setup is in [`SETUP.md`](../SETUP.md). Shared modules are described in [`shared/`](shared/README.md). Searchlight centered kernel alignment is not RSA and lives in [`cka/`](../cka/README.md).

## Method

### Binning and normalization

For each clip in `data/movie_timing.csv` (columns `video_id, onset_sec, end_sec, duration_sec, run_id`; `onset_sec` is global time), windows of `bin_sec` seconds start at `onset_sec + delay_sec` and advance by `skip_sec` (default `bin_sec`). A window that would extend past the end of its run is dropped. The fMRI value of a window is the mean over its repetition times (TRs), so a clip yields `floor((duration_sec - bin_sec) / skip_sec) + 1` windows. Embeddings are read from precomputed files with the same bin and stride; the window counts of brain and model are asserted equal.

Within each run, each vertex time series of window means is z-scored. Each embedding dimension is normalized within each run with `--model-norm`: `center` (subtract the mean; default) or `zscore` (also divide by the standard deviation). With `--hrf`, each clip's embedding is convolved with the SPM canonical hemodynamic response function and the delay label in output names becomes `hrf`.

### Searchlight and parcel RSA

The searchlight at vertex `v` uses the `k` geodesically nearest vertices on the midthickness surface (the center vertex is excluded; medial-wall vertices are dropped). With `n` windows, the brain representational dissimilarity matrix (RDM) is the `n x n` matrix of correlation distances `1 - r(x_i, x_j)`, where `x_i` is the vector of the neighborhood's fMRI values in window `i`. The model RDM is `1 - r(e_i, e_j)` between the normalized embedding vectors of windows `i` and `j`. The statistic `rho(v)` is the Spearman (`--method spearman`, default) or Pearson (`--method pearson`) correlation between the lower-triangle entries (`n(n-1)/2` pairs) of the two RDMs.

`glasser.py` uses all vertices of a parcel instead of a neighborhood (parcels with fewer than 2 vertices are skipped) and also supports `rho_a` (Kendall's tau).

`searchlight.py` additionally writes `sign(rho) * -log10(p)` maps, with one-tailed `p` from `t = rho * sqrt(df) / sqrt(1 - rho^2)`, `df = n - 2`, treating windows as independent, and Benjamini-Hochberg false-discovery-rate (BH-FDR) corrected versions at q = 0.05. These are descriptive; use the tests below for inference.

### Group statistics

`group_stats.py` takes per-subject maps `rho_s(v)` for `S` subjects and computes the mean `rho`, a one-sample t-test against 0 (no Fisher transform), one-tailed `p` for `rho > 0`, and BH-FDR across vertices. If per-subject temporal-block maps exist, it adds the corrected two-factor bootstrap of Schutt et al. (2023, Eq. 5). The variance of the group mean under resampling of subjects (`var_subj`), of blocks (`var_block`) and of both (`var_both`) is combined as `var_c2f = clip(2(var_subj + var_block) - var_both, max(var_subj, var_block), var_both)`, and `t_c2f = mean_rho / sqrt(var_c2f)` with `df = min(S - 1, n_blocks - 1)`. Blocks are non-overlapping temporal segments, aligned to run boundaries when `n_blocks` equals the number of runs.

`tfce_groupstats.py` applies the t-test to Fisher z values `z = arctanh(rho)` and runs a sign-flipping permutation test with threshold-free cluster enhancement (`mne.stats.permutation_cluster_1samp_test`, `start=0, step=0.2`, surface adjacency, seed 42). A vertex is significant at family-wise error level `alpha` if its TFCE statistic exceeds the `(1 - alpha)` quantile of the null distribution of the maximum TFCE statistic. The two scripts test different quantities (raw `rho` versus Fisher z), so their p-values are not interchangeable.

### Permutation and spin tests

`perm_searchlight.py` (group average) builds the null from `n_perm` synchronized, nonzero circular shifts of the bins inside each run, one shift for all vertices. The null `rho` at vertex `v` is `brain_norm[v] . model_norm[perm_p]` on rank-normalized RDM vectors, and `p_perm[v] = (count(null_rho[v, :] >= rho[v]) + 1) / (n_perm + 1)`.

`run_spin_permutations.py` rotates the sphere coordinates of one map by `n_spin` random rotations and sets `null[s, v] = empirical[spin_index[s, v]]`, with `p[v] = (count(null[:, v] >= empirical[v]) + 1) / (n_spin + 1)` and BH-FDR. Because the null is the same map rotated, this is a hot-spot test of one map, not a test of correspondence between two maps or against zero.

### Partial and residual-embedding RSA

`partial_rsa.py` takes a target model and nuisance models. Each RDM lower triangle is rank transformed (average ranks) when `--method spearman`. The brain vector `y` and target vector `t` are residualized by ordinary least squares on `[1, nuisance_1, ..., nuisance_m]`, and the result is the Pearson correlation of the two residuals (the partial Spearman correlation). Run keys for `--run` are defined in `shared/model_registry.py::PARTIAL_RSA_RUNS`; runs with `kind="integration"` use the target's own audio and video embeddings as nuisance.

Alternatively the joint embedding `J` itself is residualized (`shared/residuals.py`) and passed through the ordinary searchlight. `linear_residual` is the cross-validated ridge residual of `J` after regression on the concatenated nuisance embeddings; `projection_residual` removes, per window, the projection of `J` onto the span of that window's own audio and video vectors (Gram-Schmidt).

### Audio-video contrasts

`av_derived_maps.py` (group average) and `max_uni.py` (per subject, with group inference) compare the joint-model searchlight `rho` (`AV`) with audio-only (`A`) and video-only (`V`) references: conjunction `(AV > 0) & (AV > A) & (AV > V)`, superadditivity `AV - (A + V)`, and maximum-unimodal contrast `AV - max(A, V)`.

### Crossnobis and noise ceiling

`crossnobis_searchlight.py` computes the cross-validated Euclidean distance between windows `t1` and `t2` across `N` subjects from neighborhood patterns `x_i`:
`d(t1, t2) = [ || sum_i x_i(t1) - sum_i x_i(t2) ||^2 - sum_i || x_i(t1) - x_i(t2) ||^2 ] / (N (N - 1))`.
Its expectation is the true squared distance because measurement noise is independent across subjects. The distances over all window pairs are compared with the model RDM by Kendall's tau.

`noise_ceiling.py` takes `RDM_i(v)`, the correlation-distance vector of subject `i`, and `rho_s`, the Spearman correlation of lower-triangle vectors:
`NC_upper(v) = mean_i rho_s[RDM_i, mean_j RDM_j]` and `NC_lower(v) = mean_i rho_s[RDM_i, mean_{j != i} RDM_j]`.
The upper bound includes subject `i` in the group mean; the lower bound is leave-one-out.

## Inputs

Paths are set at the top of `analysis.sh` (`DATA_BASE`, `OUTPUTS_BASE`) and in the other runners. Connectome Workbench (`wb_command`) is required, set by `WORKBENCH` in the runners and `--workbench` in the scripts.

| Input | Location |
|---|---|
| Raw 7T CIFTIs (streaming mode) | `data/individual-59k/` |
| Preprocessed CIFTIs (disk mode) | `data/preprocessed/average_sub/{tag}/group_average_{tag}_cortex_59k.dtseries.nii` (group average) and `data/preprocessed/{tag}/{subject}_{tag}_cortex_59k.dtseries.nii` (subjects, from `analysis.sh preprocess`), each with a `*_run_trs.npy` |
| Timing, subject list | `data/movie_timing.csv`; `data/subjects.txt` (one ID per line, `#` comments allowed) |
| Embeddings | `outputs/model_embeddings/{model}/bin{B}s_skip{S}s/{model}_{modality}.npy`, shape `(n_windows, n_features)` |
| Template CIFTI | `group_average_raw_cortex_59k.dtseries.nii`, used for its brain-model axis (59,412 cortical grayordinates) |
| Surfaces | `data/HCP_S1200_GroupAvg_v1/GroupAverage_59k/CohortAvg.{L,R}.midthickness_MSMAll.59k_fs_LR.surf.gii`; per-subject surfaces from `MIDTHICKNESS_DIR` are used when present |
| Glasser parcellation | `data/HCP_S1200_GroupAvg_v1/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii` |

`{tag}` is `raw` when no signal preprocessing is applied, otherwise the applied steps joined by `_` from `sg` (Savitzky-Golay high-pass filter), `psc` (percent signal change) and `gsr` (global signal regression). Omni-family model names carry a pooling suffix, `_mp` (mean pool) or `_lt` (last token), for example `nemotron_layer18_mp`. A model and modality whose embedding file is missing is skipped with a log message.

`searchlight.py::get_neighbors` returns the `(n_surface_vertices, k)` neighbor array per subject and hemisphere. It uses `{subject}_{hem}_neighbors_k{K}.npy` from `--geodesic-cache-dir` if present, otherwise slices a cached file with larger `K`, otherwise computes geodesic distances with `wb_command -surface-geodesic-distance-all-to-all` (a multi-gigabyte `dconn`, deleted after extraction for subjects and kept for the group average) and saves the result.

## Usage

```bash
bash rsa/analysis.sh [MODE] [METHOD] [BATCH_SIZE] [START_FROM] [N_BLOCKS]
```

| Argument | Values | Default |
|---|---|---|
| `MODE` | `avg`, `preprocess`, `neighbors_avg`, `neighbors`, `persubject`, `groupstats`, `all` | `avg` |
| `METHOD` | `all` (searchlight and Glasser), `searchlight`, `glasser` | `all` |
| `BATCH_SIZE` | Subjects in parallel (GNU parallel) | 4 (8 for `neighbors`) |
| `START_FROM` | Subject ID at which to resume the per-subject loop | none |
| `N_BLOCKS` | Overrides the `N_BLOCKS` variable | 16 |

`avg` runs the group-average RSA for every model in `MODELS`, first running missing audio and video reference maps and appending the derived contrast maps for `av` targets with defined baselines. `preprocess` runs `preprocess_individual.py` for every subject and the group average. `neighbors_avg` and `neighbors` build the neighbor cache for the group-average surface, or additionally for every subject. `persubject` runs per-subject RSA, group statistics over `BLOCKS_SWEEP`, then the noise ceiling. `groupstats` runs inter-subject crossnobis, group statistics and the noise ceiling on existing per-subject maps. `all` runs everything.

```bash
bash rsa/analysis.sh avg searchlight              # group average, searchlight only
bash rsa/analysis.sh persubject searchlight 4     # per subject, 4 in parallel
BIN_SECS="2.0 5.0 10.0" bash rsa/analysis.sh avg  # sweep bin sizes
RSA_MODELS_OVERRIDE="nemotron_layer18_mp:av;topoomni_layer18_sheet_mp:av" bash rsa/analysis.sh avg searchlight
```

Existing outputs are skipped; delete an output to rerun it. The variables at the top of `analysis.sh` set the configuration; the main ones are below (env: can also be set on the command line).

| Variable | Default | Meaning |
|---|---|---|
| `STREAM` | `true` | `true`: `persubject` preprocesses raw CIFTIs on the fly; `false`: it reads pre-saved per-subject CIFTIs |
| `SG_FILTER`, `PSC`, `GSR` | `false` | Signal preprocessing steps (see `{tag}`) |
| `BIN_SECS` (env; `BIN_SEC` for one value), `SKIP_SEC` (env) | `2.0`, `BIN_SEC` | Bin sizes to sweep and window stride, in seconds |
| `DELAY_SEC`, `HRF` | `5.0`, `false` | Hemodynamic delay added to each clip onset; convolve embeddings with the SPM HRF |
| `MODEL_NORMS` (env) | `center` | Space-separated subset of `center`, `zscore` |
| `METHOD`, `GLASSER_METHOD` | `spearman` | Searchlight and parcel comparator |
| `K` | `100` | Searchlight neighborhood size |
| `N_BLOCKS`, `BLOCKS_SWEEP` | `16`, `(4 8 16)` | Temporal blocks saved per subject; `--n-blocks` values for `group_stats.py` (each must divide `N_BLOCKS`) |
| `RUN_PERM` | `false` | Run `perm_searchlight.py` after each group-average searchlight |
| `MODELS`, `LAYERS`, `RSA_MODELS_OVERRIDE` (env) | see file | `model:modality[,modality...]` entries, the layer sweep for the omni families, and a `;`-joined string replacing `MODELS` |
| `SKIP_LAYER_SWEEP`, `SKIP_NOISE_CEILING`, `SKIP_CROSSNOBIS` (env) | `false` | Skip those stages |

## Outputs

`OUTPUT_DIR = outputs/rsa/{tag}`. Names are built by `shared/naming.py`:

```
searchlight config   k{K}_delay{D}s_bin{B}s_skip{S}s_{method}_{model_norm}     (delay{D}s becomes hrf with --hrf)
glasser config       delay{D}s_bin{B}s_skip{S}s_{method}_{model_norm}
file stem            rsa_59k_{tag}_{searchlight config}
```

| Path under `{OUTPUT_DIR}` | Content |
|---|---|
| `group_average/{model}_{modality}/rsa_59k_{tag}_{config}_maps.dscalar.nii` | Combined maps, accumulated across scripts (below) |
| `group_average/{model}_{modality}/{searchlight config}/{stem}_searchlight.npy` | `(59412,)` `rho`; beside it `{stem}_fdr_mask.dscalar.nii`, `{stem}_fdr_{lh,rh}.border` (skipped if the mask is empty or covers over 90% of cortex) and `{stem}_perm{N}_p.npy` (`RUN_PERM=true`) |
| `group_average/{model}_{modality}/{glasser config}/ranked_report.csv` | Columns `parcel, r, p, n_vertices, rank` |
| `subject_data/{subject}/{model}_{modality}/` | Same layout per subject; block maps (`n_blocks > 1`) are `{stem}_searchlight_nblocks{N}.npy`, shape `(N, 59412)` |
| `groupstats/{model}_{modality}/{searchlight config}/group_stats_{S}subs[_nblocks{N}].dscalar.nii` | Maps `mean_rho, t_stat, sigmap_uncorr, sigmap_fdr` (plus `mean_rho_c2f, t_c2f, sigmap_c2f` with block maps); beside it FDR masks, borders and `summary.json` |
| `noise_ceiling/noise_ceiling_{N}subs_k{K}_delay{D}s_bin{B}s_skip{S}s_{method}.dscalar.nii` | Maps `nc_lower`, `nc_upper` |

Scalar maps in `..._maps.dscalar.nii`: `searchlight_{method}_rho` with `..._rho_sigmap_uncorr` and `..._rho_sigmap_fdr` (`searchlight.py`), `glasser_{glasser method}_rho` (`glasser.py`), `searchlight_{method}_sigmap_perm` (`perm_searchlight.py`), and `av_conjunction_{family}`, `av_superadditivity_{family}`, `av_max_uni_{family}` (`av_derived_maps.py`). The reference `family` is `own` (the model's own audio and video maps), `unimodal` (AudioMAE and VideoMAEv2-Large) or `text_aligned` (WavLM-Large and PE-Core ViT-L/14), as defined by `shared/model_registry.py::av_derived_baselines`; scramble, dummy-modality and residual pseudo-models have none.

## Commands by analysis

`<template>`, `<L>`, `<R>` and `<wb_command>` stand for the template CIFTI, the left and right midthickness surfaces and the Workbench executable. Every script accepts `--help`.

### Searchlight and Glasser

```bash
python rsa/searchlight.py \
    --preprocessed-dir data/preprocessed/average_sub/raw --fmri-suffix raw \
    --subject group_average --timing-csv data/movie_timing.csv \
    --embeddings-dir outputs/model_embeddings --template-cifti <template> \
    --left-surface <L> --right-surface <R> --workbench <wb_command> --output-dir outputs/rsa/raw \
    --model pe-av-small-16-frame --modality av \
    --k 100 --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --method spearman --tr 1.0 \
    --model-norm center --geodesic-cache-dir outputs/rsa/_geodesic_cache \
    --combined-output <maps.dscalar.nii> --n-blocks 1
```

Input is either `--preprocessed-dir` with `--fmri-suffix` (disk) or `--raw-dir` with `--sg-filter`, `--psc`, `--gsr`/`--no-gsr` (streaming); the two are mutually exclusive. `--n-blocks` defaults to 4 (1 disables blocks). `glasser.py` takes the same input, timing, delay, binning and `--model-norm` flags plus `--glasser-dlabel`.

Two options give diagnostic variants of the group-average map for comparison with CKA and encoding. Either one (disk mode only) makes the run write a single map, `{OUTPUT_DIR}/group_average/{model}_{modality}/diagnostics/rsa_59k_{tag}_k{K}_delay{D}s_bin{B}s_skip{S}s_{label}_{model_norm}[_norepeats]_maps.dscalar.nii`, and nothing else. `--distance euclidean` replaces the correlation distance by squared Euclidean distance in both RDMs (label `euclid-{method}`; `corr-{method}` by default). `--drop-repeated-clips` drops the windows of `video5`, `video9`, `video14` and `video18` (the clips shown once per run) after per-run normalization and adds `_norepeats` to the name.

### Permutation, spin and group tests

All commands below take the binning flags of the searchlight command (`--k`, `--bin-sec`, `--skip-sec`, `--delay-sec`, `--method`); `perm_searchlight.py` also takes the same input flags.

```bash
python rsa/perm_searchlight.py <searchlight flags> --n-perm 1000 --seed 42 --perm-batch-size 100 --combined-output <maps.dscalar.nii>

python rsa/run_spin_permutations.py --combined-cifti <maps.dscalar.nii> --map-name searchlight_spearman_rho \
    --left-sphere <L.sphere.59k_fs_LR.surf.gii> --right-sphere <R.sphere.59k_fs_LR.surf.gii> \
    --left-surface <L> --right-surface <R> --template-cifti <template> --n-spin 1000 --output-dir outputs/rsa/spin_tests

python rsa/group_stats.py --output-dir outputs/rsa/raw --model pe-av-small-16-frame --modality av \
    --k 100 --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --method spearman --model-norm center \
    --fmri-tag raw --template-cifti <template> --left-surface <L> --right-surface <R> \
    --workbench <wb_command> --n-blocks 16 --n-bootstrap 2000

python rsa/tfce_groupstats.py <same flags except --n-blocks/--n-bootstrap> --n-permutations 5000
```

The spin test appends `{map_name}_sigmap_uncorr`, `_sigmap_fdr`, `_cluster_mask_fdr` and `_cluster_borders_fdr` to the combined CIFTI and, with `--output-dir`, writes `{stem}_spin{N}_significance.dscalar.nii` and a summary JSON.

The group scripts read `{output-dir}/subject_data/*/{model}_{modality}/{config}/{stem}_searchlight.npy` (or `--analysis-label` in place of `{model}_{modality}`); `group_stats.py` also accepts `--method rho_a` for crossnobis arrays and `--fname-prefix`. `tfce_groupstats.py` writes `group_stats_{S}subs.dscalar.nii` with maps `mean_rho, t_stat, sigmap_uncorr, sigmap_fdr, tfce_stat, sigmap_tfce_fwe`, masks, borders, `summary.json` and `group_stats_{S}subs_h0_null.npy` (null maxima, reusable to recompute `sigmap_tfce_fwe`).

### Partial and residual-embedding RSA

```bash
python rsa/partial_rsa.py --run partial_corr_pe-av-small-16-frame \
    --preprocessed-dir data/preprocessed/average_sub/raw --fmri-suffix raw --subject group_average \
    --timing-csv data/movie_timing.csv --embeddings-dir outputs/model_embeddings \
    --template-cifti <template> --output-dir outputs/rsa/raw --k 100 --method spearman \
    --left-surface <L> --right-surface <R> --workbench <wb_command> [--force]

bash rsa/run_peav_partial_analysis.sh [group_average|persubject|groupstats|tfce|all] [BATCH_SIZE] [START_FROM]
bash rsa/run_partial_corr_variants.sh [fits|fits_own_unimodal|consolidate|all]
bash rsa/run_extended_analyses.sh [embed|rsa|partial|consolidate|all]     # MODEL_NORM env, default center
```

`partial_rsa.py` defaults to `--bin-sec 5.0`, `--delay-sec 5.0`, `--tr 1.0`, `--n-blocks 1`. It writes `{OUTPUT_DIR}/group_average/{label}/{searchlight config}/{stem}_searchlight.npy` and `partial_corr_r_searchlight.dscalar.nii` (`integration_partial_r_searchlight.dscalar.nii` for `kind="integration"`), with `label` taken from the run key (`{model}_av_partial_corr`, `{model}_av_INTEGRATION`, ...).

`run_peav_partial_analysis.sh` uses run key `partial_corr_pe-av-small-16-frame`, `N_BLOCKS=16`, `BLOCKS_SWEEP=(4 8 16)` and 5000 permutations. `run_partial_corr_variants.sh` runs `partial_corr_variants.py --variant {from_unimodals,text_aligned_models,own_unimodal}`; each gives a five-map CIFTI of partial correlations `corr(brain, target | nuisance)`: the joint model given each unimodal reference alone and both together, and each unimodal reference given the other.

`run_extended_analyses.sh` runs the residual-embedding suite. `embed` writes linear- and projection-residual embeddings (`notebooks/feature_extraction/compute_{linear,projection}_residual_embeddings.py`) as ordinary embedding files `{model}_av_linear_resid_unimodal` and `{model}_av_projection_resid_own`; `rsa` runs the searchlight on them; `partial` runs `partial_corr_{model}` (given AudioMAE and VideoMAEv2-Large); `consolidate` runs `residualized_maps.py` per model in `RESIDUALIZED_AV_MODELS` and writes `rsa_59k_{tag}_{config}_residualized_maps.dscalar.nii` with scalars `partial_correlation`, `linear_resid_unimodal`, `projection_resid_own` and, for PE-AV only, `linear_resid_own`.

### Maximum-unimodal contrast

```bash
python rsa/max_uni.py --output-dir outputs/rsa/raw --run within_architecture_unimodal \
    --k 100 --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --method spearman --model-norm center \
    --fmri-tag raw --template-cifti <template> --out-dir <out>
# or: --target pe-av-small-16-frame av --baselines audiomae a videomaev2-large v
```

`--run` is one of `within_architecture_unimodal`, `cross_architecture_specialist`, `cross_family_specialist_strict` (`MAX_UNI_RUNS`). Per subject, `contrast = rho_AV - max(rho_A, rho_V)`, followed by a one-sample t-test and BH-FDR across vertices; `--n-blocks` (default 4) and `--n-bootstrap` (default 2000) control the corrected two-factor variant. Output: `{out_stem}_{S}subs.dscalar.nii` with maps `max_uni, rho_target, rho_max_uni, t_stat, sigmap_uncorr, sigmap_fdr` (plus `t_c2f_max_uni, sigmap_c2f_max_uni`), an FDR mask and a summary JSON.

### Crossnobis and noise ceiling

```bash
python rsa/crossnobis_searchlight.py --subjects-list data/subjects.txt --raw-dir data/individual-59k \
    [--sg-filter] [--psc] [--gsr] --timing-csv data/movie_timing.csv \
    --embeddings-dir outputs/model_embeddings --template-cifti <template> \
    --left-surface <L> --right-surface <R> --workbench <wb_command> --output-dir outputs/rsa/raw \
    --model pe-av-small-16-frame --modality av --k 100 --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --tr 1.0

python rsa/noise_ceiling.py --raw-dir data/individual-59k --subjects 100610 102311 ... \
    --timing-csv data/movie_timing.csv --k 100 --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --method spearman \
    --left-surface <L> --right-surface <R> --workbench <wb_command> --template-cifti <template> --output-dir <out>
```

Crossnobis loads all subjects into memory (`N x n_windows x 59412` float32), uses z-scored embeddings for the model RDM, and has no `--method` or `--model-norm` flag. Output: `{OUTPUT_DIR}/groupstats/{model}_{modality}/k{K}_delay{D}s_bin{B}s_skip{S}s_rho_a_isub/crossnobis_isub_rho_a_k{K}_delay{D}s_bin{B}s_skip{S}s.{npy,dscalar.nii}` and `..._sigmap.dscalar.nii`. `run_crossnobis.sh` and the per-subject crossnobis step of `analysis.sh` pass per-subject flags (`--subject`, `--preprocessed-dir`, `--fmri-suffix`) that this script does not define; use the command above.

`noise_ceiling.py` reads preprocessed CIFTIs with `--preprocessed-root` and `--fmri-suffix` instead of `--raw-dir`. Its `--delay-sec` defaults to 0.0, so pass the searchlight delay when the CIFTIs are not already delay-aligned.

### Scramble, dummy-modality and integration controls

A native joint model is compared with a version whose audio-video pairing is temporally permuted (`_avscramble`, `_avscramble_seed{K}`; the video of window `i` is paired with the audio of window `perm(i)`) or whose one modality is replaced by a content-free placeholder (`_clsav_from_a`, `_clsav_from_v`).

```bash
bash rsa/run_diff_study.sh [plain|partial|consolidate|all]        # default plain
DIFF_STUDY_MODELS="pe-av-small-16-frame" DIFF_STUDY_CONDITIONS="avscramble" bash rsa/run_diff_study.sh all
```

`plain` runs per-subject searchlight and group statistics, then the group-average searchlight, for every `{base_model}_{condition}`. `partial` runs the `partial_rsa.py` integration runs (`integration_{short}_{condition}`). `consolidate` runs `scramble_paired_stats.py` for every pair, then `scramble_diff_maps.py` and `dummy_diff_maps.py`.

- `scramble_paired_stats.py` (`--intact-model M --scrambled-model M_avscramble --modality av`, plus the group-statistics flags, optional `--n-perm N`) computes per subject and vertex `diff = rho_intact - rho_scrambled`, then a one-sample t-test, BH-FDR, optionally the corrected two-factor variant and a sign-flip permutation test. Output: `{output-dir}/groupstats/{intact}_vs_{scrambled}_{modality}/{config}/scramble_paired_stats_{S}subs[_nblocks{N}][_perm{N}].dscalar.nii` with maps `mean_rho_intact, mean_rho_scrambled, mean_diff, t_stat, sigmap_uncorr, sigmap_fdr` (plus `*_c2f`, `sigmap_perm`, `sigmap_perm_fdr`).
- `scramble_diff_maps.py` and `dummy_diff_maps.py` take no flags (model lists and paths are constants; `MODEL_NORM` selects the normalization). Per model, `binding = integration_intact - integration_scrambled` and `modality_presence_diff = integration_intact - max(integration_dummy_from_a, integration_dummy_from_v)`, where `integration` is the partial-correlation map of the joint embedding given its own audio and video embeddings. They write `scramble_consolidated_maps.dscalar.nii`, `dummy_consolidated_maps.dscalar.nii` and `dummy_partial_diff_maps.dscalar.nii`.
- `temporal_scramble_binding.py` (`--models`, `--rsa-root`, `--config`, `--template-cifti`, `--glasser-dlabel`, `--out-dir`) writes per-model `binding` maps, a cross-model convergence map (mean of per-model z-scored maps and the count of models with positive value) and ranked Glasser parcels. `integration_convergence.py` does the same for the integration maps; the sign count is descriptive.

Audio-video pairing permutation null:

```bash
bash rsa/run_av_scramble_permutations.sh --batch-size N        # first N missing seeds in 1..TOTAL
bash rsa/run_av_scramble_permutations.sh --seeds 401 450       # explicit seeds
bash rsa/run_av_scramble_permutations.sh --aggregate-only [--require-complete]
# also: --total-permutations (default 500), --alpha (default 0.01), --no-aggregate, --dry-run
```

For each seed the script extracts the scrambled embedding (`notebooks/feature_extraction/pe_av_extract_scramble.py --scramble-av --seed S`) and runs the group-average searchlight. `av_scramble_permutation_inference.py` aggregates the completed seeds: `delta_rho = intact_rho - mean(null_rho)`, the empirical one-tailed `p = (count(null_rho >= intact_rho) + 1) / (n + 1)` per vertex, BH-FDR, and single-step maximum-statistic FWE correction (the null is the maximum `rho` over cortex per seed). Output is `{OUTPUT_DIR}/group_average/_av_scramble_permutation/{model}_{modality}/{config}/av_scramble_{run_tag}_all_maps.dscalar.nii` (maps include `intact_rho, null_mean_rho, delta_rho, p_perm, p_perm_fdr, p_perm_maxT_fwe`), per-mask CIFTIs and a summary JSON, with `run_tag = n{completed}_of{total}_alpha{alpha}`.

### Topo-Omni sheet localizers

`topoomni_sheet_localizer.py` and `topoomni_av_separability_localizer.py` cluster the stimulus windows with Ward's linkage on an independent model's embedding and score each candidate cluster by a Welch t-test per cortical-sheet unit (in-cluster versus out-of-cluster windows), summarized as the median t over units. A top-down dendrogram traversal stops splitting when neither child scores above its parent; cluster size is limited to `--n-min`..`--n-max`. The separability localizer also tests one-versus-rest enrichment of the real audio-video condition in each terminal cluster with Fisher's exact test, over `--design dummy` (`av, clsav_from_a, clsav_from_v`) or `--design scramble` (`av, avscramble`). For each cluster with `p < --p-threshold` (default `1e-4`), the top `--top-pct-units` percent of units by `|t|` (or Benjamini-Hochberg selection at `--fdr-q` with `--unit-selection-mode fdr`) are written as new embeddings (`..._c{N}`, and `..._all` for their union) under `--embeddings-dir`, with a summary JSON.

```bash
python rsa/topoomni_sheet_localizer.py --sheet-model topoomni_layer18_sheet_mp \
    --cluster-embedding-model pe-av-small-16-frame --cluster-embedding-modality event_t \
    --auditory-regressor-model whisper_speech_proxy \
    --embeddings-dir outputs/model_embeddings --bin-sec 5.0 --skip-sec 5.0 --n-min 10 --n-max 375
python rsa/topoomni_av_separability_localizer.py --sheet-model topoomni_layer18_sheet_mp \
    --cluster-embedding-model pe-av-small-16-frame --design scramble \
    --embeddings-dir outputs/model_embeddings --n-min 10 --n-max 450
bash rsa/run_sheet_localizer_brain_maps.sh <speech> <driver> <sheet>             # env: SUFFIX
DESIGN=scramble bash rsa/run_av_separability_brain_maps.sh <driver> <sheet>      # env: DESIGN (dummy|scramble), SUFFIX
```

Embedding names follow `localizer_naming.py`: `localizer_{kind}[_{design}]_drv-{driver}_sheet-{sheet}[suffix][_c{N}|_all]`. The brain-map runners run `searchlight.py` on every per-cluster and combined embedding and merge the results (`label_sheet_localizer_maps.py`, `label_av_separability_maps.py`) into `{OUTPUT_DIR}/group_average/{base_name}/{base_name}_all_clusters_maps.dscalar.nii`. `spatial_stats.py` supplies Moran's I and a compactness index for selected units on the sheet grid.

### Full-sheet RSA

`full_sheet_rsa.py` computes RSA between two seed regions of the group-average fMRI (default `cca_a` and `cca_p`, masks from `--masks-dir`, `--seed-a-mask`, `--seed-p-mask`) and every unit of the Topo-Omni cortical sheet. The sheet has 304 rows x 512 columns = 155,648 units, flattened row-major: vision encoder rows 0-159 and columns 0-255, audio encoder rows 0-159 and columns 256-511, and the thinker (the language-model backbone consuming fused audio and video tokens) rows 160-303. The embedding `outputs/model_embeddings/topoomni_fullsheet/bin5s_skip5s/topoomni_fullsheet_av.npy` comes from `topo_omni_extract_full_sheet.py`. Unit neighborhoods use the trained coordinates, which the checkpoint does not ship; `shared/sheet_rsa.py::load_true_coords` regenerates them as `init_coords.permute_coordinates(seed=42)` (a seeded permutation of the raster lattice within each architectural block) and caches them at `--true-coords-cache`.

The seed RDM is the correlation distance across bins between the binned fMRI patterns of all vertices in the seed mask. For each unit, `rho` is the Spearman (or Pearson) correlation between the seed RDM and the RDM of that unit's `k = 100` nearest sheet units (`scipy.spatial.cKDTree` on true coordinates). The null uses `n_perm` within-run circular shifts, identical for both seeds so that the null of `rho_a - rho_p` is paired: `p = (1 + count(|null_diff| >= |diff|)) / (1 + n_perm)`, with BH-FDR over units. `sanity_corr` in the per-unit tables (correlation of each unit with the seed's mean time series) is a side check, not the RSA result.

```bash
bash rsa/run_full_sheet_rsa.sh        # env: K, OUTPUT_DIR, BIN_SEC, SKIP_SEC, DELAY_SEC, TR, METHOD, N_PERM, SEED, GPU_BATCH_SIZE, PERM_BATCH_SIZE, TRUE_COORDS_CACHE
python rsa/full_sheet_rsa.py --output-dir <out> --k 100 --gpu-batch-size 64 --perm-batch-size 20
```

Outputs in `--output-dir`: `{seed}_av.csv` and `{seed}_av_rho.npy` per seed (columns `unit_index, tower, true_row, true_col, raster_row, raster_col, rho, p_perm, p_fdr`), `diff.csv`, `diff_p_perm.npy`, `diff_p_fdr.npy`, sheet-map PNGs, and `metadata.json`. The metadata holds the parameters and the `characterize` summary (`shared/sheet_rsa.py`): per-tower mean `rho` for each seed and their difference, the cross-seed correlation of per-unit `rho`, hotspots (top `--pref-top-decile` of units by `rho`) per tower, their Jaccard overlap against a random-subset null, and a nearest-neighbor contiguity statistic. A topography control recomputes `rho` with each unit's true `k` sheet neighbors versus `k` units drawn from the same tower with coordinates ignored (`--topo-n-draws` draws); because adjacent units of a smooth sheet are redundant, this comparison does not by itself test whether the topography is meaningful.

### Other scripts

- `draw_rsa_borders.py --rsa-npy <searchlight.npy> --threshold-mode percentile --top-pct 5 --min-verts 10` keeps the top `--top-pct` percent of cortex jointly over both hemispheres (`--threshold-mode sd`, the default, keeps `mean + --n-sd * SD`, `--n-sd` default 2), removes connected islands smaller than `--min-verts`, and writes `{stem}_{top5pct|2sd}_{lh,rh}.border`.
- `kreilability.py --subjects-list ... --timing-csv ... --embeddings-dir ... --template-cifti <template> --model M --modality av --k 100 150 200 --n-splits 50` splits the subjects at random into halves and, for each split and `k`, reports the Pearson correlation of the two half-average searchlight maps and the Dice overlap of their top-10% masks (`{--outdir}/split_half_reliability_results.json`, resumable; `--bin-sec` defaults to 2.0).
- `rdm_diagonal.py --embeddings-dir ... --timing-csv ... --output-dir ... --model M --modality av` writes within-clip (block-diagonal) and across-clip (off-diagonal) masks of the model RDM; with the fMRI arguments it also writes `{stem}_rsa_diagonal_maps.dscalar.nii` with `glasser_{method}_rho_{offdiag,blockdiag}` and `searchlight_{method}_rho_{offdiag,blockdiag}`.
- `channel_subset_searchlight.py` writes the PE-AV `av` embedding columns of one channel cluster as a standalone embedding and runs `searchlight.py` on it; cluster and paths are constants at the top of the file. `channel_class_rsa.py` and `run_channel_class_rsa.sh` depend on a channel labelling that no longer exists, and the shell script exits with an error.

## Tests

```bash
pytest tests/test_rsa.py tests/test_model_norm.py tests/test_sheet_rsa.py tests/test_perm_searchlight.py \
       tests/test_max_uni.py tests/test_av_derived_maps.py tests/test_residualized_maps.py \
       tests/test_av_scramble_permutation_inference.py tests/test_paired_bootstrap_resampling.py \
       tests/test_channel_class_rsa.py tests/test_channel_subset_searchlight.py
python rsa/shared/residuals.py; python rsa/localizer_naming.py; python rsa/spatial_stats.py; python rsa/full_sheet_rsa.py --demo   # self-checks
```
