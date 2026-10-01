# RSA: representational similarity analysis

Compares cortical activity in the HCP 7T movie-watching fMRI data with the representational geometry of model embeddings (audio, video and joint audio-video models). Two spatial scales are supported: a vertex-wise searchlight over the cortical surface, and Glasser parcels. Group-level inference, partial-correlation and residualization analyses, audio-video integration controls, noise ceilings, and a Topo-Omni cortical-sheet analysis build on the same core.

Run everything from the repository root in the `movie` conda environment (`rsa/environment.yml`):

```bash
conda activate movie
cd movie_watching
```

## What is computed

All quantities below are computed per vertex (searchlight) or per Glasser parcel, on binned data.

**Binning.** For each clip in `data/movie_timing.csv` (columns `video_id, onset_sec, end_sec, duration_sec, run_id`; `onset_sec` is global time, cumulative across runs), windows of `bin_sec` seconds start at `onset_sec + delay_sec` and advance by `skip_sec` (default `skip_sec = bin_sec`, no overlap). A window is dropped if it would extend past the end of its run. The fMRI value of a window is the mean over its repetition times (TRs). The number of windows per clip is `floor((duration_sec - bin_sec) / skip_sec) + 1`. The model embedding for each window is read from a precomputed file with the same bin and stride (see Inputs), so brain and model window counts are asserted equal (`align_and_assert_bins`).

**Normalization.** Within each run, each vertex time series of window means is z-scored (subtract the mean, divide by the standard deviation over windows). Each embedding dimension is normalized within each run with `--model-norm`: `center` (subtract the mean; default; keeps each dimension's variance) or `zscore` (also divide by the standard deviation). There is no un-normalized option. With `--hrf`, each clip's embedding is additionally convolved with the SPM canonical hemodynamic response function and the delay label in output names is replaced by `hrf`.

**Brain RDM.** For a searchlight centered on vertex `v`, take the `k` geodesically nearest vertices on the midthickness surface (the center vertex itself is excluded; medial-wall vertices are dropped). With `n` windows, the brain representational dissimilarity matrix (RDM) is the `n x n` matrix of correlation distances `1 - r(x_i, x_j)`, where `x_i` is the vector of the neighborhood's fMRI values in window `i` and `r` is Pearson correlation across neighborhood vertices. The Glasser analysis uses all vertices of a parcel instead of a neighborhood (parcels with fewer than 2 vertices are skipped).

**Model RDM.** Correlation distance between the (normalized) embedding vectors of windows `i` and `j`: `1 - r(e_i, e_j)` across embedding dimensions.

**Searchlight statistic.** The lower-triangle entries (`n(n-1)/2` pairs) of the brain and model RDMs are compared with Spearman correlation (Pearson on ranks; `--method spearman`, default) or Pearson correlation (`--method pearson`). The result at vertex `v` is `rho(v)`. The Glasser analysis additionally supports `rho_a` (Kendall's tau computed with `scipy.stats.kendalltau`).

**Descriptive significance maps (single run).** `searchlight.py` also writes `sign(rho) * -log10(p)` maps with a one-tailed `p` from `t = rho * sqrt(df) / sqrt(1 - rho^2)`, `df = n - 2`, treating the `n` windows as independent samples, plus Benjamini-Hochberg false-discovery-rate (BH-FDR) corrected versions at q = 0.05. Use the group-level or permutation tests below for inference.

**Group statistics over subjects (`group_stats.py`).** With per-subject maps `rho_s(v)` for `S` subjects: mean `rho`, one-sample t-test of `rho_s(v)` against 0 (no Fisher z transform), one-tailed `p` for `rho > 0`, BH-FDR across vertices. If per-subject temporal-block maps exist, a corrected two-factor bootstrap (Schutt et al. 2023, Eq. 5) is added: the variance of the group mean under resampling of subjects (`var_subj`), of blocks (`var_block`), and of both (`var_both`) is combined as `var_c2f = clip(2(var_subj + var_block) - var_both, max(var_subj, var_block), var_both)`; `t_c2f = mean_rho / sqrt(var_c2f)` with `df = min(S - 1, n_blocks - 1)`. Blocks are non-overlapping temporal segments (aligned to run boundaries when `n_blocks` equals the number of runs).

**TFCE group statistics (`tfce_groupstats.py`).** Same input as `group_stats.py`, but the t-test is applied to Fisher z values `z = arctanh(rho)`, and a sign-flipping permutation test with threshold-free cluster enhancement (TFCE; `mne.stats.permutation_cluster_1samp_test`, `start=0, step=0.2`, surface adjacency, seed 42) gives family-wise-error (FWE) corrected significance: a vertex is significant if its TFCE statistic exceeds the `(1 - alpha)` quantile of the null distribution of the maximum TFCE statistic. The reported `mean_rho` is the arithmetic mean of `rho`. `group_stats.py` and `tfce_groupstats.py` use different tests (raw `rho` versus Fisher z), so p-values from the two are not interchangeable.

**Permutation test (`perm_searchlight.py`, group average).** The null is built from `n_perm` synchronized, nonzero circular shifts of the bins inside each run (the same shift is applied to every vertex). A shift reorders conditions of the model RDM, so the null `rho` at vertex `v` is `brain_norm[v] . model_norm[perm_p]` (rank-normalized RDM vectors, one matrix product per vertex batch). `p_perm[v] = (count(null_rho[v, :] >= rho[v]) + 1) / (n_perm + 1)`.

**Partial RSA (`partial_rsa.py`).** For a target model and a list of nuisance models, each RDM upper triangle is rank transformed (average ranks) when `--method spearman`. The brain vector `y` and target vector `t` are residualized by ordinary least squares on `[1, nuisance_1, ..., nuisance_m]` (no ridge penalty), and the result is the Pearson correlation of the two residual vectors: the classical partial Spearman correlation of brain and target given the nuisance RDMs. Run keys (`--run`) are defined in `shared/model_registry.py::PARTIAL_RSA_RUNS`; `kind="integration"` runs use the target's own audio and video embeddings as nuisance (the "best-additive integration contrast").

**Residual-embedding RSA.** Instead of residualizing RDMs, the joint embedding itself is residualized and run through the ordinary searchlight (`shared/residuals.py`): `linear_residual` is the cross-validated ridge residual of the joint embedding `J` after regressing on concatenated nuisance embeddings; `projection_residual` removes, per time window, the projection of the joint embedding onto the span of that window's own audio and video vectors (Gram-Schmidt).

**Audio-video contrasts.** `av_derived_maps.py` (group average) and `max_uni.py` (per subject, with group inference) compare the joint (`AV`) searchlight `rho` with audio-only (`A`) and video-only (`V`) references: conjunction `(AV > 0) & (AV > A) & (AV > V)`, superadditivity `AV - (A + V)`, and maximum-unimodal contrast `AV - max(A, V)`.

**Crossnobis (`crossnobis_searchlight.py`).** Inter-subject cross-validated Euclidean distance between windows `t1`, `t2` over `N` subjects with neighborhood patterns `x_i`: `d(t1, t2) = [ || sum_i x_i(t1) - sum_i x_i(t2) ||^2 - sum_i || x_i(t1) - x_i(t2) ||^2 ] / (N (N - 1))`. Its expectation is the true squared distance because measurement noise is independent across subjects. The distances over all window pairs are compared with the model RDM by Kendall's tau (`scipy.stats.kendalltau`).

**Noise ceiling (`noise_ceiling.py`).** With `RDM_i(v)` the raw correlation-distance vector of subject `i` and `N` subjects: `NC_upper(v) = mean_i rho_s[RDM_i, mean_j RDM_j]`, `NC_lower(v) = mean_i rho_s[RDM_i, mean_{j != i} RDM_j]`, with `rho_s` the Spearman correlation of lower-triangle vectors. `NC_upper` includes subject `i` in the group mean; `NC_lower` is leave-one-out.

## Scripts

Runners:

| Script | Purpose |
|---|---|
| `analysis.sh` | Main runner: neighbor caches, group-average RSA, per-subject RSA, group statistics, crossnobis, noise ceiling |
| `run_extended_analyses.sh` | Residual-embedding suite (embed, rsa, partial, consolidate) for group average |
| `run_peav_partial_analysis.sh` | PE-AV classical partial RSA: group average, per subject, group statistics, TFCE |
| `run_partial_corr_variants.sh` | Three families of five partial-correlation maps for PE-AV |
| `run_diff_study.sh` | Scramble and dummy-modality controls against native joint models |
| `run_av_scramble_permutations.sh` | Resumable batches of audio-video pairing permutations, aggregated into an empirical null |
| `run_full_sheet_rsa.sh` | `full_sheet_rsa.py` with default parameters |
| `run_sheet_localizer_brain_maps.sh`, `run_av_separability_brain_maps.sh` | Searchlight and label-merge for localizer outputs |
| `run_crossnobis.sh` | Wrapper for `crossnobis_searchlight.py` (see Crossnobis section) |

Analysis scripts:

| Script | Purpose |
|---|---|
| `searchlight.py` | Vertex-wise searchlight RSA |
| `glasser.py` | Glasser parcel RSA |
| `precompute_neighbors.py` | Geodesic k-nearest-neighbor cache for one subject |
| `group_stats.py`, `tfce_groupstats.py` | Group inference over per-subject maps |
| `perm_searchlight.py` | Within-run circular-shift permutation test (group average) |
| `run_spin_permutations.py` | Spin test of one map |
| `partial_rsa.py` | Partial RSA |
| `partial_corr_variants.py` | Consolidate the three five-map partial-correlation families into three CIFTIs |
| `residualized_maps.py` | Consolidate partial / residual-embedding RSA maps per model |
| `av_derived_maps.py`, `max_uni.py` | Audio-video contrasts |
| `crossnobis_searchlight.py`, `noise_ceiling.py` | Crossnobis and noise ceiling |
| `kreilability.py` | Split-half reliability of searchlight maps across neighborhood sizes |
| `rdm_diagonal.py` | Within-clip and across-clip RDM masking, optional RSA on the masked RDMs |
| `draw_rsa_borders.py` | Workbench borders around top-RSA islands |
| `scramble_paired_stats.py`, `scramble_diff_maps.py`, `dummy_diff_maps.py`, `temporal_scramble_binding.py`, `integration_convergence.py`, `av_scramble_permutation_inference.py` | Scramble / dummy / integration analyses |
| `topoomni_sheet_localizer.py`, `topoomni_av_separability_localizer.py`, `label_sheet_localizer_maps.py`, `label_av_separability_maps.py`, `localizer_naming.py`, `spatial_stats.py` | Topo-Omni sheet localizers |
| `full_sheet_rsa.py`, `shared/sheet_rsa.py` | Seed-region RSA against the full Topo-Omni cortical sheet |
| `channel_subset_searchlight.py` | Searchlight on a channel subset of the PE-AV embedding |
| `channel_class_rsa.py`, `run_channel_class_rsa.sh` | Retired (depend on the removed four-class channel labelling); `run_channel_class_rsa.sh` exits immediately |

Searchlight centered kernel alignment (non-cross-validated, cross-validated and whitened) is not RSA and lives in [`cka/`](../cka/README.md).

## Inputs

Paths are set at the top of `analysis.sh` (`DATA_BASE`, `OUTPUTS_BASE`) and in the other runners.

| Input | Location |
|---|---|
| Raw 7T CIFTIs (streaming mode) | `data/individual-59k/` |
| Preprocessed CIFTIs (disk mode, group average) | `data/preprocessed/average_sub/{tag}/group_average_{tag}_cortex_59k.dtseries.nii` and `group_average_{tag}_run_trs.npy` |
| Preprocessed CIFTIs (disk mode, subjects) | `data/preprocessed/{tag}/{subject}_{tag}_cortex_59k.dtseries.nii` and `{subject}_{tag}_run_trs.npy` (written by `preprocess_individual.py`, run by `analysis.sh preprocess`) |
| Timing | `data/movie_timing.csv` |
| Subject list | `data/subjects.txt`: one ID per line; `#` comments and blank lines are ignored |
| Embeddings | `outputs/model_embeddings/{model}/bin{B}s_skip{S}s/{model}_{modality}.npy`, shape `(n_windows, n_features)`; modality codes are `a`, `v`, `av` and text variants |
| Template CIFTI | `group_average_raw_cortex_59k.dtseries.nii`, used only for its brain-model axis (59,412 cortical grayordinates) |
| Surfaces | `data/HCP_S1200_GroupAvg_v1/GroupAverage_59k/CohortAvg.{L,R}.midthickness_MSMAll.59k_fs_LR.surf.gii`; per-subject midthickness surfaces from `MIDTHICKNESS_DIR` are used when present, otherwise the group average |
| Glasser parcellation | `data/HCP_S1200_GroupAvg_v1/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii` |
| Workbench | `/opt/workbench/bin_linux64/wb_command` |

The `{tag}` is `raw` when no signal preprocessing is applied, otherwise the applied steps joined by `_` from `sg` (Savitzky-Golay high-pass filter), `psc` (percent signal change), `gsr` (global signal regression).

Model names in `shared/model_registry.py` carry an explicit pooling suffix for the omni model families: `_mp` (mean pool) or `_lt` (last token), for example `nemotron_layer18_mp`, `omni3b_layer18_lt`, `topoomni_layer18_sheet_mp`.

### Geodesic neighbor cache

`searchlight.py::get_neighbors` returns the `(n_surface_vertices, k)` nearest-neighbor array per subject and hemisphere. Lookup order: exact `{subject}_{hem}_neighbors_k{K}.npy` in `--geodesic-cache-dir`; otherwise slice a cached file with larger `K` (smallest such `K`) and save the exact-`K` file; otherwise compute all-to-all geodesic distances with `wb_command -surface-geodesic-distance-all-to-all` (a multi-gigabyte `dconn`), extract the `k` nearest, and save. The per-subject `dconn` is deleted after extraction.

## Usage

```bash
bash rsa/analysis.sh [MODE] [METHOD] [BATCH_SIZE] [START_FROM] [N_BLOCKS]
```

| Argument | Values | Default |
|---|---|---|
| `MODE` | `avg`, `preprocess`, `neighbors_avg`, `neighbors`, `persubject`, `groupstats`, `all` | `avg` |
| `METHOD` | `all` (searchlight and Glasser), `searchlight`, `glasser` (ignored by the neighbor modes) | `all` |
| `BATCH_SIZE` | Subjects in parallel (GNU parallel); joblib threads per subject are `nproc / BATCH_SIZE` | 4 (8 for `neighbors`) |
| `START_FROM` | Subject ID at which to start the per-subject loop | none |
| `N_BLOCKS` | Overrides `N_BLOCKS` | 16 |

Modes:

- `avg`: group-average searchlight and/or Glasser RSA for every model in `MODELS`. For `av` targets with defined baselines, missing group-average audio and video reference maps are run first and the nine derived maps are appended (see Outputs).
- `preprocess`: `preprocess_individual.py` for every subject plus the group average (disk mode input).
- `neighbors_avg`, `neighbors`: build the neighbor cache for the group-average surface, or additionally for every subject.
- `persubject`: per-subject searchlight and Glasser RSA (streaming or disk mode), then group statistics over the `BLOCKS_SWEEP`, then the noise ceiling.
- `groupstats`: inter-subject crossnobis, group statistics, noise ceiling, on existing per-subject maps.
- `all`: `avg`, `persubject`, crossnobis, group statistics, noise ceiling.

```bash
bash rsa/analysis.sh avg                          # group average, searchlight and Glasser
bash rsa/analysis.sh avg searchlight              # group average, searchlight only
bash rsa/analysis.sh neighbors_avg                # build group-average neighbor cache once
bash rsa/analysis.sh persubject searchlight 4     # per subject, 4 in parallel
bash rsa/analysis.sh persubject all 8 100610      # resume from subject 100610
bash rsa/analysis.sh groupstats
BIN_SECS="2.0 5.0 10.0" bash rsa/analysis.sh avg  # sweep bin sizes
MODEL_NORMS="center zscore" bash rsa/analysis.sh avg
RSA_MODELS_OVERRIDE="nemotron_layer18_mp:av;topoomni_layer18_sheet_mp:av" bash rsa/analysis.sh avg searchlight
```

Re-running skips any subject/model whose outputs exist; delete the output to force a rerun. `parallel --citation` silences the GNU parallel notice once.

### Configuration

Variables at the top of `analysis.sh`; those marked env can be set on the command line.

| Variable | Default | Meaning |
|---|---|---|
| `STREAM` | `true` | `true`: `persubject` preprocesses raw CIFTIs on the fly; `false`: it reads pre-saved per-subject CIFTIs (group-average runs always read the preprocessed group-average CIFTI) |
| `SG_FILTER`, `PSC`, `GSR` | `false`, `false`, `false` | Signal preprocessing steps (see `{tag}`) |
| `BIN_SECS` (env; `BIN_SEC` for one value) | `2.0` | Bin sizes (seconds) to sweep |
| `SKIP_SEC` (env) | `BIN_SEC` | Window stride (seconds) |
| `DELAY_SEC` | `5.0` | Hemodynamic delay added to each clip onset |
| `HRF` | `false` | Convolve embeddings with the SPM HRF (passes `--hrf`) |
| `MODEL_NORMS` (env) | `center` | Space-separated subset of `center`, `zscore` |
| `METHOD` | `spearman` | Searchlight comparator |
| `GLASSER_METHOD` | `spearman` | Parcel comparator (`rho_a` is supported by `glasser.py`) |
| `K` | `100` | Searchlight neighborhood size |
| `TR` | `1.0` | Repetition time (seconds) |
| `N_BLOCKS` | `16` | Temporal blocks saved by per-subject `searchlight.py` (group-average runs use 1) |
| `BLOCKS_SWEEP` | `(4 8 16)` | `--n-blocks` values for which `group_stats.py` is run (each must divide `N_BLOCKS`) |
| `N_BOOTSTRAP` | `2000` | Bootstrap iterations in `group_stats.py` |
| `RUN_PERM` | `false` | Also run `perm_searchlight.py` after each group-average searchlight |
| `CROSSNOBIS_BIN_SEC`, `CROSSNOBIS_SKIP_SEC` | `5.0`, `5.0` | Bin and stride for crossnobis |
| `GPU_BATCH_SIZE` (env) | `512` | Vertices per GPU batch (CPU fallback on out-of-memory) |
| `MODELS` | `pe-av-small-16-frame:av`, `cav-mae-sync:av`, plus the layer sweep | Array of `model:modality[,modality...]`; add a line to add a model |
| `LAYERS` | `(35 34 27 18 9 1)` | Layer sweep for `omni3b_layer*` and `topoomni_layer*` (layers 9, 18, 27, 34 add `_mp` and `_lt` entries, layers 1 and 35 use their bare names), plus `nemotron_layer{9,18,27,35,36}_{mp,lt}` |
| `SKIP_LAYER_SWEEP` (env) | `false` | `true` keeps only the two first `MODELS` entries |
| `RSA_MODELS_OVERRIDE` (env) | none | `;`-joined `model:modality,...` string replacing `MODELS` |
| `SKIP_NOISE_CEILING`, `SKIP_CROSSNOBIS` (env) | `false` | Skip those stages in `persubject` / `groupstats` / `all` |

A model/modality whose embedding file is missing is skipped with a log message.

## Outputs

`OUTPUT_DIR = outputs/rsa/{tag}`. The configuration labels are built by `shared/naming.py`:

```
searchlight config   k{K}_delay{D}s_bin{B}s_skip{S}s_{method}_{model_norm}          (delay{D}s becomes hrf with --hrf)
glasser config       delay{D}s_bin{B}s_skip{S}s_{method}_{model_norm}
file stem            rsa_59k_{tag}_{searchlight config}
```

```
{OUTPUT_DIR}/group_average/{model}_{modality}/
    rsa_59k_{tag}_{config}_maps.dscalar.nii        combined maps, accumulated across scripts (see below)
    {searchlight config}/
        {stem}_searchlight.npy                     (59412,) rho
        {stem}_fdr_mask.dscalar.nii                binary BH-FDR mask
        {stem}_fdr_{lh,rh}.border                  Workbench borders (skipped if the mask is empty or covers >90% of cortex)
        {stem}_perm{N}_p.npy                       permutation p-values (RUN_PERM=true)
    {glasser config}/ranked_report.csv             columns parcel, r, p, n_vertices, rank

{OUTPUT_DIR}/subject_data/{subject}/{model}_{modality}/            same layout per subject
{OUTPUT_DIR}/subject_data/{subject}/pipeline.log
{OUTPUT_DIR}/subject_data/subject_reports/{subject}.json           per model: n_bins, max_r, mean_r, timestamp
{OUTPUT_DIR}/groupstats/{model}_{modality}/{searchlight config}/
    group_stats_{S}subs[_nblocks{N}].dscalar.nii   maps: mean_rho, t_stat, sigmap_uncorr, sigmap_fdr
                                                   (+ mean_rho_c2f, t_c2f, sigmap_c2f when block maps exist)
    group_stats_{S}subs[_nblocks{N}]_fdr_mask.dscalar.nii
    group_stats_{S}subs[_nblocks{N}]_fdr_c2f_mask.dscalar.nii   (if block maps exist)
    group_stats_{S}subs_fdr_{lh,rh}.border
    summary.json
{OUTPUT_DIR}/noise_ceiling/noise_ceiling_{N}subs_k{K}_delay{D}s_bin{B}s_skip{S}s_{method}.dscalar.nii   maps nc_lower, nc_upper
```

Per-subject block maps (`n_blocks > 1`) are saved next to the searchlight array as `{stem}_searchlight_nblocks{N}.npy`, shape `(N, 59412)`.

Maps in `rsa_59k_{tag}_{config}_maps.dscalar.nii` (names are the scalar-axis names):

- `searchlight_{method}_rho`, `searchlight_{method}_rho_sigmap_uncorr`, `searchlight_{method}_rho_sigmap_fdr` (from `searchlight.py`)
- `glasser_{glasser method}_rho` (from `glasser.py`)
- `searchlight_{method}_sigmap_perm` (from `perm_searchlight.py`)
- `av_conjunction_{family}`, `av_superadditivity_{family}`, `av_max_uni_{family}` for `family` in `own`, `unimodal`, `text_aligned` (from `av_derived_maps.py`)

`own` uses the model's own audio and video reference maps (`cls-a`/`cls-v` for PE-AV and CAV-MAE; the pre-fusion `*_encoder_penultimate` audio and video maps for the omni families), `unimodal` uses AudioMAE and VideoMAEv2-Large, `text_aligned` uses WavLM-Large and PE-Core ViT-L/14. References are given by `shared/model_registry.py::av_derived_baselines`; scramble, dummy-modality and residual pseudo-models have none.

## Commands by analysis

### Searchlight and Glasser (single run)

```bash
python rsa/searchlight.py \
    --preprocessed-dir data/preprocessed/average_sub/raw --fmri-suffix raw \
    --subject group_average --timing-csv data/movie_timing.csv \
    --embeddings-dir outputs/model_embeddings --template-cifti <template> \
    --left-surface <L midthickness> --right-surface <R midthickness> \
    --workbench /opt/workbench/bin_linux64/wb_command --output-dir outputs/rsa/raw \
    --model pe-av-small-16-frame --modality av \
    --k 100 --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --method spearman --tr 1.0 \
    --model-norm center --geodesic-cache-dir outputs/rsa/_geodesic_cache \
    --combined-output <maps.dscalar.nii> --n-blocks 1
```

Flags: input is either `--preprocessed-dir` and `--fmri-suffix` (disk) or `--raw-dir` with `--sg-filter`, `--psc`, `--gsr`/`--no-gsr` (streaming; the two inputs are mutually exclusive). `--delay-sec` default 5.0; `--skip-sec` default `bin-sec`; `--n-blocks` default 4 (1 disables blocks); `--gpu-batch-size` default 512; `--model-norm {zscore,center}` default `center`; `--method {spearman,pearson}` (required); `--modality` one of `v a av at vt avt t caption_t transcript_t event_t transcript_avt event_avt`. The environment variable `_RSA_N_JOBS` sets the joblib worker count (set by `analysis.sh`). `glasser.py` takes the same input, timing, delay, binning and `--model-norm` flags plus `--glasser-dlabel`, and `--method {spearman,pearson,rho_a}`.

**Diagnostic maps.** Two options of `searchlight.py` give variants of the group-average map for comparison with CKA and encoding (`reports/measure_comparison/`). Either one (disk mode, `--preprocessed-dir`, only) makes the run write a single map and nothing else (no `.npy`, significance maps, borders, blocks or combined-map entries):

| Option | Effect |
|---|---|
| `--distance euclidean` | Squared Euclidean distance between windows instead of correlation distance, in the brain RDM (window vectors over the `k` neighbors, mean over windows removed) and in the model RDM (normalized embeddings); `--distance correlation` is the default. Name label `euclid-{method}` (`corr-{method}` for the default distance) |
| `--drop-repeated-clips` | Drops the windows of `video5`, `video9`, `video14`, `video18` (the clip shown once per run) after the per-run normalization of responses and embeddings, so the RDMs are those of the remaining windows. Adds `_norepeats` to the name |

Output: `{OUTPUT_DIR}/group_average/{model}_{modality}/diagnostics/rsa_59k_{tag}_k{K}_delay{D}s_bin{B}s_skip{S}s_{label}_{model_norm}[_norepeats]_maps.dscalar.nii`, one map named `{label}`. Example, the two diagnostic maps of one model (arguments as above, `--n-blocks 1`; they are not part of `analysis.sh`):

```bash
python rsa/searchlight.py <arguments as above> --distance euclidean            # ..._euclid-spearman_center_maps
python rsa/searchlight.py <arguments as above> --drop-repeated-clips           # ..._corr-spearman_center_norepeats_maps
```

### Permutation test, spin test

```bash
python rsa/perm_searchlight.py <searchlight flags> --n-perm 1000 --seed 42 \
    --perm-batch-size 100 --gpu-batch-size 256 --combined-output <maps.dscalar.nii>

python rsa/run_spin_permutations.py --combined-cifti <maps.dscalar.nii> \
    --map-name searchlight_spearman_rho \
    --left-sphere <L.sphere.59k_fs_LR.surf.gii> --right-sphere <R.sphere.59k_fs_LR.surf.gii> \
    --left-surface <L midthickness> --right-surface <R midthickness> \
    --template-cifti <template> --n-spin 1000 --alpha 0.05 --min-cluster-size 10 \
    --seed 42 --output-dir outputs/rsa/spin_tests
```

The spin test rotates the sphere coordinates of one map by `n_spin` random rotations and sets `null[s, v] = empirical[spin_index[s, v]]`; `p[v] = (count(null[:, v] >= empirical[v]) + 1) / (n_spin + 1)`, then BH-FDR. It appends `{map_name}_sigmap_uncorr`, `_sigmap_fdr`, `_cluster_mask_fdr`, `_cluster_borders_fdr` to the combined CIFTI and, with `--output-dir`, writes `{stem}_spin{N}_significance.dscalar.nii` and `{stem}_spin{N}_summary.json`. The null is the same map rotated, so it is a local hot-spot test of one map: it is neither a test of correspondence between two maps nor a test against zero.

### Group statistics

```bash
python rsa/group_stats.py --output-dir outputs/rsa/raw --model pe-av-small-16-frame --modality av \
    --k 100 --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --method spearman --model-norm center \
    --fmri-tag raw --template-cifti <template> --left-surface <L> --right-surface <R> \
    --workbench /opt/workbench/bin_linux64/wb_command \
    --n-blocks 16 --n-bootstrap 2000 --alpha 0.05 --min-cluster-size 10

python rsa/tfce_groupstats.py <same flags except --n-blocks/--n-bootstrap> --n-permutations 5000 --n-jobs -1
```

Both read `{output-dir}/subject_data/*/{model}_{modality}/{config}/{stem}_searchlight.npy` (or `--analysis-label` in place of `{model}_{modality}`; `group_stats.py` also accepts `--method rho_a` for crossnobis arrays and `--fname-prefix`, default `rsa_59k`). `tfce_groupstats.py` writes `group_stats_{S}subs.dscalar.nii` with maps `mean_rho, t_stat, sigmap_uncorr, sigmap_fdr, tfce_stat, sigmap_tfce_fwe`, standalone `_fdr_mask` and `_tfce_fwe_mask` CIFTIs, `group_stats_{S}subs_h0_null.npy` (null maxima; reused to patch `sigmap_tfce_fwe` without rerunning), borders, and `summary.json`.

### Partial RSA

```bash
python rsa/partial_rsa.py --run partial_corr_pe-av-small-16-frame \
    --preprocessed-dir data/preprocessed/average_sub/raw --fmri-suffix raw --subject group_average \
    --timing-csv data/movie_timing.csv --embeddings-dir outputs/model_embeddings \
    --template-cifti <template> --output-dir outputs/rsa/raw \
    --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --tr 1.0 --k 100 --method spearman --model-norm center \
    --left-surface <L> --right-surface <R> --workbench <wb_command> \
    --geodesic-cache-dir outputs/rsa/_geodesic_cache --n-blocks 1 [--force]
```

Defaults: `--bin-sec 5.0`, `--delay-sec 5.0`, `--tr 1.0`, `--gpu-batch-size 256`, `--n-blocks 1`; streaming uses `--raw-dir` with a single subject. Output: `{OUTPUT_DIR}/group_average/{label}/{searchlight config}/{stem}_searchlight.npy` and `partial_corr_r_searchlight.dscalar.nii` (`integration_partial_r_searchlight.dscalar.nii` for `kind="integration"`), where `label` comes from the run key (`{model}_av_partial_corr`, `{model}_av_INTEGRATION`, ...).

Runners:

```bash
bash rsa/run_peav_partial_analysis.sh [group_average|persubject|groupstats|tfce|all] [BATCH_SIZE] [START_FROM]
bash rsa/run_partial_corr_variants.sh [fits|fits_own_unimodal|consolidate|all]
```

`run_peav_partial_analysis.sh` uses run key `partial_corr_pe-av-small-16-frame` and `N_BLOCKS=16`, `BLOCKS_SWEEP=(4 8 16)`, `N_PERMUTATIONS=5000`. `run_partial_corr_variants.sh` runs the group-average fits and `partial_corr_variants.py --variant {from_unimodals,text_aligned_models,own_unimodal}`, each producing a five-map CIFTI of partial correlations (`corr(brain, target | nuisance)`): the joint model given each unimodal reference alone and both together, and each unimodal reference given the other.

### Residual-embedding suite

```bash
bash rsa/run_extended_analyses.sh [embed|rsa|partial|consolidate|all]     # default all; MODEL_NORM env, default center
```

`embed` writes linear- and projection-residual embeddings (`notebooks/feature_extraction/compute_linear_residual_embeddings.py`, `compute_projection_residual_embeddings.py`) as ordinary `{model}_av_linear_resid_unimodal`, `{model}_av_projection_resid_own` embedding files. `rsa` runs the searchlight on them through `analysis.sh avg searchlight`; `partial` runs `partial_corr_{model}` (partial RSA given AudioMAE and VideoMAEv2-Large); `consolidate` calls `residualized_maps.py` per model in `RESIDUALIZED_AV_MODELS`, writing `rsa_59k_{tag}_{config}_residualized_maps.dscalar.nii` in the model's `_av` directory with scalars `partial_correlation`, `linear_resid_unimodal`, `projection_resid_own`, `linear_resid_own` (the last for PE-AV only). Consolidation appends each scalar independently, so interrupted runs resume.

### Maximum-unimodal contrast with group inference

```bash
python rsa/max_uni.py --output-dir outputs/rsa/raw --run within_architecture_unimodal \
    --k 100 --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --method spearman --model-norm center \
    --fmri-tag raw --template-cifti <template> --out-dir <out>
# or: --target pe-av-small-16-frame av --baselines audiomae a videomaev2-large v   (flat MODEL MODALITY pairs)
```

`--run` is one of `within_architecture_unimodal`, `cross_architecture_specialist`, `cross_family_specialist_strict` (`MAX_UNI_RUNS`). Per subject, `contrast = rho_AV - max(rho_A, rho_V)`, then a one-sample t-test and BH-FDR across subjects; `--n-blocks` (default 4) and `--n-bootstrap` (default 2000) control the corrected two-factor variant. Output `{out_stem}_{S}subs.dscalar.nii` with maps `max_uni, rho_target, rho_max_uni, t_stat, sigmap_uncorr, sigmap_fdr` (+ `t_c2f_max_uni, sigmap_c2f_max_uni`), `_fdr_mask`, and `_summary.json`.

### Crossnobis and noise ceiling

```bash
python rsa/crossnobis_searchlight.py --subjects-list data/subjects.txt --raw-dir data/individual-59k \
    [--sg-filter] [--psc] [--gsr] --timing-csv data/movie_timing.csv \
    --embeddings-dir outputs/model_embeddings --template-cifti <template> \
    --left-surface <L> --right-surface <R> --workbench <wb_command> --output-dir outputs/rsa/raw \
    --model pe-av-small-16-frame --modality av --k 100 --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --tr 1.0 \
    --geodesic-cache-dir outputs/rsa/_geodesic_cache
```

Loads all subjects into memory (`N x n_windows x 59412` float32), uses the model RDM from z-scored embeddings, and writes `{OUTPUT_DIR}/groupstats/{model}_{modality}/k{K}_delay{D}s_bin{B}s_skip{S}s_rho_a_isub/crossnobis_isub_rho_a_k{K}_delay{D}s_bin{B}s_skip{S}s.{npy,dscalar.nii}` and `..._sigmap.dscalar.nii`. There is no `--method` or `--model-norm` flag. `run_crossnobis.sh` passes per-subject flags (`--subject`, `--preprocessed-dir`, `--fmri-suffix`) that this script does not define.

```bash
python rsa/noise_ceiling.py --raw-dir data/individual-59k [--sg-filter --psc --gsr] \
    --subjects 100610 102311 ... --timing-csv data/movie_timing.csv --k 100 \
    --bin-sec 5.0 --skip-sec 5.0 --delay-sec 5.0 --method spearman \
    --left-surface <L> --right-surface <R> --workbench <wb_command> \
    --geodesic-cache-dir outputs/rsa/_geodesic_cache --template-cifti <template> --output-dir <out>
```

Disk input uses `--preprocessed-root` with `--fmri-suffix`. Defaults: `--delay-sec 0.0` (pass the same delay as the searchlight when CIFTIs are not already delay-aligned), `--batch-size 256`; `--binned-cache-dir` caches per-subject binned arrays. Output: `noise_ceiling_{N}subs_k{K}_delay{D}s_bin{B}s_skip{S}s_{method}.dscalar.nii`.

### Scramble, dummy-modality and integration controls

A native joint model is compared with a version whose audio-video pairing is temporally permuted (`_avscramble`, `_avscramble_seed{K}`; video of window `i` is paired with audio of window `perm(i)`) or whose one modality is replaced by a content-free placeholder (`_clsav_from_a`, `_clsav_from_v`).

```bash
bash rsa/run_diff_study.sh [plain|partial|consolidate|all]        # default plain
DIFF_STUDY_MODELS="pe-av-small-16-frame" DIFF_STUDY_CONDITIONS="avscramble" bash rsa/run_diff_study.sh all
```

`plain`: per-subject searchlight and group statistics, then group-average searchlight, for every `{base_model}_{condition}` (via `analysis.sh` with `RSA_MODELS_OVERRIDE`). `partial`: `partial_rsa.py` integration runs (`integration_{short}_{condition}`). `consolidate`: `scramble_paired_stats.py` for every pair, then `scramble_diff_maps.py` and `dummy_diff_maps.py`.

- `scramble_paired_stats.py --output-dir <dir> --intact-model M --scrambled-model M_avscramble --modality av --k 100 --bin-sec 5.0 --delay-sec 5.0 --method spearman --fmri-tag raw --template-cifti <template> [--n-perm N]`: per subject and vertex, `diff = rho_intact - rho_scrambled`, a one-sample t-test on `diff`, BH-FDR, optional corrected two-factor variant (block files for both models), optional sign-flip permutation test (`--n-perm`). Output in `{output-dir}/groupstats/{intact}_vs_{scrambled}_{modality}/{config}/scramble_paired_stats_{S}subs[_nblocks{N}][_perm{N}].dscalar.nii` with maps `mean_rho_intact, mean_rho_scrambled, mean_diff, t_stat, sigmap_uncorr, sigmap_fdr` (+ `*_c2f`, `sigmap_perm`, `sigmap_perm_fdr`), plus masks and `_summary.json`.
- `scramble_diff_maps.py`, `dummy_diff_maps.py` (no flags; model lists and paths are constants; `MODEL_NORM` env selects the normalization): per model, `binding = integration_intact - integration_scrambled` (scramble) and `modality_presence_diff = integration_intact - max(integration_dummy_from_a, integration_dummy_from_v)` (dummy), where `integration` is the partial-correlation map of the joint embedding given its own audio and video embeddings. Output CIFTIs `scramble_consolidated_maps.dscalar.nii` and `dummy_consolidated_maps.dscalar.nii`, `dummy_partial_diff_maps.dscalar.nii`.
- `temporal_scramble_binding.py --models ... --rsa-root <group_average dir> --config <config> --template-cifti <template> --glasser-dlabel <dlabel> --out-dir <out>`: per-model `binding` maps (`binding_maps_per_model.dscalar.nii`), cross-model convergence (`binding_convergence_map.dscalar.nii`: mean of per-model z-scored maps and count of models with positive value), `binding_ranked_parcels.csv`, `binding_summary.json`.
- `integration_convergence.py`: same convergence summary over the integration maps (`convergence_map.dscalar.nii` with `convergence_mean_z`, `convergence_sign_count`; `convergence_ranked_parcels.csv`; `convergence_summary.json`). The sign count is descriptive, not a significance count.
- AV-pairing permutation null:

```bash
bash rsa/run_av_scramble_permutations.sh --batch-size N        # first N missing seeds in 1..TOTAL
bash rsa/run_av_scramble_permutations.sh --seeds 401 450       # explicit seeds
bash rsa/run_av_scramble_permutations.sh --aggregate-only [--require-complete]
# other options: --total-permutations (default 500), --alpha (default 0.01), --no-aggregate, --dry-run
```

For each seed the script extracts the scrambled embedding (`notebooks/feature_extraction/pe_av_extract_scramble.py --scramble-av --seed S`, in the `avtransformer` environment), runs the group-average searchlight, then `av_scramble_permutation_inference.py` aggregates all completed seeds: `delta_rho = intact_rho - mean(null_rho)`, empirical one-tailed `p = (count(null_rho >= intact_rho) + 1) / (n + 1)` per vertex, BH-FDR, and single-step maximum-statistic FWE correction (null of the maximum `rho` over cortex per seed). Output under `{OUTPUT_DIR}/group_average/_av_scramble_permutation/{model}_{modality}/{config}/`: `av_scramble_{run_tag}_all_maps.dscalar.nii` (16 maps, including `intact_rho, null_mean_rho, null_std_rho, delta_rho, p_perm, p_perm_fdr, p_perm_maxT_fwe` and the corresponding sigmaps and masks), `_significance_maps`, per-mask CIFTIs, `maxT_fwe_{sigmap,mask}_{0p01,0p005}.dscalar.nii`, and `av_scramble_{run_tag}_summary.json`, with `run_tag = n{completed}_of{total}_alpha{alpha}`.

### Topo-Omni sheet localizers

`topoomni_sheet_localizer.py` and `topoomni_av_separability_localizer.py` cluster the stimulus windows with Ward's linkage on an independent model's embedding and score each candidate cluster by a Welch t-test per cortical-sheet unit (in-cluster versus out-of-cluster windows), summarized as the median t over units; a top-down dendrogram traversal stops splitting when neither child scores above its parent (cluster size limited to `--n-min`..`--n-max`). The separability localizer then tests one-versus-rest enrichment of the real audio-video condition in each terminal cluster with Fisher's exact test, over `--design dummy` (conditions `av, clsav_from_a, clsav_from_v`) or `--design scramble` (`av, avscramble`). The top `|t|` units (top `--top-pct-units` percent, or Benjamini-Hochberg selection at `--fdr-q` with `--unit-selection-mode fdr`) of each cluster with `p < --p-threshold` (default `1e-4`) are written as new embeddings (`..._c{N}`, and `..._all` for their union) under `--embeddings-dir`, together with `_localizer_summary_*.json` / `_localizer_av_separability_summary_*.json` (names from `localizer_naming.py`).

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

Name rule (`localizer_naming.py`): `localizer_{kind}[_{design}]_drv-{driver}_sheet-{sheet}[suffix][_c{N}|_all]`. The brain-map runners run `searchlight.py` on every per-cluster and combined embedding and merge the results with `label_sheet_localizer_maps.py` / `label_av_separability_maps.py` into `{OUTPUT_DIR}/group_average/{base_name}/{base_name}_all_clusters_maps.dscalar.nii` (raw runs under `.../raw/`, cluster summary at `.../summary.json`). `spatial_stats.py` supplies Moran's I and a compactness index for selected units on the sheet grid.

### Full-sheet RSA (Topo-Omni cortical sheet)

`full_sheet_rsa.py` computes RSA between two seed regions of the group-average fMRI and every unit of the complete Topo-Omni cortical sheet, using true (trained) coordinates for the unit neighborhoods. Seed names default to `cca_a` (anterior) and `cca_p` (posterior); masks are `--masks-dir/--seed-a-mask` and `--seed-p-mask`.

Sheet: 304 rows x 512 columns = 155,648 units, row-major flattened. Towers, from each unit's raster position (unaffected by the coordinate permutation): vision encoder rows 0-159, columns 0-255 (32 layers x 5 rows); audio encoder rows 0-159, columns 256-511 (32 layers x 5 rows); thinker (language-model backbone consuming fused audio and video tokens) rows 160-303, all columns (36 layers x 4 rows, layer `L` at rows `160+4L .. 163+4L`). `topo_omni_extract_full_sheet.py` extracts the embedding `outputs/model_embeddings/topoomni_fullsheet/bin5s_skip5s/topoomni_fullsheet_av.npy` (intact joint audio-video pass only). Coordinates: no trained `coords.npy` ships with the checkpoint; `shared/sheet_rsa.py::load_true_coords` regenerates `init_coords.permute_coordinates(seed=42)` (a seeded permutation of the raster lattice within each architectural block) and caches it at `--true-coords-cache`.

Seed RDM: correlation distance across all bins between binned fMRI patterns of all vertices in the seed mask (not the mean time series). Per unit, `rho` is the Spearman (or Pearson) correlation between the seed RDM and the RDM of that unit's `k` nearest sheet units (`k = 100` by default, `scipy.spatial.cKDTree` on true coordinates). Null: `n_perm` within-run circular shifts, the same shifts for both seeds, so the null of `rho_a - rho_p` is paired; `p = (1 + count(|null_diff| >= |diff|)) / (1 + n_perm)`; BH-FDR over units. `sanity_corr` in the per-unit tables is a separate check (Pearson correlation of each unit with the seed's mean time series), not the RSA result.

```bash
bash rsa/run_full_sheet_rsa.sh        # env: K, OUTPUT_DIR, BIN_SEC, SKIP_SEC, DELAY_SEC, TR, METHOD, N_PERM, SEED, GPU_BATCH_SIZE, PERM_BATCH_SIZE, TRUE_COORDS_CACHE
python rsa/full_sheet_rsa.py --output-dir <out> --k 100 --gpu-batch-size 64 --perm-batch-size 20
python rsa/full_sheet_rsa.py --demo   # tower geometry self-check
```

Defaults: `--n-perm 1000`, `--seed 42`, `--fdr-alpha 0.05`, `--pref-top-decile 0.1`, `--topo-n-draws 3`, `--topo-n-perm 50`, `--method spearman`, `--bin-sec 5.0`, `--skip-sec 5.0`, `--delay-sec 5.0`, `--tr 1.0`. The fMRI is `{preprocessed-dir}/group_average_{suffix}_cortex_59k.dtseries.nii`.

Outputs in `--output-dir`: `{seed}_av.csv` and `{seed}_av_rho.npy` per seed (columns `unit_index, tower, true_row, true_col, raster_row, raster_col, rho, p_perm, p_fdr`), `diff.csv`, `diff_p_perm.npy`, `diff_p_fdr.npy`, `metadata.json` (parameters, per-tower mean `rho`, hotspot composition and overlap, topography control, difference inference), and sheet maps `{seed}_av_rho_sheet_map.png`, `diff_rho_sheet_map.png`, `diff_p_sheet_map.png`, `diff_fdr_sig_sheet_map.png`. Per-seed maps share one color scale (1st to 99th percentile of pooled `rho`). The observed `rho` is checked against an existing `{seed}_av_rho.npy` before overwriting.

`characterize` in `shared/sheet_rsa.py` reports: per-tower mean `rho` for each seed and their difference, the cross-seed Pearson correlation of per-unit `rho`, hotspots (top `--pref-top-decile` of units by `rho`) per tower, their Jaccard overlap with a random-subset null, and a nearest-neighbor contiguity statistic of hotspot units on the true coordinates. Topography control: `rho` computed with each unit's true `k` sheet neighbors versus `k` units drawn uniformly from the same tower with coordinates ignored (`--topo-n-draws` draws). Adjacent units of a smooth sheet are redundant, so this comparison does not by itself test whether the topography is meaningful.

### Other scripts

- `python rsa/draw_rsa_borders.py --rsa-npy <searchlight.npy> --threshold-mode percentile --top-pct 5 --min-verts 10 [--out-dir <dir>]`: thresholds a searchlight array (`percentile`: top `--top-pct` percent of cortex jointly over both hemispheres; `sd`: `mean + --n-sd * SD`, default 2), removes connected islands smaller than `--min-verts` (default 10), and writes `{stem}_{top5pct|2sd}_{lh,rh}.border` with `wb_command -metric-rois-to-border`. Defaults point to the PE-AV group-average searchlight array.
- `python rsa/kreilability.py --subjects-list data/subjects.txt --timing-csv data/movie_timing.csv --embeddings-dir outputs/model_embeddings --template-cifti <template> --model pe-av-small-16-frame --modality av --k 100 150 200 --n-splits 50`: random split-half of subjects; per split and per `k`, searchlight on each half-average and the Pearson correlation of the two maps plus the Dice overlap of their top-10% masks; results in `{--outdir}/split_half_reliability_results.json` (resumable). Uses a per-half average midthickness surface (`--indiv-surf-template`), `--bin-sec` default 2.0, embeddings z-scored per dimension.
- `python rsa/rdm_diagonal.py --embeddings-dir ... --timing-csv ... --output-dir ... --model M --modality av`: within-clip (block-diagonal) and across-clip (off-diagonal) masks of the model RDM, written as `{stem}_rdm_{full,offdiag,blockdiag}.npy`, `_segment_labels.npy`, `_rdm_summary.json`; with the fMRI arguments (`--preprocessed-dir`, `--template-cifti`, `--glasser-dlabel`, surfaces, `--workbench`) it also writes `{stem}_rsa_diagonal_maps.dscalar.nii` with `glasser_{method}_rho_{offdiag,blockdiag}` and `searchlight_{method}_rho_{offdiag,blockdiag}`.
- `python rsa/channel_subset_searchlight.py [--output-dir <dir>] [--gpu-batch-size 512]`: writes the columns of the PE-AV `av` embedding belonging to one channel cluster as a standalone embedding and runs `searchlight.py` on it. Cluster, labels file and all other paths are constants at the top of the file.

## Tests

```bash
pytest tests/test_rsa.py tests/test_model_norm.py tests/test_sheet_rsa.py tests/test_perm_searchlight.py \
       tests/test_max_uni.py tests/test_av_derived_maps.py tests/test_residualized_maps.py \
       tests/test_av_scramble_permutation_inference.py tests/test_paired_bootstrap_resampling.py \
       tests/test_channel_class_rsa.py tests/test_channel_subset_searchlight.py
python rsa/shared/residuals.py      # projection residual orthogonality self-check
python rsa/localizer_naming.py      # name construction checks
python rsa/spatial_stats.py         # spatial statistics self-check
python rsa/full_sheet_rsa.py --demo # sheet tower geometry
```
