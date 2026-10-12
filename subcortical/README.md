# subcortical: RSA, CKA and encoding of subcortical voxels

Applies the cortical movie-watching analyses to the subcortical volume voxels of the 7T HCP CIFTI files (the part `preprocess_individual.py` discards when it extracts cortex). Binning, embedding alignment, the searchlight kernels, CKA, the encoding variance partition and CIFTI input and output are imported from `rsa/`, `cka/`, `encoding/` and `cifti_io.py`; the code here handles per-structure bookkeeping, neighbor construction and visualization. `RESULTS_group_average.md` lists the results of the first correlation-distance group-average run.

## Method

Data. Raw subcortical dtseries of 19 structures (cerebellum, thalamus, caudate, putamen, pallidum, accumbens, amygdala, hippocampus and ventral diencephalon, each left and right, and the brain stem; 62,053 voxels), in the order of `subcortical_io.SUBCORTICAL_STRUCTURES`. Per run, optional Savitzky-Golay filtering, percent signal change and global signal regression (`--sg-filter`, `--psc`, `--gsr`; none by default, tag `raw`); runs are then concatenated. The group average is the running mean over the subjects in `--subjects-list` (175 subjects), cached as `group_average_{tag}_subcortical.dtseries.nii` and `group_average_{tag}_run_trs.npy`.

Segments. As in the cortical pipeline: windows of `--bin-sec` seconds every `--skip-sec` seconds (default `--bin-sec`), delayed by `--delay-sec`, within each run, averaged over time and z-scored within run (626 windows at 5 s). Model embeddings are cut into the same windows (hemodynamic convolution only with `--hrf`).

Neighbors. For voxel `v`, the neighborhood is `v` plus its `k - 1` nearest voxels (`--k`, default 100) inside the same structure. The subcortical grid is identical for all subjects, so neighbor arrays are computed once and cached in `outputs/subcortical/_neighbor_cache/`. Neighbors are the `k` nearest voxels by shortest path on the within-structure voxel graph (26-connectivity, edge weight in millimetres, Dijkstra); if a source voxel reaches fewer than `k` voxels, the remainder is filled by Euclidean distance. For the cerebellum, neighbors come from geodesic distance on the SUIT cerebellar surface (`PIAL_SUIT` mesh, each voxel mapped to its nearest `PIAL_FSL` vertex, distances from `wb_command -surface-geodesic-distance-all-to-all`), with the voxel-graph geodesic as fallback if SUITPy or `wb_command` fail. `subcortical_io.neighbor_columns` offsets each structure's neighbors by the structure's first column, so one searchlight pass over all 62,053 voxels equals the per-structure passes.

Correlation-distance searchlight RSA (`subcortical_rsa.py`). The brain representational dissimilarity matrix (RDM) of a neighborhood is `1 - Pearson correlation` between segments of the neighborhood's voxel patterns, and the model RDM is the same quantity for the embedding (z-scored per run). `rho(v)` is the Spearman (`--method spearman`, default) or Pearson correlation between the lower triangles of the two RDMs. One-tailed uncorrected p-values come from `t = rho * sqrt(df) / sqrt(1 - rho^2)` with `df = n_segments - 2`, followed by Benjamini-Hochberg false-discovery-rate correction at 0.05 across all voxels. These outputs carry `pearson_spearman` in their names.

Euclidean RSA of audio, video and joint embeddings (`subcortical_avj.py rsa`). As `rsa/rdm_searchlight.py --distance euclidean --comparator spearman`: the brain RDM is the squared Euclidean distance between window patterns of the neighborhood, the model RDMs are the Euclidean distances between per-run-centered windows of the audio (A), video (V) and joint (J) embeddings of a model (A and V are the model's own, tag `unimodal_own`). For each x in {a, v, j}: `rho_x` is the Spearman correlation of the brain RDM with the x RDM; `sp_x` the Spearman correlation of the brain RDM with the residual of the x RDM after least-squares regression, with intercept, on the other two (Tong et al. 2022); `partial_x` the partial Spearman correlation given the other two (Clarke & Tyler 2014).

CKA (`subcortical_avj.py cka`). As `cka/cka_searchlight.py run`: with H the centering matrix and Y the (windows, voxels) responses of a neighborhood, the brain matrix is `B = H Y Y' H`, the model matrix `G_x = H X_x X_x' H` (per-run-centered embedding), and `cka_x = <B, G_x>_F / (||B||_F ||G_x||_F)`, not cross-validated. `sp_x = [R^-1 c]_x / sqrt([R^-1]_xx)` with c the three CKA values and R the cosines among the three G; the commonality maps split the squared multiple correlations `R2(S) = c_S' R_SS^-1 c_S` of the seven subsets S of {A, V, J} by inclusion-exclusion (see `cka/README.md`).

Encoding (`encoding/variance_partition.py --fmri-space subcortical`). Voxelwise banded ridge regression of the binned responses on A, V, J and their four combinations, evaluated leave-one-run-out over the windows not in the repeated clip (`--split loro`), embeddings demeaned over the same rows as the responses (`--feature-scaling demean`); R2 is the pooled held-out `1 - SSR/SST`, and the seven subset R2 values are partitioned by inclusion-exclusion, e.g. `unique_j = R2(AVJ) - R2(AV)` (see `encoding/README.md`).

Nuclei. Eight nuclei are too small for a searchlight: inferior colliculus (IC), superior colliculus (SC), medial geniculate (MGN) and lateral geniculate (LGN), left and right. Each is the set of voxels of its parent structure (brain stem for IC and SC; the thalamus of the same side for MGN and LGN) within `--nucleus-radius` mm (default 5) of an MNI seed coordinate (`NUCLEI_SEEDS` in `subcortical_io.py`). `subcortical_rsa.py` correlates one correlation-distance RDM over all voxels of the nucleus with the model RDM (`rho` and `p` in the CSV); for the other analyses, the nucleus value is the mean of the searchlight or voxel map over the nucleus voxels.

Noise ceiling. With `RDM_i(v)` the correlation-distance neighborhood RDM of subject `i` at voxel `v` and `N` subjects, `NC_upper(v)` is the mean over `i` of the Spearman correlation between `RDM_i(v)` and the mean of all `N` subjects' RDMs, and `NC_lower(v)` replaces that mean by the mean over the other `N - 1` subjects. It bounds a subject-level rho, not the rho of the group average.

Integration contrast (partial RSA). For an audiovisual embedding, the partial correlation between the correlation-distance brain RDM and the target RDM after regressing out the nuisance RDMs configured for the run in `rsa/shared/model_registry.py::PARTIAL_RSA_RUNS` (the model's own unimodal audio and video embeddings for intact runs; the single real modality for the `clsav_from_*` controls). Controls are `avscramble` (real audio with temporally permuted video), `clsav_from_a` (audio with a fixed placeholder video) and `clsav_from_v` (video with a placeholder audio). `diff_maps.py` forms per model `binding = partial_intact - partial_avscramble`, `modality_presence_diff = partial_intact - max(partial_clsav_from_a, partial_clsav_from_v)`, and the plain-RSA differences `searchlight_rho(intact) - searchlight_rho(control)`.

## Scripts

| Script | Purpose |
|---|---|
| `subcortical_io.py` | Structure catalogue, extraction, group-average accumulation, neighbor builders (`structure_neighbors`, `neighbor_columns`), nucleus masks, template CIFTI. `python subcortical/subcortical_io.py` runs a self-check |
| `precompute_neighbors.py` | Builds and caches all neighbor arrays (one subject suffices): `--raw-dir DIR --subject ID --k 100 --cache-dir DIR` |
| `subcortical_rsa.py` | Correlation-distance searchlight and nuclei RSA, for the group average (`--subject group_average`, needs `--subjects-list`) or one subject (`--subject ID`, streaming preprocessing) |
| `subcortical_avj.py` | Euclidean RSA (`rsa`) and CKA (`cka`) of the group average with the A, V and J embeddings of each `--models` entry |
| `subcortical_noise_ceiling.py` | Noise ceiling over `--subjects ID ...` (streaming, raw preprocessing) |
| `subcortical_partial_rsa.py` | Integration contrast for one `--run KEY` of `PARTIAL_RSA_RUNS`; flags as `subcortical_rsa.py` |
| `diff_maps.py` | Consolidates the contrasts above for the model roster at the top of the file (no flags) |
| `subcortical_visualization.py` | Builds the Workbench bundle (`--output-dir`, `--template`, `--bundle-dir`, `--wb-command`, `--download-atlases`) |
| `analysis.sh`, `run_diff_study.sh` | Runners (below) |

`subcortical_rsa.py` requires `--raw-dir`, `--timing-csv`, `--embeddings-dir`, `--template-cifti`, `--output-dir`, `--model`, `--modality` and `--bin-sec`. Defaults: `--k 100`, `--delay-sec 5`, `--method spearman`, `--tr 1`, `--nucleus-radius 5`, `--gpu-batch-size 512`. Embeddings are read from `{embeddings-dir}/{model}/bin{N}s_skip{N}s/{model}_{modality}.npy`. `subcortical_avj.py` requires `--models`, `--group-average-dir`, `--timing-csv`, `--embeddings-dir`, `--template-cifti`, `--output-dir` and `--neighbor-cache-dir`; defaults `--k 100`, `--bin-sec 5`, `--skip-sec 5`, `--delay-sec 5`, `--model-norm center`.

## Usage

Requires Connectome Workbench (`wb_command`) and, for the cerebellar surface geodesic, SUITPy. Paths and parameters are set at the top of `analysis.sh` (`K=100`, `BIN_SEC=5.0`, `SKIP_SEC=5.0`, `DELAY_SEC=5.0`, `TR=1.0`, `METHOD=spearman`, the `MODELS` list of `model:modality` entries and the `AVJ_MODELS` list).

```bash
bash subcortical/analysis.sh MODE [N_BLOCKS]
bash subcortical/run_diff_study.sh [plain|partial|consolidate|all]
```

| `MODE` | Action |
|---|---|
| `neighbors` | Build the neighbor caches. Run first |
| `groupavg` (default) | Group-average correlation-distance searchlight and nuclei RSA for every `MODELS` entry, then rebuild the visualization bundle (unless `REFRESH_VISUALIZATION=false`) |
| `avj` | Euclidean RSA, CKA and the encoding variance partition of the group average for every `AVJ_MODELS` entry (PE-AV and Nemotron layer 18 by default) |
| `noiseceiling` | Noise ceiling on the subjects in `NC_SUBJECTS` (10 IDs by default; override by environment variable) |
| `persubject` | Per-subject RSA for every subject in `subjects.txt` and every `MODELS` entry |
| `groupstats` | `rsa/group_stats.py --fname-prefix rsa_subcortical --config k{k}_delay{d}s_bin{b}s_skip{s}s_pearson_{method}_zscore` across the per-subject maps |
| `visualize` | Rebuild the Workbench bundle |

`MODELS` holds PE-AV with its three controls and, unless `SKIP_LAYER_SWEEP=true`, the layer sweeps of the omni3b, topoomni and nemotron families (`_mp` and `_lt` variants where they exist) and, unless `SKIP_SCRAMBLE_DUMMY_SWEEP=true`, the `_avscramble`, `_clsav_from_a` and `_clsav_from_v` controls of each native audiovisual model. In `run_diff_study.sh`, `plain` runs `analysis.sh groupavg`; `partial` runs `subcortical_partial_rsa.py` for every model in `BASE_MODELS` and condition in `CONDITIONS` (`intact avscramble clsav_from_a clsav_from_v`; restrict with the space-separated `DIFF_STUDY_MODELS` and `DIFF_STUDY_CONDITIONS`); `consolidate` runs `diff_maps.py`, then `analysis.sh visualize`.

## Outputs

`config` is `k{k}_delay{delay}s_bin{bin}s_skip{skip}s_pearson_{method}_zscore` for the correlation-distance analyses (embeddings z-scored within run; the noise ceiling, which uses no embeddings, has no `_zscore`) and `tag` is the preprocessing tag (`raw`, `sg_psc`, ...). Existing outputs are skipped. Under `outputs/subcortical/`:

| Path | Content |
|---|---|
| `subcortical_template.dscalar.nii` | One-map CIFTI whose brain-model axis is the 19 concatenated structures; used as `--template-cifti` (built automatically) |
| `_neighbor_cache/` | `{STRUCTURE}_neighbors_k{k}_geodesic.npy` (`(n_voxels, k)` indices local to the structure); cerebellum surface results in `CEREBELLUM_{LEFT,RIGHT}/` |
| `group_average_cache/` | Group-average dtseries and `run_trs.npy` |
| `group_average/{model}_{modality}/{config}/` | `rsa_subcortical_{tag}_{config}_searchlight.npy` (`rho` for all voxels); `..._maps.dscalar.nii` (maps `searchlight_rho`, `sigmap_uncorr`, `sigmap_fdr`, `fdr_mask`; sigmap = sign(rho) x -log10 p); `nuclei_roi_rsa_{tag}_{config}.csv` (columns `nucleus`, `n_vox`, `rho`, `p`, `parent`); `struct_summary.json` (per structure: voxels, mean and max rho, FDR-passing voxels, neighbor mode) |
| `group_average/{model}/rsa_subcortical_{tag}_k{k}_delay{d}s_bin{b}s_skip{s}s_euclidean_spearman_center_unimodal_own_{kind}.dscalar.nii` | `kind` = `models` (`rho_a`, `rho_v`, `rho_j`), `semipartial` (`sp_a`, `sp_v`, `sp_j`), `partial` (`partial_a`, `partial_v`, `partial_j`); definitions in `..._provenance.json` |
| `group_average/{model}/cka_subcortical_{tag}_k{k}_delay{d}s_bin{b}s_skip{s}s_noncv_center_unimodal_own_{kind}.dscalar.nii` | `kind` = `models` (`cka_a`, `cka_v`, `cka_j`), `semipartial` (`sp_a`, `sp_v`, `sp_j`), `commonality` (`r2_*`, `unique_*`, `shared_*`); `..._provenance.json` |
| `group_average/{model}/delay{d}s_bin{b}s_skip{s}s/encoding_{r2,pearson_r}_loro_demean_unimodal_own_{models,partition}.dscalar.nii` | Subset R2 (`r2_a` ... `r2_avj`) and Pearson r, the partition (`unique_*`, `shared_*`, `gain_j_over_*`, `j_minus_av`, `shared_av`); `..._per_fold.npz`, `..._provenance.json` |
| `subject_data/{subject}/{model}_{modality}/{config}/` | The correlation-distance files for one subject |
| `group_average/{model}_{modality}_INTEGRATION/{config}/integration_partial_r_searchlight.dscalar.nii` | Partial-RSA map named `partial_rsa_{model}_{modality}_INTEGRATION` |
| `group_average/{model}_av/dummy_partial_diff_maps.dscalar.nii` | `modality_presence_diff` |
| `group_average/{model}_avscramble_av/scramble_consolidated_maps.dscalar.nii` | `plain_rsa_scrambled`, `partial_rsa_scrambled`, `diff_intact_minus_scrambled_plain_rsa`, `binding` |
| `group_average/{model}_clsav_from_{a,v}_av/dummy_consolidated_maps.dscalar.nii` | `plain_rsa_dummy`, `diff_intact_minus_dummy_plain_rsa`, `partial_rsa_dummy` |
| `noise_ceiling/noise_ceiling_subcortical_{N}subs_{config}.dscalar.nii` | Maps `nc_lower`, `nc_upper` |
| `groupstats/{model}_{modality}/{config}/group_stats_{N}subs.dscalar.nii` | `mean_rho`, `t_stat`, `sigmap_uncorr`, `sigmap_fdr`, `fdr_mask` (output of `rsa/group_stats.py`) |
| `workbench_visualization/` | Derived bundle, safe to delete and rebuild: `subcortical_wb_view.spec`, `manifest.json`, `README_wb_view.md`, meshes, and one `{category}__{model_modality}__{config}__{result}__{STRUCTURE}.func.gii` per result and anatomical unit (SUIT cerebellum, brain stem, one bilateral file per paired structure) |

To view, open the movie scene (`outputs/rsa_movie.scene`) in `wb_view` and import `workbench_visualization/subcortical_wb_view.spec`. To add an analysis without rebuilding, place its exported `.func.gii` next to the others for that structure and load it manually; vertex counts are validated against the existing mesh.

## Current results (October 2026)

Group average of 175 subjects, 626 windows of 5 s (stride 5 s, delay 5 s), k = 100, A and V from the model itself (`unimodal_own`). Each cell is the mean of the map over the voxels of the structure; for the nuclei (5-mm spheres, 4 to 116 voxels) it is the mean of the searchlight or voxel map over the nucleus voxels, not a nucleus-level RDM. Columns: Euclidean-Spearman RSA (`rho_x`, `sp_j`, `partial_j`); `rho (1 - r)`, the earlier correlation-distance searchlight rho of the joint embedding (z-scored embeddings) for comparison; non-cross-validated CKA (`c_x`, `CKA sp_j`); held-out encoding R² (leave-one-run-out, demeaned embeddings) and `unique J` = R²(AVJ) − R²(AV).

PE-AV

| Structure | voxels | rho_a | rho_v | rho_j | sp_j | partial_j | rho (1 - r) | c_a | c_v | c_j | CKA sp_j | R²(A) | R²(V) | R²(J) | R²(AVJ) | unique J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cerebellum L | 17,105 | +0.0277 | +0.0431 | +0.0369 | -0.0042 | -0.0013 | +0.0392 | +0.0563 | +0.0664 | +0.0755 | +0.0205 | +0.0080 | +0.0213 | +0.0239 | +0.0214 | -0.0013 |
| Cerebellum R | 17,638 | +0.0341 | +0.0471 | +0.0384 | -0.0114 | -0.0076 | +0.0383 | +0.0640 | +0.0640 | +0.0793 | +0.0255 | +0.0130 | +0.0134 | +0.0263 | +0.0207 | +0.0015 |
| Thalamus L | 2,515 | +0.0267 | +0.0403 | +0.0276 | -0.0132 | -0.0131 | +0.0232 | +0.0561 | +0.0703 | +0.0778 | +0.0202 | +0.0032 | +0.0107 | +0.0131 | +0.0104 | -0.0009 |
| Thalamus R | 2,461 | +0.0238 | +0.0353 | +0.0275 | -0.0087 | -0.0065 | +0.0257 | +0.0513 | +0.0692 | +0.0742 | +0.0180 | +0.0024 | +0.0160 | +0.0115 | +0.0118 | -0.0023 |
| LGN L | 4 | +0.0647 | +0.0496 | +0.0608 | +0.0019 | +0.0021 | +0.0151 | +0.0627 | +0.0845 | +0.0895 | +0.0198 | +0.0002 | -0.0096 | -0.0023 | -0.0066 | +0.0001 |
| LGN R | 6 | +0.0521 | +0.0722 | +0.0526 | -0.0272 | -0.0219 | +0.0234 | +0.0680 | +0.0881 | +0.0958 | +0.0237 | -0.0023 | +0.0009 | +0.0051 | +0.0021 | +0.0058 |
| MGN L | 70 | +0.0530 | +0.0484 | +0.0468 | -0.0105 | -0.0104 | +0.0250 | +0.0676 | +0.0840 | +0.0933 | +0.0243 | -0.0002 | +0.0014 | +0.0068 | +0.0019 | -0.0000 |
| MGN R | 70 | +0.0701 | +0.0560 | +0.0592 | -0.0132 | -0.0108 | +0.0325 | +0.0686 | +0.0874 | +0.0948 | +0.0222 | +0.0104 | +0.0159 | +0.0198 | +0.0197 | -0.0011 |
| IC L | 108 | +0.0124 | +0.0536 | +0.0128 | -0.0336 | -0.0380 | +0.0292 | +0.0539 | +0.0630 | +0.0713 | +0.0186 | +0.0153 | +0.0028 | +0.0116 | +0.0158 | -0.0010 |
| IC R | 116 | +0.0171 | +0.0379 | +0.0153 | -0.0191 | -0.0233 | +0.0329 | +0.0525 | +0.0663 | +0.0717 | +0.0161 | +0.0115 | +0.0121 | +0.0121 | +0.0208 | -0.0013 |
| SC L | 101 | +0.0262 | +0.0508 | +0.0188 | -0.0347 | -0.0367 | +0.0349 | +0.0411 | +0.0623 | +0.0593 | +0.0052 | +0.0115 | +0.0383 | +0.0304 | +0.0361 | -0.0007 |
| SC R | 102 | +0.0397 | +0.0433 | +0.0286 | -0.0268 | -0.0247 | +0.0372 | +0.0441 | +0.0649 | +0.0619 | +0.0049 | +0.0289 | +0.0585 | +0.0466 | +0.0608 | -0.0013 |
| Hippocampus L | 1,484 | +0.0341 | +0.0148 | +0.0222 | -0.0029 | -0.0047 | +0.0298 | +0.0527 | +0.0694 | +0.0790 | +0.0262 | -0.0035 | +0.0088 | +0.0063 | +0.0031 | -0.0010 |
| Hippocampus R | 1,512 | +0.0375 | +0.0068 | +0.0249 | +0.0056 | +0.0043 | +0.0332 | +0.0473 | +0.0698 | +0.0775 | +0.0266 | -0.0032 | +0.0170 | +0.0112 | +0.0115 | -0.0014 |
| Amygdala L | 620 | +0.0425 | +0.0626 | +0.0602 | +0.0077 | +0.0071 | +0.0417 | +0.0596 | +0.0910 | +0.0941 | +0.0229 | +0.0101 | +0.0497 | +0.0417 | +0.0486 | -0.0009 |
| Amygdala R | 654 | +0.0345 | +0.0560 | +0.0561 | +0.0123 | +0.0127 | +0.0405 | +0.0505 | +0.0958 | +0.0912 | +0.0175 | +0.0085 | +0.0709 | +0.0546 | +0.0685 | -0.0013 |
| Caudate L | 1,370 | +0.0440 | +0.0105 | +0.0328 | +0.0113 | +0.0087 | +0.0211 | +0.0416 | +0.0431 | +0.0544 | +0.0202 | +0.0031 | -0.0008 | +0.0044 | +0.0018 | -0.0001 |
| Caudate R | 1,431 | +0.0187 | +0.0016 | +0.0096 | -0.0000 | -0.0008 | +0.0174 | +0.0387 | +0.0436 | +0.0519 | +0.0168 | -0.0014 | +0.0010 | +0.0005 | -0.0020 | -0.0015 |
| Putamen L | 2,114 | +0.0170 | +0.0263 | +0.0077 | -0.0249 | -0.0251 | +0.0169 | +0.0503 | +0.0498 | +0.0617 | +0.0194 | +0.0016 | -0.0012 | +0.0018 | -0.0010 | -0.0007 |
| Putamen R | 2,016 | +0.0132 | +0.0107 | +0.0061 | -0.0104 | -0.0104 | +0.0152 | +0.0458 | +0.0473 | +0.0581 | +0.0190 | -0.0004 | -0.0014 | +0.0003 | -0.0027 | -0.0006 |
| Brain stem | 6,731 | +0.0191 | +0.0318 | +0.0243 | -0.0061 | -0.0048 | +0.0125 | +0.0590 | +0.0723 | +0.0797 | +0.0191 | -0.0009 | +0.0011 | +0.0008 | -0.0020 | -0.0010 |
| All 62,053 voxels | 62,053 | +0.0278 | +0.0374 | +0.0313 | -0.0071 | -0.0053 | +0.0305 | +0.0572 | +0.0656 | +0.0759 | +0.0217 | +0.0062 | +0.0129 | +0.0170 | +0.0140 | -0.0004 |

Nemotron layer 18 (mean-pooled)

| Structure | voxels | rho_a | rho_v | rho_j | sp_j | partial_j | rho (1 - r) | c_a | c_v | c_j | CKA sp_j | R²(A) | R²(V) | R²(J) | R²(AVJ) | unique J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cerebellum L | 17,105 | +0.0494 | +0.0453 | +0.0590 | +0.0132 | +0.0129 | +0.0615 | +0.0553 | +0.0762 | +0.0852 | +0.0029 | +0.0088 | +0.0232 | +0.0330 | +0.0289 | +0.0004 |
| Cerebellum R | 17,638 | +0.0544 | +0.0256 | +0.0455 | +0.0241 | +0.0222 | +0.0591 | +0.0640 | +0.0696 | +0.0844 | +0.0083 | +0.0109 | +0.0169 | +0.0318 | +0.0235 | +0.0010 |
| Thalamus L | 2,515 | +0.0359 | +0.0179 | +0.0311 | +0.0152 | +0.0150 | +0.0399 | +0.0453 | +0.0677 | +0.0746 | +0.0048 | +0.0022 | +0.0109 | +0.0163 | +0.0113 | +0.0005 |
| Thalamus R | 2,461 | +0.0420 | +0.0373 | +0.0490 | +0.0071 | +0.0107 | +0.0422 | +0.0441 | +0.0650 | +0.0706 | -0.0014 | +0.0022 | +0.0148 | +0.0186 | +0.0156 | -0.0002 |
| LGN L | 4 | +0.0363 | +0.0244 | +0.0329 | -0.0044 | -0.0004 | +0.0254 | +0.0439 | +0.0820 | +0.0837 | -0.0046 | -0.0020 | -0.0106 | -0.0153 | -0.0201 | -0.0008 |
| LGN R | 6 | +0.0377 | +0.0564 | +0.0716 | +0.0427 | +0.0375 | +0.0385 | +0.0474 | +0.0863 | +0.0891 | -0.0026 | -0.0019 | -0.0042 | +0.0021 | -0.0026 | +0.0028 |
| MGN L | 70 | +0.0398 | +0.0133 | +0.0287 | +0.0167 | +0.0173 | +0.0388 | +0.0566 | +0.0828 | +0.0910 | +0.0021 | +0.0034 | +0.0026 | +0.0048 | +0.0036 | -0.0002 |
| MGN R | 70 | +0.0537 | +0.0463 | +0.0673 | +0.0371 | +0.0360 | +0.0477 | +0.0472 | +0.0836 | +0.0884 | +0.0049 | +0.0058 | +0.0144 | +0.0223 | +0.0181 | +0.0000 |
| IC L | 108 | +0.0337 | +0.0320 | +0.0441 | +0.0151 | +0.0201 | +0.0467 | +0.0379 | +0.0626 | +0.0669 | +0.0020 | +0.0190 | +0.0072 | +0.0200 | +0.0260 | -0.0010 |
| IC R | 116 | +0.0206 | +0.0327 | +0.0392 | +0.0033 | +0.0148 | +0.0490 | +0.0412 | +0.0607 | +0.0657 | -0.0023 | +0.0125 | +0.0053 | +0.0165 | +0.0174 | +0.0012 |
| SC L | 101 | +0.0267 | +0.0469 | +0.0527 | +0.0035 | +0.0108 | +0.0648 | +0.0341 | +0.0650 | +0.0648 | -0.0098 | -0.0082 | +0.0512 | +0.0527 | +0.0534 | +0.0002 |
| SC R | 102 | +0.0360 | +0.0470 | +0.0568 | +0.0068 | +0.0152 | +0.0698 | +0.0447 | +0.0608 | +0.0658 | -0.0094 | +0.0023 | +0.0541 | +0.0655 | +0.0672 | +0.0007 |
| Hippocampus L | 1,484 | +0.0167 | +0.0458 | +0.0536 | +0.0343 | +0.0300 | +0.0540 | +0.0342 | +0.0909 | +0.0912 | +0.0151 | -0.0050 | +0.0188 | +0.0226 | +0.0158 | +0.0032 |
| Hippocampus R | 1,512 | +0.0179 | +0.0490 | +0.0550 | +0.0223 | +0.0232 | +0.0543 | +0.0305 | +0.0933 | +0.0896 | +0.0053 | -0.0059 | +0.0279 | +0.0309 | +0.0253 | +0.0028 |
| Amygdala L | 620 | +0.0602 | +0.0716 | +0.0831 | +0.0060 | +0.0034 | +0.0645 | +0.0602 | +0.1075 | +0.1183 | +0.0299 | +0.0144 | +0.0448 | +0.0507 | +0.0446 | +0.0012 |
| Amygdala R | 654 | +0.0466 | +0.0874 | +0.0907 | -0.0061 | -0.0051 | +0.0592 | +0.0434 | +0.1132 | +0.1154 | +0.0257 | +0.0153 | +0.0652 | +0.0691 | +0.0637 | -0.0000 |
| Caudate L | 1,370 | +0.0383 | +0.0095 | +0.0222 | +0.0045 | +0.0069 | +0.0328 | +0.0378 | +0.0422 | +0.0499 | +0.0008 | +0.0008 | -0.0022 | +0.0019 | -0.0013 | +0.0000 |
| Caudate R | 1,431 | +0.0165 | +0.0150 | +0.0182 | -0.0041 | -0.0007 | +0.0258 | +0.0311 | +0.0438 | +0.0469 | -0.0069 | -0.0019 | -0.0026 | -0.0011 | -0.0044 | +0.0001 |
| Putamen L | 2,114 | +0.0245 | +0.0286 | +0.0296 | -0.0193 | -0.0133 | +0.0275 | +0.0378 | +0.0461 | +0.0524 | -0.0025 | -0.0009 | -0.0013 | +0.0005 | -0.0029 | -0.0002 |
| Putamen R | 2,016 | +0.0253 | +0.0176 | +0.0182 | -0.0314 | -0.0203 | +0.0240 | +0.0363 | +0.0420 | +0.0477 | -0.0058 | -0.0011 | -0.0023 | -0.0003 | -0.0042 | -0.0001 |
| Brain stem | 6,731 | +0.0320 | +0.0379 | +0.0444 | +0.0006 | +0.0031 | +0.0214 | +0.0425 | +0.0667 | +0.0701 | -0.0077 | -0.0016 | +0.0009 | +0.0025 | -0.0005 | -0.0002 |
| All 62,053 voxels | 62,053 | +0.0423 | +0.0349 | +0.0471 | +0.0111 | +0.0115 | +0.0483 | +0.0513 | +0.0699 | +0.0784 | +0.0028 | +0.0055 | +0.0146 | +0.0229 | +0.0177 | +0.0005 |

Comparison with correlation distance. Over all 62,053 voxels the mean rho_j is almost unchanged (PE-AV 0.0313 Euclidean against 0.0305 correlation distance; Nemotron 0.0471 against 0.0483), but the maps differ: the spatial Pearson correlation between the two rho_j maps over voxels is r = 0.53 for PE-AV (within left cerebellum 0.51, left thalamus 0.71, left hippocampus 0.41, brain stem −0.07) and r = 0.38 for Nemotron (0.36, 0.42, 0.20, 0.11). The two runs differ in brain distance (squared Euclidean against 1 − Pearson r, which removes the neighborhood-mean response) and in embedding normalization (centered against z-scored).

Joint beyond audio and video. For PE-AV the joint RDM adds nothing measurable beyond its own audio and video RDMs: mean sp_j = −0.0071 and partial_j = −0.0053 over all voxels, positive only in amygdala, caudate L, hippocampus R, pallidum, accumbens and LGN L (up to +0.012), and encoding unique J = −0.0004. For Nemotron sp_j = +0.0111 and partial_j = +0.0115 over all voxels (sp_j +0.034 / +0.022 in hippocampus L / R, +0.024 in cerebellum R, +0.037 in MGN R), while encoding unique J is +0.0005 (hippocampus +0.003). CKA ranks c_j above c_v above c_a in most structures for both models (c_v ≥ c_j in SC, in amygdala R for PE-AV and in hippocampus R for Nemotron); the CKA semi-partial of J is +0.022 for PE-AV and +0.003 for Nemotron over all voxels.

Where the signal is. Encoding R² is highest in amygdala (R²(AVJ) 0.045 to 0.069), the superior colliculus spheres (0.036 to 0.067) and cerebellum (0.021 to 0.029), carried mostly by V; it is within 0.002 of zero or below in caudate, putamen, pallidum, accumbens and the brain stem as a whole. rho_j is highest in amygdala and the MGN and LGN spheres and lowest in putamen and caudate; c_j is lowest in caudate and putamen.

Noise ceiling. The ceiling on disk (`noise_ceiling_subcortical_10subs_..._pearson_spearman`) is for single-subject correlation-distance RSA from 10 subjects: nc_lower ranges from 0.0034 (accumbens L, pallidum L) to 0.0112 (cerebellum L), 0.0057 / 0.0061 in thalamus L / R, 0.0066 / 0.0068 in hippocampus, 0.0059 / 0.0060 in amygdala, 0.0062 in the brain stem and 0.0064 to 0.0079 in the nuclei; nc_upper is 0.300 to 0.308 everywhere. It bounds subject-level rho, so it is not a ceiling for the group-average values above, and there is no Euclidean, CKA or encoding ceiling for subcortex yet.

## Tests

```bash
pytest tests/test_subcortical_visualization.py tests/test_subcortical_avj.py -v
```

`test_subcortical_avj.py` checks that `neighbor_columns` keeps every neighborhood inside its structure and that one searchlight pass over the concatenated neighbor columns equals the per-structure passes.
