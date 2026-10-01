# subcortical: representational similarity analysis (RSA) of subcortical voxels

Applies the cortical movie-watching RSA to the subcortical volume voxels of the 7T HCP CIFTI files (the part `preprocess_individual.py` discards when it extracts cortex). Binning, embedding alignment, the searchlight kernel and CIFTI input/output are reused from `rsa/shared/rsa_utils.py`, `rsa/searchlight.py` and `cifti_io.py`; the code here handles per-structure bookkeeping, neighbor construction and visualization. `RESULTS_group_average.md` holds numerical results of one run.

## What is computed

**Data.** Raw subcortical dtseries of 19 structures: cerebellum left and right, brain stem, thalamus, caudate, putamen, pallidum, accumbens, amygdala, hippocampus, ventral diencephalon (each left and right except the brain stem), ordered as listed in `subcortical_io.SUBCORTICAL_STRUCTURES`. Per run, optional Savitzky-Golay filtering, percent signal change and global signal regression (`--sg-filter`, `--psc`, `--gsr`; none by default, tag `raw`), then runs are concatenated. The group average is the running mean over the subjects in `--subjects-list`, cached as `group_average_{tag}_subcortical.dtseries.nii` and `group_average_{tag}_run_trs.npy`.

**Segments.** As in the cortical pipeline: windows of `--bin-sec` seconds every `--skip-sec` seconds (default: equal to `--bin-sec`), delayed by `--delay-sec`, within each run, averaged over time and z-scored within run (`fmri_binned`, segments x voxels). Model embeddings are cut into the same windows (hemodynamic convolution only with `--hrf`).

**Searchlight RSA (per structure).** For voxel `v`, the neighborhood is `v` plus its `k - 1` nearest voxels (`--k`, default 100) inside the same structure. The brain representational dissimilarity matrix (RDM) of the neighborhood is `1 - Pearson correlation` between segments of the neighborhood's voxel patterns; the model RDM is the same quantity for the embedding. `rho(v)` is the rank (`--method spearman`, default) or Pearson correlation between the lower triangles of the two RDMs. One-tailed uncorrected p-values come from `t = rho * sqrt(df) / sqrt(1 - rho^2)` with `df = n_segments - 2`, followed by Benjamini-Hochberg false discovery rate correction at 0.05 across all voxels.

**Neighbors.** The subcortical grid is identical for all subjects, so neighbor arrays are computed once and cached in `outputs/subcortical/_neighbor_cache/`. For every structure, neighbors are the `k` nearest voxels by shortest path on the within-structure voxel graph (26-connectivity, edge weight = millimetre distance, Dijkstra); for a source voxel with fewer than `k` reachable voxels the remainder is filled by Euclidean distance. For the cerebellum, neighbors are by geodesic distance on the SUIT cerebellar surface (`PIAL_SUIT` mesh, each voxel mapped to its nearest `PIAL_FSL` vertex, distances from `wb_command -surface-geodesic-distance-all-to-all`); if SUITPy or `wb_command` fail, the voxel-graph geodesic is used. The voxel-graph geodesic for the cerebellum is cached as well.

**Nuclei (region-of-interest RSA).** Eight nuclei are too small for a searchlight: inferior colliculus (IC), superior colliculus (SC), medial geniculate (MGN), lateral geniculate (LGN), left and right. Each is the set of voxels of its parent structure (brain stem for IC and SC; thalamus left or right for MGN and LGN) within `--nucleus-radius` mm (default 5) of an MNI seed coordinate (`NUCLEI_SEEDS` in `subcortical_io.py`). One RDM is computed over all voxels of the nucleus and correlated with the model RDM (`rho` and `p` in the CSV).

**Noise ceiling.** For `N` subjects, `RDM_i(v)` is subject `i`'s neighborhood RDM at voxel `v`. `NC_upper(v)` is the mean over `i` of the Spearman correlation between `RDM_i(v)` and the mean of all `N` subjects' RDMs; `NC_lower(v)` replaces that mean by the mean over the other `N - 1` subjects. A model's group-average `rho` cannot exceed `NC_upper` in expectation; read each `rho` relative to its own ceiling, because subcortical signal-to-noise differs from cortex.

**Integration contrast (partial RSA).** For an audiovisual embedding, the partial correlation between the brain RDM and the target RDM after regressing out the nuisance RDMs configured for that run in `rsa/shared/model_registry.py::PARTIAL_RSA_RUNS` (the unimodal audio and video embeddings of the same model for intact runs; the single real modality for the `clsav_from_*` controls). Controls: `avscramble` (real audio with temporally permuted video), `clsav_from_a` (audio with a fixed placeholder video), `clsav_from_v` (video with a placeholder audio). `diff_maps.py` then forms, per model: `binding = partial_intact - partial_avscramble`; `modality_presence_diff = partial_intact - max(partial_clsav_from_a, partial_clsav_from_v)`; and the plain-RSA differences `searchlight_rho(intact) - searchlight_rho(control)`.

## Scripts

| Script | Purpose |
|--------|---------|
| `subcortical_io.py` | Structure catalogue, extraction, group-average accumulation, per-structure column slices, neighbor builders, nucleus masks, template CIFTI. `python subcortical/subcortical_io.py` runs a self-check. |
| `precompute_neighbors.py` | Build and cache all neighbor arrays (one subject suffices): `--raw-dir DIR --subject 132118 --k 100 --cache-dir DIR --workbench PATH`. |
| `subcortical_rsa.py` | Searchlight plus nuclei RSA, group average (`--subject group_average`, needs `--subjects-list`) or one subject (`--subject ID`, streaming preprocessing). |
| `subcortical_noise_ceiling.py` | Noise ceiling over `--subjects ID ...` (streaming, raw preprocessing), `--batch-size 256`. |
| `subcortical_partial_rsa.py` | Integration contrast for one `--run KEY` of `PARTIAL_RSA_RUNS` (flags as `subcortical_rsa.py`); the nuisance RDMs are removed by the ordinary-least-squares projection of `rsa/partial_rsa.py`. |
| `diff_maps.py` | Consolidates the contrasts above for the model roster at the top of the file (no flags; paths and the `k100_delay5s_bin5s_skip5s_spearman` configuration are constants). |
| `subcortical_visualization.py` | Builds the Workbench bundle. `--output-dir`, `--template`, `--bundle-dir`, `--wb-command`, `--download-atlases` (off by default). |
| `analysis.sh`, `run_diff_study.sh` | Runners (below). |

`subcortical_rsa.py` required flags: `--raw-dir`, `--timing-csv`, `--embeddings-dir`, `--template-cifti`, `--output-dir`, `--model`, `--modality`, `--bin-sec`; defaults: `--k 100`, `--delay-sec 5`, `--method spearman`, `--tr 1`, `--nucleus-radius 5`, `--gpu-batch-size 512`, `--workbench /opt/workbench/bin_linux64/wb_command`. Embeddings are read from `{embeddings-dir}/{model}/bin{N}s_skip{N}s/{model}_{modality}.npy`.

## Running

Environment: conda environment `movie`; `wb_command` (Workbench) and, for the cerebellar surface geodesic, SUITPy.

```bash
bash subcortical/analysis.sh MODE [N_BLOCKS]
```

| MODE | Action |
|------|--------|
| `neighbors` | Build the neighbor caches. Run first. |
| `groupavg` (default) | Group-average searchlight and nuclei RSA for every entry of `MODELS`, then rebuild the visualization bundle (unless `REFRESH_VISUALIZATION=false`). |
| `noiseceiling` | Noise ceiling on the subjects in `NC_SUBJECTS` (10 subject IDs by default; override via environment). |
| `persubject` | Per-subject RSA for every subject in `subjects.txt`, sequentially, for every `MODELS` entry. |
| `groupstats` | `rsa/group_stats.py --fname-prefix rsa_subcortical` across the per-subject maps. `N_BLOCKS` (default 16) sets the expected number of temporal block files; `subcortical_rsa.py` writes none, so the two-factor bootstrap is not run. |
| `visualize` | Rebuild the Workbench bundle. |

Configuration variables at the top of `analysis.sh`: paths (`DATA_BASE`, `OUTPUTS_BASE`, `CIFTI_DIR`, `SUBJECTS_LIST`, `TIMING_CSV`, `EMBEDDINGS_DIR`, `WORKBENCH`), `K=100`, `BIN_SEC=5.0`, `SKIP_SEC=5.0`, `DELAY_SEC=5.0`, `TR=1.0`, `METHOD=spearman`, and the model registry `MODELS` (entries `model:modality`): PE-AV with its three controls, plus (unless `SKIP_LAYER_SWEEP=true`) layers 35, 27, 18, 9, 1 of the omni3b and topoomni families and layers 9, 18, 27, 36 of nemotron, with `_mp` (mean-pooled) and `_lt` (last-token) variants where they exist, plus (unless `SKIP_SCRAMBLE_DUMMY_SWEEP=true`) the `_avscramble`, `_clsav_from_a`, `_clsav_from_v` controls of each native audiovisual model and pooling.

```bash
bash subcortical/run_diff_study.sh [plain|partial|consolidate|all]
```

`plain` runs `analysis.sh groupavg`; `partial` runs `subcortical_partial_rsa.py` for every model in `BASE_MODELS` and condition in `CONDITIONS` (intact, avscramble, clsav_from_a, clsav_from_v; restrict with `DIFF_STUDY_MODELS` and `DIFF_STUDY_CONDITIONS`, space-separated); `consolidate` runs `diff_maps.py` then `analysis.sh visualize`.

## Outputs

Root `outputs/subcortical/`:

| Path | Content |
|------|---------|
| `subcortical_template.dscalar.nii` | one-map CIFTI whose brain-model axis is the 19 concatenated structures; used as `--template-cifti` (built automatically) |
| `_neighbor_cache/` | `{STRUCTURE}_neighbors_k{k}_geodesic.npy` (`(n_voxels, k)` indices local to the structure); cerebellum surface results in `CEREBELLUM_{LEFT,RIGHT}/` |
| `group_average_cache/` | group-average dtseries and `run_trs.npy` |
| `group_average/{model}_{modality}/{config}/` | `rsa_subcortical_{tag}_{config}_searchlight.npy` (`rho` for all voxels), `..._maps.dscalar.nii` (maps `searchlight_rho`, `sigmap_uncorr`, `sigmap_fdr`, `fdr_mask`; sigmap = sign(rho) x -log10 p), `nuclei_roi_rsa_{tag}_{config}.csv` (columns `nucleus`, `n_vox`, `rho`, `p`, `parent`), `struct_summary.json` (per structure: voxels, mean and max rho, FDR-passing voxels, neighbor mode) |
| `subject_data/{subject}/{model}_{modality}/{config}/` | the same files for one subject |
| `group_average/{model}_{modality}_INTEGRATION/{config}/integration_partial_r_searchlight.dscalar.nii` | partial-RSA map named `partial_rsa_{model}_{modality}_INTEGRATION` |
| `group_average/{model}_av/dummy_partial_diff_maps.dscalar.nii` | `modality_presence_diff` |
| `group_average/{model}_avscramble_av/scramble_consolidated_maps.dscalar.nii` | `plain_rsa_scrambled`, `partial_rsa_scrambled`, `diff_intact_minus_scrambled_plain_rsa`, `binding` |
| `group_average/{model}_clsav_from_{a,v}_av/dummy_consolidated_maps.dscalar.nii` | `plain_rsa_dummy`, `diff_intact_minus_dummy_plain_rsa`, `partial_rsa_dummy` |
| `noise_ceiling/noise_ceiling_subcortical_{N}subs_{config}.dscalar.nii` | maps `nc_lower`, `nc_upper` |
| `groupstats/` | output of `rsa/group_stats.py` |
| `workbench_visualization/` | derived bundle (safe to delete and rebuild): `subcortical_wb_view.spec`, `manifest.json`, `README_wb_view.md`, meshes, and one `{category}__{model_modality}__{config}__{result}__{STRUCTURE}.func.gii` per result and anatomical unit (SUIT cerebellum, brain stem, one bilateral file per paired structure) |

`config` is `k{k}_delay{delay}s_bin{bin}s_skip{skip}s_{method}`; `tag` is the preprocessing tag (`raw`, `sg_psc`, ...). Outputs are skipped when they already exist.

To view: open the movie scene (`rsa_movie.scene` in the outputs root) and import `workbench_visualization/subcortical_wb_view.spec`. To add an analysis without rebuilding, place its exported `.func.gii` next to the others for that structure and load it manually; vertex counts are validated against the existing mesh.

## Tests

```bash
pytest tests/test_subcortical_visualization.py -v
```

Covers the anatomical grouping (each structure in exactly one unit), result discovery (subject results excluded), file-name encoding of the analysis identity, and the vertex-count self-check.
