# subcortical: RSA of subcortical voxels

Applies the cortical movie-watching RSA to the subcortical volume voxels of the 7T HCP CIFTI files (the part `preprocess_individual.py` discards when it extracts cortex). Binning, embedding alignment, the searchlight kernel and CIFTI input and output are reused from `rsa/shared/rsa_utils.py`, `rsa/searchlight.py` and `cifti_io.py`; the code here handles per-structure bookkeeping, neighbor construction and visualization. `RESULTS_group_average.md` lists the results of one group-average run.

## Method

Data. Raw subcortical dtseries of 19 structures (cerebellum, thalamus, caudate, putamen, pallidum, accumbens, amygdala, hippocampus and ventral diencephalon, each left and right, and the brain stem), in the order of `subcortical_io.SUBCORTICAL_STRUCTURES`. Per run, optional Savitzky-Golay filtering, percent signal change and global signal regression (`--sg-filter`, `--psc`, `--gsr`; none by default, tag `raw`); runs are then concatenated. The group average is the running mean over the subjects in `--subjects-list`, cached as `group_average_{tag}_subcortical.dtseries.nii` and `group_average_{tag}_run_trs.npy`.

Segments. As in the cortical pipeline: windows of `--bin-sec` seconds every `--skip-sec` seconds (default `--bin-sec`), delayed by `--delay-sec`, within each run, averaged over time and z-scored within run. Model embeddings are cut into the same windows (hemodynamic convolution only with `--hrf`).

Searchlight RSA, per structure. For voxel `v`, the neighborhood is `v` plus its `k - 1` nearest voxels (`--k`, default 100) inside the same structure. The brain representational dissimilarity matrix (RDM) of the neighborhood is `1 - Pearson correlation` between segments of the neighborhood's voxel patterns, and the model RDM is the same quantity for the embedding. `rho(v)` is the Spearman (`--method spearman`, default) or Pearson correlation between the lower triangles of the two RDMs. One-tailed uncorrected p-values come from `t = rho * sqrt(df) / sqrt(1 - rho^2)` with `df = n_segments - 2`, followed by Benjamini-Hochberg false-discovery-rate correction at 0.05 across all voxels.

Neighbors. The subcortical grid is identical for all subjects, so neighbor arrays are computed once and cached in `outputs/subcortical/_neighbor_cache/`. Neighbors are the `k` nearest voxels by shortest path on the within-structure voxel graph (26-connectivity, edge weight in millimetres, Dijkstra); if a source voxel reaches fewer than `k` voxels, the remainder is filled by Euclidean distance. For the cerebellum, neighbors come from geodesic distance on the SUIT cerebellar surface (`PIAL_SUIT` mesh, each voxel mapped to its nearest `PIAL_FSL` vertex, distances from `wb_command -surface-geodesic-distance-all-to-all`), with the voxel-graph geodesic as fallback if SUITPy or `wb_command` fail.

Nuclei. Eight nuclei are too small for a searchlight: inferior colliculus (IC), superior colliculus (SC), medial geniculate (MGN) and lateral geniculate (LGN), left and right. Each is the set of voxels of its parent structure (brain stem for IC and SC; the thalamus of the same side for MGN and LGN) within `--nucleus-radius` mm (default 5) of an MNI seed coordinate (`NUCLEI_SEEDS` in `subcortical_io.py`). One RDM over all voxels of the nucleus is correlated with the model RDM (`rho` and `p` in the CSV).

Noise ceiling. With `RDM_i(v)` the neighborhood RDM of subject `i` at voxel `v` and `N` subjects, `NC_upper(v)` is the mean over `i` of the Spearman correlation between `RDM_i(v)` and the mean of all `N` subjects' RDMs, and `NC_lower(v)` replaces that mean by the mean over the other `N - 1` subjects. A model's group-average `rho` is read relative to its own ceiling, because subcortical signal-to-noise differs from cortex.

Integration contrast (partial RSA). For an audiovisual embedding, the partial correlation between the brain RDM and the target RDM after regressing out the nuisance RDMs configured for the run in `rsa/shared/model_registry.py::PARTIAL_RSA_RUNS` (the model's own unimodal audio and video embeddings for intact runs; the single real modality for the `clsav_from_*` controls). Controls are `avscramble` (real audio with temporally permuted video), `clsav_from_a` (audio with a fixed placeholder video) and `clsav_from_v` (video with a placeholder audio). `diff_maps.py` forms per model `binding = partial_intact - partial_avscramble`, `modality_presence_diff = partial_intact - max(partial_clsav_from_a, partial_clsav_from_v)`, and the plain-RSA differences `searchlight_rho(intact) - searchlight_rho(control)`.

## Scripts

| Script | Purpose |
|---|---|
| `subcortical_io.py` | Structure catalogue, extraction, group-average accumulation, neighbor builders, nucleus masks, template CIFTI. `python subcortical/subcortical_io.py` runs a self-check |
| `precompute_neighbors.py` | Builds and caches all neighbor arrays (one subject suffices): `--raw-dir DIR --subject ID --k 100 --cache-dir DIR` |
| `subcortical_rsa.py` | Searchlight and nuclei RSA, for the group average (`--subject group_average`, needs `--subjects-list`) or one subject (`--subject ID`, streaming preprocessing) |
| `subcortical_noise_ceiling.py` | Noise ceiling over `--subjects ID ...` (streaming, raw preprocessing) |
| `subcortical_partial_rsa.py` | Integration contrast for one `--run KEY` of `PARTIAL_RSA_RUNS`; flags as `subcortical_rsa.py` |
| `diff_maps.py` | Consolidates the contrasts above for the model roster at the top of the file (no flags) |
| `subcortical_visualization.py` | Builds the Workbench bundle (`--output-dir`, `--template`, `--bundle-dir`, `--wb-command`, `--download-atlases`) |
| `analysis.sh`, `run_diff_study.sh` | Runners (below) |

`subcortical_rsa.py` requires `--raw-dir`, `--timing-csv`, `--embeddings-dir`, `--template-cifti`, `--output-dir`, `--model`, `--modality` and `--bin-sec`. Defaults: `--k 100`, `--delay-sec 5`, `--method spearman`, `--tr 1`, `--nucleus-radius 5`, `--gpu-batch-size 512`. Embeddings are read from `{embeddings-dir}/{model}/bin{N}s_skip{N}s/{model}_{modality}.npy`.

## Usage

Requires Connectome Workbench (`wb_command`) and, for the cerebellar surface geodesic, SUITPy. Paths and parameters are set at the top of `analysis.sh` (`K=100`, `BIN_SEC=5.0`, `SKIP_SEC=5.0`, `DELAY_SEC=5.0`, `TR=1.0`, `METHOD=spearman`, and the `MODELS` list of `model:modality` entries).

```bash
bash subcortical/analysis.sh MODE [N_BLOCKS]
bash subcortical/run_diff_study.sh [plain|partial|consolidate|all]
```

| `MODE` | Action |
|---|---|
| `neighbors` | Build the neighbor caches. Run first |
| `groupavg` (default) | Group-average searchlight and nuclei RSA for every `MODELS` entry, then rebuild the visualization bundle (unless `REFRESH_VISUALIZATION=false`) |
| `noiseceiling` | Noise ceiling on the subjects in `NC_SUBJECTS` (10 IDs by default; override by environment variable) |
| `persubject` | Per-subject RSA for every subject in `subjects.txt` and every `MODELS` entry |
| `groupstats` | `rsa/group_stats.py --fname-prefix rsa_subcortical` across the per-subject maps. `subcortical_rsa.py` writes no temporal block files, so the two-factor bootstrap is not run |
| `visualize` | Rebuild the Workbench bundle |

`MODELS` holds PE-AV with its three controls and, unless `SKIP_LAYER_SWEEP=true`, the layer sweeps of the omni3b, topoomni and nemotron families (`_mp` and `_lt` variants where they exist) and, unless `SKIP_SCRAMBLE_DUMMY_SWEEP=true`, the `_avscramble`, `_clsav_from_a` and `_clsav_from_v` controls of each native audiovisual model. In `run_diff_study.sh`, `plain` runs `analysis.sh groupavg`; `partial` runs `subcortical_partial_rsa.py` for every model in `BASE_MODELS` and condition in `CONDITIONS` (`intact avscramble clsav_from_a clsav_from_v`; restrict with the space-separated `DIFF_STUDY_MODELS` and `DIFF_STUDY_CONDITIONS`); `consolidate` runs `diff_maps.py`, then `analysis.sh visualize`.

## Outputs

`config` is `k{k}_delay{delay}s_bin{bin}s_skip{skip}s_{method}` and `tag` is the preprocessing tag (`raw`, `sg_psc`, ...). Existing outputs are skipped. Under `outputs/subcortical/`:

| Path | Content |
|---|---|
| `subcortical_template.dscalar.nii` | One-map CIFTI whose brain-model axis is the 19 concatenated structures; used as `--template-cifti` (built automatically) |
| `_neighbor_cache/` | `{STRUCTURE}_neighbors_k{k}_geodesic.npy` (`(n_voxels, k)` indices local to the structure); cerebellum surface results in `CEREBELLUM_{LEFT,RIGHT}/` |
| `group_average_cache/` | Group-average dtseries and `run_trs.npy` |
| `group_average/{model}_{modality}/{config}/` | `rsa_subcortical_{tag}_{config}_searchlight.npy` (`rho` for all voxels); `..._maps.dscalar.nii` (maps `searchlight_rho`, `sigmap_uncorr`, `sigmap_fdr`, `fdr_mask`; sigmap = sign(rho) x -log10 p); `nuclei_roi_rsa_{tag}_{config}.csv` (columns `nucleus`, `n_vox`, `rho`, `p`, `parent`); `struct_summary.json` (per structure: voxels, mean and max rho, FDR-passing voxels, neighbor mode) |
| `subject_data/{subject}/{model}_{modality}/{config}/` | The same files for one subject |
| `group_average/{model}_{modality}_INTEGRATION/{config}/integration_partial_r_searchlight.dscalar.nii` | Partial-RSA map named `partial_rsa_{model}_{modality}_INTEGRATION` |
| `group_average/{model}_av/dummy_partial_diff_maps.dscalar.nii` | `modality_presence_diff` |
| `group_average/{model}_avscramble_av/scramble_consolidated_maps.dscalar.nii` | `plain_rsa_scrambled`, `partial_rsa_scrambled`, `diff_intact_minus_scrambled_plain_rsa`, `binding` |
| `group_average/{model}_clsav_from_{a,v}_av/dummy_consolidated_maps.dscalar.nii` | `plain_rsa_dummy`, `diff_intact_minus_dummy_plain_rsa`, `partial_rsa_dummy` |
| `noise_ceiling/noise_ceiling_subcortical_{N}subs_{config}.dscalar.nii` | Maps `nc_lower`, `nc_upper` |
| `groupstats/` | Output of `rsa/group_stats.py` |
| `workbench_visualization/` | Derived bundle, safe to delete and rebuild: `subcortical_wb_view.spec`, `manifest.json`, `README_wb_view.md`, meshes, and one `{category}__{model_modality}__{config}__{result}__{STRUCTURE}.func.gii` per result and anatomical unit (SUIT cerebellum, brain stem, one bilateral file per paired structure) |

To view, open the movie scene (`outputs/rsa_movie.scene`) in `wb_view` and import `workbench_visualization/subcortical_wb_view.spec`. To add an analysis without rebuilding, place its exported `.func.gii` next to the others for that structure and load it manually; vertex counts are validated against the existing mesh.

## Tests

```bash
pytest tests/test_subcortical_visualization.py -v
```
