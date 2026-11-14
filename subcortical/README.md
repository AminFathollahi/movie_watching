# subcortical: encoding of subcortical voxels

Applies the cortical movie-watching analyses to the subcortical volume voxels of the 7T HCP CIFTI files (the part `preprocessing/preprocess_individual.py` discards when it extracts cortex). Binning, embedding alignment, the noise-ceiling kernels, the encoding variance partition and CIFTI input and output are imported from `shared/` and `encoding/`; the code here handles per-structure bookkeeping, neighbor construction and visualization.

## Method

Data. Raw subcortical dtseries of 19 structures (cerebellum, thalamus, caudate, putamen, pallidum, accumbens, amygdala, hippocampus and ventral diencephalon, each left and right, and the brain stem; 62,053 voxels), in the order of `subcortical_io.SUBCORTICAL_STRUCTURES`. Per run, optional Savitzky-Golay filtering, percent signal change and global signal regression (`--sg-filter`, `--psc`, `--gsr`; none by default, tag `raw`); runs are then concatenated. The group average is the running mean over the subjects in `--subjects-list` (175 subjects), cached as `group_average_{tag}_subcortical.dtseries.nii` and `group_average_{tag}_run_trs.npy`.

Segments. As in the cortical pipeline: windows of `--bin-sec` seconds every `--skip-sec` seconds (default `--bin-sec`), delayed by `--delay-sec`, within each run, averaged over time and z-scored within run (626 windows at 5 s). Model embeddings are cut into the same windows (hemodynamic convolution only with `--hrf`).

Neighbors. For voxel `v`, the neighborhood is `v` plus its `k - 1` nearest voxels (`--k`, default 100) inside the same structure. The subcortical grid is identical for all subjects, so neighbor arrays are computed once and cached in `outputs/subcortical/_neighbor_cache/`. Neighbors are the `k` nearest voxels by shortest path on the within-structure voxel graph (26-connectivity, edge weight in millimetres, Dijkstra); if a source voxel reaches fewer than `k` voxels, the remainder is filled by Euclidean distance. For the cerebellum, neighbors come from geodesic distance on the SUIT cerebellar surface (`PIAL_SUIT` mesh, each voxel mapped to its nearest `PIAL_FSL` vertex, distances from `wb_command -surface-geodesic-distance-all-to-all`), with the voxel-graph geodesic as fallback if SUITPy or `wb_command` fail. `subcortical_io.neighbor_columns` offsets each structure's neighbors by the structure's first column, so one searchlight pass over all 62,053 voxels equals the per-structure passes.

Encoding (`encoding/variance_partition.py --fmri-space subcortical`, run for every audio/video pair of `encoding/analysis.sh`). Voxelwise banded ridge regression of the binned responses on A, V, J and their four combinations, evaluated leave-one-run-out over the windows not in the second to fourth showings of the repeated clip (`--split loro`; the first showing, `video5`, is kept), embeddings z-scored over the same rows as the responses (`--feature-scaling zscore`, the primary scaling; `demean` is the robustness variant); R2 is the pooled held-out `1 - SSR/SST`, and the seven subset R2 values are partitioned by inclusion-exclusion, e.g. `unique_j = R2(AVJ) - R2(AV)` (see `encoding/README.md`).

Nuclei. Eight nuclei are too small for a searchlight: inferior colliculus (IC), superior colliculus (SC), medial geniculate (MGN) and lateral geniculate (LGN), left and right. Each is the set of voxels of its parent structure (brain stem for IC and SC; the thalamus of the same side for MGN and LGN) within 5 mm of an MNI seed coordinate (`NUCLEI_SEEDS` in `subcortical_io.py`); the nucleus value of a voxel map is its mean over the nucleus voxels.

Noise ceiling. With `RDM_i(v)` the correlation-distance neighborhood RDM of subject `i` at voxel `v` and `N` subjects, `NC_upper(v)` is the mean over `i` of the Spearman correlation between `RDM_i(v)` and the mean of all `N` subjects' RDMs, and `NC_lower(v)` replaces that mean by the mean over the other `N - 1` subjects. It bounds a subject-level rho, not the rho of the group average.

## Scripts

| Script | Purpose |
|---|---|
| `subcortical_io.py` | Structure catalogue, extraction, group-average accumulation, neighbor builders (`structure_neighbors`, `neighbor_columns`), nucleus masks, template CIFTI. `python subcortical/subcortical_io.py` runs a self-check |
| `precompute_neighbors.py` | Builds and caches all neighbor arrays (one subject suffices): `--raw-dir DIR --subject ID --k 100 --cache-dir DIR` |
| `subcortical_noise_ceiling.py` | Noise ceiling over `--subjects ID ...` (streaming, raw preprocessing) |
| `subcortical_visualization.py` | Builds the Workbench bundle (`--output-dir`, `--template`, `--bundle-dir`, `--wb-command`, `--download-atlases`) |
| `analysis.sh` | Runner (below) |

## Usage

Requires Connectome Workbench (`wb_command`) and, for the cerebellar surface geodesic, SUITPy. Paths and parameters are set at the top of `analysis.sh` (`K=100`, `BIN_SEC=5.0`, `SKIP_SEC=5.0`, `DELAY_SEC=5.0`, `TR=1.0`, `METHOD=spearman`, `NC_SUBJECTS`). The encoding of the subcortical voxels is run by `encoding/analysis.sh` (`SPACES` includes `subcortex`).

```bash
bash subcortical/analysis.sh MODE
```

| `MODE` | Action |
|---|---|
| `neighbors` (default) | Build the neighbor caches |
| `noiseceiling` | Noise ceiling on the subjects in `NC_SUBJECTS` (10 IDs by default; override by environment variable) |
| `visualize` | Rebuild the Workbench bundle |

## Outputs

`config` is `k{k}_delay{delay}s_bin{bin}s_skip{skip}s_pearson_{method}` for the noise ceiling. Existing outputs are skipped. Under `outputs/subcortical/`:

| Path | Content |
|---|---|
| `subcortical_template.dscalar.nii` | One-map CIFTI whose brain-model axis is the 19 concatenated structures; used as `--template-cifti` (built automatically) |
| `_neighbor_cache/` | `{STRUCTURE}_neighbors_k{k}_geodesic.npy` (`(n_voxels, k)` indices local to the structure); cerebellum surface results in `CEREBELLUM_{LEFT,RIGHT}/` |
| `group_average_cache/` | Group-average dtseries and `run_trs.npy` |
| `encoding/group_average/partitions/delay{d}s_bin{b}s_skip{s}s/` | Encoding of PE-AV and Nemotron layer 18 with the joint model in the names: fits `encoding_{r2,pearson_r}_loro_{scaling}_{J}_{tag}_models.dscalar.nii` (`r2_a` ... `r2_avj`, `r_a` ... `r_avj`), `..._per_fold.npz`, `..._provenance.json`, with `--save-predictions` the held-out predictions; and, built by the collect step, one file per quantity with the pairs as maps (`unique_j_{J}_{scaling}.dscalar.nii`, `shared_av_{scaling}.dscalar.nii`, `gain_*`, `own_*`, `own_vs_controls_{J}_{scaling}.dscalar.nii`) and the colour maps `av_bivariate_{r2,gain}_{scaling}.dlabel.nii`. Other encoders keep `encoding/group_average/{model}/delay{d}s_bin{b}s_skip{s}s/`. Details in `encoding/README.md` (Collected files) |
| `encoding/group_average/time_resolved/delay{d}s_bin{b}s_skip{s}s/` | Time-resolved voxel series (`.dtseries.nii`), `time_resolved_{scaling}.dscalar.nii` and `av_bivariate_seconds_{scaling}.dlabel.nii` for all pairs (`encoding/README.md`, Time-resolved files) |
| `noise_ceiling/noise_ceiling_subcortical_{N}subs_{config}.dscalar.nii` | Maps `nc_lower`, `nc_upper` |
| `workbench_visualization/` | Derived bundle, safe to delete and rebuild: `subcortical_wb_view.spec`, `manifest.json`, `README_wb_view.md`, meshes, and one `{category}__{model_modality}__{config}__{result}__{STRUCTURE}.func.gii` per result and anatomical unit (SUIT cerebellum, brain stem, one bilateral file per paired structure) |

To view, open the movie scene (`outputs/rsa_movie.scene`) in `wb_view` and import `workbench_visualization/subcortical_wb_view.spec`. To add an analysis without rebuilding, place its exported `.func.gii` next to the others for that structure and load it manually; vertex counts are validated against the existing mesh.

## Current results (October 2026)

Group average of 175 subjects, 626 windows of 5 s (stride 5 s, delay 5 s), k = 100, A and V from the model itself. Each cell is the mean of the map over the voxels of the structure; for the nuclei (5-mm spheres, 4 to 116 voxels) it is the mean of the voxel map over the nucleus voxels. Columns: held-out encoding R² (leave-one-run-out with the first showing of the repeated clip kept in training and its later showings excluded, demeaned embeddings) and `unique J` = R²(AVJ) − R²(AV).

PE-AV

| Structure | voxels | R²(A) | R²(V) | R²(J) | R²(AVJ) | unique J |
|---|---:|---:|---:|---:|---:|---:|
| Cerebellum L | 17,105 | +0.0087 | +0.0224 | +0.0260 | +0.0227 | -0.0012 |
| Cerebellum R | 17,638 | +0.0133 | +0.0147 | +0.0281 | +0.0220 | +0.0022 |
| Thalamus L | 2,515 | +0.0038 | +0.0120 | +0.0153 | +0.0122 | -0.0003 |
| Thalamus R | 2,461 | +0.0035 | +0.0169 | +0.0138 | +0.0130 | -0.0021 |
| LGN L | 4 | +0.0003 | -0.0095 | -0.0010 | -0.0115 | -0.0042 |
| LGN R | 6 | -0.0003 | +0.0023 | +0.0109 | +0.0064 | +0.0060 |
| MGN L | 70 | +0.0011 | +0.0002 | +0.0064 | +0.0018 | +0.0004 |
| MGN R | 70 | +0.0122 | +0.0150 | +0.0208 | +0.0199 | -0.0003 |
| IC L | 108 | +0.0165 | +0.0025 | +0.0157 | +0.0198 | +0.0009 |
| IC R | 116 | +0.0162 | +0.0113 | +0.0129 | +0.0233 | -0.0006 |
| SC L | 101 | +0.0212 | +0.0397 | +0.0420 | +0.0404 | -0.0001 |
| SC R | 102 | +0.0389 | +0.0602 | +0.0568 | +0.0658 | -0.0014 |
| Hippocampus L | 1,484 | -0.0030 | +0.0090 | +0.0069 | +0.0038 | -0.0011 |
| Hippocampus R | 1,512 | -0.0022 | +0.0181 | +0.0127 | +0.0129 | -0.0011 |
| Amygdala L | 620 | +0.0101 | +0.0533 | +0.0446 | +0.0520 | -0.0007 |
| Amygdala R | 654 | +0.0091 | +0.0757 | +0.0586 | +0.0731 | -0.0010 |
| Caudate L | 1,370 | +0.0022 | -0.0008 | +0.0043 | +0.0012 | -0.0000 |
| Caudate R | 1,431 | -0.0018 | +0.0004 | +0.0003 | -0.0027 | -0.0014 |
| Putamen L | 2,114 | +0.0011 | -0.0009 | +0.0018 | -0.0012 | -0.0005 |
| Putamen R | 2,016 | -0.0007 | -0.0013 | +0.0005 | -0.0030 | -0.0005 |
| Brain stem | 6,731 | -0.0002 | +0.0013 | +0.0017 | -0.0013 | -0.0009 |
| All 62,053 voxels | 62,053 | +0.0066 | +0.0138 | +0.0186 | +0.0151 | -0.0001 |

Nemotron layer 18 (mean-pooled)

| Structure | voxels | R²(A) | R²(V) | R²(J) | R²(AVJ) | unique J |
|---|---:|---:|---:|---:|---:|---:|
| Cerebellum L | 17,105 | +0.0098 | +0.0260 | +0.0366 | +0.0319 | +0.0006 |
| Cerebellum R | 17,638 | +0.0124 | +0.0184 | +0.0339 | +0.0257 | +0.0012 |
| Thalamus L | 2,515 | +0.0042 | +0.0124 | +0.0187 | +0.0138 | +0.0008 |
| Thalamus R | 2,461 | +0.0043 | +0.0171 | +0.0218 | +0.0184 | -0.0001 |
| LGN L | 4 | -0.0022 | -0.0140 | -0.0154 | -0.0181 | -0.0007 |
| LGN R | 6 | -0.0016 | -0.0053 | +0.0037 | -0.0034 | +0.0035 |
| MGN L | 70 | +0.0033 | +0.0008 | +0.0050 | +0.0035 | -0.0003 |
| MGN R | 70 | +0.0122 | +0.0146 | +0.0238 | +0.0196 | -0.0011 |
| IC L | 108 | +0.0233 | +0.0098 | +0.0225 | +0.0291 | -0.0010 |
| IC R | 116 | +0.0186 | +0.0080 | +0.0173 | +0.0214 | -0.0004 |
| SC L | 101 | +0.0030 | +0.0583 | +0.0651 | +0.0645 | +0.0005 |
| SC R | 102 | +0.0201 | +0.0601 | +0.0773 | +0.0781 | +0.0005 |
| Hippocampus L | 1,484 | -0.0050 | +0.0209 | +0.0245 | +0.0173 | +0.0033 |
| Hippocampus R | 1,512 | -0.0059 | +0.0302 | +0.0334 | +0.0275 | +0.0028 |
| Amygdala L | 620 | +0.0155 | +0.0485 | +0.0547 | +0.0488 | +0.0012 |
| Amygdala R | 654 | +0.0191 | +0.0692 | +0.0744 | +0.0686 | +0.0001 |
| Caudate L | 1,370 | +0.0008 | -0.0018 | +0.0021 | -0.0009 | +0.0001 |
| Caudate R | 1,431 | -0.0016 | -0.0018 | -0.0003 | -0.0032 | +0.0000 |
| Putamen L | 2,114 | -0.0006 | -0.0012 | +0.0007 | -0.0027 | -0.0002 |
| Putamen R | 2,016 | -0.0006 | -0.0022 | +0.0002 | -0.0034 | -0.0000 |
| Brain stem | 6,731 | -0.0003 | +0.0016 | +0.0037 | +0.0006 | -0.0002 |
| All 62,053 voxels | 62,053 | +0.0066 | +0.0163 | +0.0251 | +0.0198 | +0.0006 |

Joint beyond audio and video. Encoding unique J is −0.0004 over all voxels for PE-AV and +0.0005 for Nemotron (hippocampus +0.003).

Where the signal is. Encoding R² is highest in amygdala (R²(AVJ) 0.049 to 0.073), the superior colliculus spheres (0.040 to 0.078) and cerebellum (0.022 to 0.032), carried mostly by V; it is within 0.002 of zero or below in caudate, putamen, pallidum, accumbens and the brain stem as a whole.

Noise ceiling. The ceiling on disk (`noise_ceiling_subcortical_10subs_..._pearson_spearman`) is for single-subject correlation-distance searchlights from 10 subjects: nc_lower ranges from 0.0034 (accumbens L, pallidum L) to 0.0112 (cerebellum L), 0.0057 / 0.0061 in thalamus L / R, 0.0066 / 0.0068 in hippocampus, 0.0059 / 0.0060 in amygdala, 0.0062 in the brain stem and 0.0064 to 0.0079 in the nuclei; nc_upper is 0.300 to 0.308 everywhere. It bounds subject-level rho, not the encoding R² above, and there is no encoding ceiling for subcortex yet.

## Tests

```bash
pytest tests/test_subcortical_visualization.py tests/test_subcortical_neighbors.py -v
```

`test_subcortical_neighbors.py` checks that `neighbor_columns` keeps every neighborhood inside its structure.
