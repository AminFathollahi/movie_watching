# Subcortical RSA extension

Extends the cortex-only movie-watching RSA pipeline to the 62,053 subcortical
volume voxels present in the raw HCP 7T CIFTIs but discarded by
`preprocess_individual.py:extract_cortex()`. Reuses the cortical RSA/CIFTI
machinery wherever geometry-agnostic (`rsa/shared/rsa_utils.py`,
`rsa/searchlight.py::run_searchlight`, `cifti_io.py`); the only new logic is
per-structure column bookkeeping and neighbor-array construction
(`subcortical_io.py`).

## ROIs

19 anatomical structures (direct CIFTI structures, k=100 within-structure
searchlight each): cerebellum L/R, brain stem, thalamus, caudate, putamen,
pallidum, accumbens, amygdala, hippocampus, ventral diencephalon L/R.

8 sub-nuclei too small for a searchlight (whole-mask ROI-RSA instead): IC,
SC, MGN, LGN (L/R) — MNI coordinate spheres intersected with their parent
structure (brain stem for IC/SC, thalamus for MGN/LGN). Coordinates are the
Sitek et al. 2019 (*Front. Neurosci.*) 7T probabilistic-atlas peaks and
standard geniculate coordinates — no atlas download needed. Upgrade path:
swap the sphere builder in `subcortical_io.build_nucleus_masks` for a
Sitek/Bianciardi probabilistic-mask loader with the same interface
(`# ponytail:` note at the call site).

## Neighbor geometry

The subcortical voxel grid is byte-identical across subjects (standard MNI
subcortical parcellation), so neighbor arrays are computed **once** and
cached — unlike the per-subject cortical geodesic.

- All 19 structures: within-mask voxel-graph geodesic (26-connectivity,
  Dijkstra, mm-weighted). Euclidean is only the automatic fallback when a
  structure's voxel graph is disconnected (not observed for any structure
  here).
- Cerebellum L/R additionally get the **purist SUIT surface geodesic**: each
  cerebellar voxel is mapped to its nearest SUIT `PIAL_FSL` (MNI-space)
  vertex, and k-NN is computed by geodesic distance on the anatomically
  folded `PIAL_SUIT` mesh via `wb_command -surface-geodesic-distance-all-to-all`
  (the same wrapper the cortical pipeline uses). This is the headline
  cerebellar result; the voxel-graph geodesic is saved alongside for
  comparison. Falls back to voxel-graph geodesic (never Euclidean) if
  SUITPy/wb_command are unavailable.

Caches live under `outputs/subcortical/_neighbor_cache/`.

## Reliability caveat

Subcortical BOLD SNR is lower than cortex, and IC/SC are tiny nuclei.
**Every rho must be read as a fraction of its own noise ceiling**
(`rsa/noise_ceiling.py`, reused verbatim against the subcortical template),
not compared directly to cortical rho. A structure whose ceiling is at noise
is flagged, not over-interpreted.

## Scripts

| Script | Purpose | Output Directory |
|--------|---------|------------------|
| `subcortical_io.py` | Extraction, struct slices, neighbor building, nucleus mask construction, template generation, group-average accumulation | `outputs/rsa/group_average` |
| `precompute_neighbors.py` | One-time geodesic and surface-distance neighbor cache build for all structures | `outputs/rsa/_neighbor_cache` |
| `subcortical_rsa.py` | Searchlight and parcel-level (nuclei) RSA driver for subcortical structures; supports group-average and per-subject modes | `outputs/rsa/{prep}/group_average` or `per_subject` |
| `subcortical_noise_ceiling.py` | Inter-subject noise ceiling (NC_upper, NC_lower) for subcortical searchlight RSA | `outputs/rsa/noise_ceiling` |
| `subcortical_partial_rsa.py` | Partial correlation RSA controlling for individual subcortical structures | `outputs/rsa/{prep}/partial` |
| `diff_maps.py` | Differential subcortical RSA maps (e.g., AV-V, AV-A contrasts) | `outputs/rsa/{prep}/group_average` |
| `subcortical_visualization.py` | Mesh generation and Workbench overlay export; generates .func.gii files for each subcortical structure | `outputs/rsa/workbench_visualization` |

## Runners

`analysis.sh` modes:
- `bash analysis.sh neighbors` — One-time neighbor cache build
- `bash analysis.sh groupavg` — Group-average subcortical searchlight RSA
- `bash analysis.sh noiseceiling` — Inter-subject noise ceiling
- `bash analysis.sh persubject [N_JOBS] [RESUME_SUBJECT]` — Per-subject RSA (optional)
- `bash analysis.sh groupstats` — Group statistics on per-subject maps (optional)
- `bash analysis.sh visualize` — Generate Workbench visualization spec

## Run

```bash
conda activate movie
bash subcortical/analysis.sh neighbors      # one-time, ~4 min
bash subcortical/analysis.sh groupavg       # primary first pass
bash subcortical/analysis.sh noiseceiling   # mandatory reliability gate
bash subcortical/analysis.sh visualize      # rebuild complete wb_view spec
```

For viewing, open `outputs/rsa_movie.scene` and import
`outputs/subcortical/workbench_visualization/subcortical_wb_view.spec`. The
spec loads every anatomical mesh and all current group-average, groupstats,
and noise-ceiling overlays (partial-RSA/integration, scramble, and dummy
runs included — anything under `group_average/`, `groupstats/`, or
`noise_ceiling/`). To add a later analysis (e.g. a new omni3b run) without
rerunning `visualize`, drop its exported `.func.gii` next to the existing
ones for that structure and load it manually in wb_view — the naming
convention and vertex-count self-check guarantee it is compatible with the
already-loaded mesh. `workbench_visualization/` is fully derived from the
`.dscalar.nii` results and safe to delete and rebuild at any time.

Per-subject RSA (`group_stats.py` t-test across subjects) and
`subcortical_encoding.py` are the natural next steps but are **gated on the
group-average + noise-ceiling result being promising** — not run by default.
`rsa/group_stats.py` needs no code changes to serve the subcortical
template: pass `--template-cifti` pointing at `subcortical_template.dscalar.nii`,
omit `--workbench` (border-drawing is cortex-only and already gracefully
skipped when `--workbench` is absent — `--left-surface`/`--right-surface`
are still required by argparse but can be any placeholder path since they
go unread in that case).
