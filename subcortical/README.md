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

## Files

- `subcortical_io.py` — extraction, `struct_slices`, `build_neighbors`,
  `build_cerebellum_surface_neighbors`, `build_nucleus_masks`,
  `make_subcortical_template`, group-average accumulator. Self-check:
  `python subcortical_io.py`.
- `precompute_neighbors.py` — one-time neighbor cache build (all structures).
- `subcortical_rsa.py` — searchlight + nuclei ROI-RSA driver. Supports
  `--subject group_average` (primary pass) and `--subject <ID>` (per-subject,
  gated on the group-average result).
- `subcortical_noise_ceiling.py` — streaming-mode inter-subject noise
  ceiling on a subject subset (reuses `rsa/noise_ceiling.py`'s GPU/CPU
  kernels verbatim).
- `analysis.sh` — `neighbors | groupavg | noiseceiling` modes.

## Run

```bash
conda activate movie
bash subcortical/analysis.sh neighbors      # one-time, ~4 min
bash subcortical/analysis.sh groupavg       # primary first pass
bash subcortical/analysis.sh noiseceiling   # mandatory reliability gate
```

Per-subject RSA (`group_stats.py` t-test across subjects) and
`subcortical_encoding.py` are the natural next steps but are **gated on the
group-average + noise-ceiling result being promising** — not run by default.
`rsa/group_stats.py` needs no code changes to serve the subcortical
template: pass `--template-cifti` pointing at `subcortical_template.dscalar.nii`,
omit `--workbench` (border-drawing is cortex-only and already gracefully
skipped when `--workbench` is absent — `--left-surface`/`--right-surface`
are still required by argparse but can be any placeholder path since they
go unread in that case).
