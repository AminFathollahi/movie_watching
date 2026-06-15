# CF Modeling — Connective Field Modeling

Reimplementation of the connective field analysis from **Hedger et al. (2025)**
"Vicarious somatotopy: somatosensory and visual representations converge in human
cortex during naturalistic movie watching."

The pipeline fits banded ridge models predicting cortical fMRI from Laplace-Beltrami
Operator Eigenfunctions (LBOEs) of two Glasser ROI subsurfaces, then derives
audiovisual integration zone maps via null-corrected variance partitioning.

## Architecture

This pipeline imports vicsompy **directly from its source repository** without
`pip install` (required because torch 2.6+ / RTX 5070Ti is incompatible with
the vendored torch version in vicsompy's dependency list).  All core modeling
logic — `MssCf`, `Subsurface`, `generate_leave_one_run_out` — is used verbatim.

The main additions are:
- **Modular ROI definition**: any Glasser HCP-MMP1 ROI pair via CLI args.
- **LBOE cap**: ROIs with < 200 vertices automatically reduce n_lboe to
  min(200, n_L−2, n_R−2) (see `01_extract_geometry.py`).
- **CIFTI grayordinate targets**: model is evaluated on the full 59k cortex
  (not just within-ROI vertices); data-space conversion handled by
  `lib/data_adapter.py`.
- **No splicing**: `splice_lookups()` is not called (requires lookup-table CSVs
  that do not exist for custom ROI pairs).
- **GPU acceleration**: himalaya `torch_cuda` backend; controlled by `--backend`.
- **torch 2.11+ GPU patches** (applied automatically in `02_fit_cf_model.py`):
  - `torch_cuda.arange` monkey-patched to default `device="cuda"` — fixes a
    cross-device index crash in himalaya's random-search solver that only
    manifests with torch ≥ 2.x (Hedger et al. ran CPU-only torch 2.1.1).
  - `CfModel._offload_fitted_to_cpu()` moves dual weights / deltas / kernel
    matrices from GPU to CPU + calls `gc.collect()` + `empty_cache()` before
    `get_params()` / `test_xval()`, which would otherwise OOM trying to
    allocate ≈5 GB for the `(n_kernels × T × 59k)` split-prediction tensor.

## Attribution

This pipeline builds directly on code and methods from two open-source packages.
See [NOTICE](NOTICE) for full details.

| Component | Source | License |
|---|---|---|
| `vicsompy.modeling.MssCf` | [Hedger et al. (2025) / vicsompy](https://github.com/nicholashedger/vicsompy) | MIT |
| `vicsompy.surface.Subsurface` | [Hedger et al. (2025) / vicsompy](https://github.com/nicholashedger/vicsompy) | MIT |
| `vicsompy.utils.generate_leave_one_run_out` | Hedger et al. (2025) / vicsompy (credits [Gallant Lab](https://github.com/gallantlab/voxelwise_tutorials)) | MIT |

If you use this pipeline, please cite:

> Hedger, N. et al. (2025). Vicarious somatotopy: somatosensory and visual
> representations converge in human cortex during naturalistic movie watching.

## Environment

```bash
conda activate movie   # torch 2.11+cu128, himalaya 0.4.11
```

Do not `pip install vicsompy` — direct import via `sys.path` is intentional.

## Required Preprocessed fMRI

CF modeling reads the **concatenated-run** output of `preprocess_individual.py`
(SG high-pass → PSC → GSR applied **per run**, then concatenated).  Z-scoring
is applied **inside** `02_fit_cf_model.py`, matching Hedger et al. exactly.

### Disk mode (default, `STREAM=false`)

Preprocess once and save per-subject continuous CIFTIs:

```bash
conda activate movie

python preprocess_individual.py \
    --raw-dir /media/amin/Samsung_T5/HCP/Data/fMRI_CIFTI \
    --out-dir /path/to/outputs/preprocessed \
    --subjects-list /path/to/data/subjects.txt \
    --sg-filter --psc --gsr \
    --save-individual \
    --save-average
```

This produces, for each subject and the group average:
```
{sub}_sg_psc_cortex_59k.dtseries.nii        — (n_cortex, T_total) continuous CIFTI
{sub}_sg_psc_run_trs.npy                    — [T_run1, T_run2, T_run3, T_run4]
group_average_sg_psc_cortex_59k.dtseries.nii
group_average_sg_psc_run_trs.npy
```

File names are determined by `PREPROCESSING_FLAG` (auto-built from `SG_FILTER`/`PSC`/`GSR`
in `analysis.sh`).  With the current defaults (`SG_FILTER=true PSC=true GSR=false`)
the suffix is `sg_psc`.  Add `GSR=true` to get `sg_psc_gsr`.

> **Group-average CIFTI location**: `preprocess` mode saves the group-average file to
> `{PREPROCESSED_INDIV_DIR}/` then moves it to `{PREPROCESSED_DIR}/` (i.e.
> `data/preprocessed/average_sub/sg_psc/`).  If you have an older file sitting directly
> in `data/preprocessed/`, move it manually:
> ```bash
> mkdir -p data/preprocessed/average_sub/sg_psc
> mv data/preprocessed/group_average_sg_psc_*.{nii,npy} data/preprocessed/average_sub/sg_psc/
> ```

### Streaming mode (`STREAM=true`)

Set `STREAM=true` in `analysis.sh` to preprocess raw CIFTIs on-the-fly
inside `02_fit_cf_model.py`; only R²_nc maps are written to disk.

## Structure

```
cf_modeling/
├── NOTICE                         # third-party code attributions (read this)
├── README.md                      # this file
├── analysis.sh                # master runner — all paths and ROI pairs here
│
├── 00_make_roi_masks.py           # (optional) export CSV masks from Glasser dlabel
├── 01_extract_geometry.py         # build Subsurfaces + LBOEs (59k_fs_LR)
├── 02_fit_cf_model.py             # fit CF model; save R² + CIFTI maps
├── integration_maps.py            # integration_score, modality_balance, bimodal dlabel
├── overlap.py                     # RSA overlap (group_average) or group stats (per_subject)
│
├── lib/                           # reusable wrappers around vicsompy
│   ├── __init__.py
│   ├── cf_model.py                # CfModel(MssCf) — grayordinate targets, no splicing
│   ├── config_builder.py          # build_temp_yaml() for MssCf initialisation
│   ├── data_adapter.py            # grayord↔sphere-space conversion (BrainModelAxis)
│   └── subject_adapter.py         # minimal mock subject for MssCf
│
├── shared/
│   └── ridge_utils.py             # build_pipeline, project_onto_lboes, fit_null_r2
│                                  # generate_leave_one_run_out (from vicsompy.utils)
│
├── viz_cf_modeling.ipynb          # visualisation: CIFTI multimaps + flatmaps
└── environment.yml                # (reference only) conda env spec
```

> The `vendor/` directory is no longer imported — all vicsompy code is imported
> directly from `VICSOMPY_REPO` via `sys.path` injection. See `vendor/DEPRECATED.md`.

## Usage

```bash
conda activate movie
cd /home/amin/Research/Representation/Movie/movie_watching   # repo root

# Full pipeline (geometry → group-average → per-subject)
bash cf_modeling/analysis.sh all

# Individual stages
bash cf_modeling/analysis.sh masks        # optional CSV mask generation
bash cf_modeling/analysis.sh geometry     # build subsurfaces only
bash cf_modeling/analysis.sh avg          # group-average only
bash cf_modeling/analysis.sh persubject   # per-subject, sequential
bash cf_modeling/analysis.sh persubject 8 # per-subject, 8 parallel jobs
bash cf_modeling/analysis.sh persubject 8 100610  # resume from subject 100610
```

Add or remove ROI pairs by editing `PERSUBJECT_PAIRS` and `AVG_PAIRS` in
`analysis.sh`.

## Script-by-script overview

### `00_make_roi_masks.py` (optional)

Generates `{roi}_L_mask.csv` / `{roi}_R_mask.csv` from the Glasser
59k_fs_LR dlabel.nii.  Each CSV has a `mask` column with 59292 rows (bool).
`01_extract_geometry.py` reads the dlabel directly and does **not** require
these files — run `00_make_roi_masks.py` only if you need standalone masks
for external tools or forced-new vicsompy subsurfaces.

```bash
python cf_modeling/00_make_roi_masks.py \
    --glasser-dlabel /path/to/...59k_fs_LR.dlabel.nii \
    --rois 3b V1 A1 TA2 MST A5 FFC \
    --masks-dir /path/to/outputs/cf_modeling/masks
```

### `01_extract_geometry.py`

Builds `StableSubsurface` objects (Subsurface subclass with σ=1e-6 Laplacian
regularisation) and computes up to 200 LBOEs for two Glasser ROIs.

- **Surface**: pycortex `fiducial` (midthickness) from `hcp_999999_draw_NH`.
  Uses sphere LBOEs in the same mathematical sense as Hedger et al. (2025);
  the `fiducial` surface is used because `sphere` is not in our pycortex subject.
- **Parcellation**: 59k_fs_LR Glasser dlabel.nii (read directly; CSV masks not required).
- **LBOE cap**: `n_lboe = min(N_LBOE, n_L−2, n_R−2)` — handles small ROIs.
- **Caching**: saves `sub_{roi}.pkl` (our convention) and
  `{roi}_subsurface.pickle` (vicsompy convention) to the subsurfaces directory.
  Re-running uses the cached files.

```bash
python cf_modeling/01_extract_geometry.py \
    --mode group_average --roi-a 3b --roi-b V1 \
    --pycortex-store /path/to/data/hedger2026 \
    --glasser-dlabel /path/to/...59k_fs_LR.dlabel.nii \
    --output-base /path/to/outputs/cf_modeling \
    --vicsompy-repo /path/to/Vicarious_somatotopy
```

### `02_fit_cf_model.py`

Main modeling script.  One process per (subject, ROI pair).

**Disk mode** (default — requires pre-saved CIFTIs):
```bash
python cf_modeling/02_fit_cf_model.py \
    --mode group_average --roi-a 3b --roi-b V1 \
    --preprocessed-dir /path/to/preprocessed/average_sub/sg_psc \
    --fmri-suffix sg_psc \
    --template-cifti /path/to/preprocessed/average_sub/sg_psc/group_average_sg_psc_cortex_59k.dtseries.nii \
    --output-base /path/to/outputs/cf_modeling \
    --vicsompy-repo /path/to/Vicarious_somatotopy
```

**Streaming mode** (per_subject only):
```bash
python cf_modeling/02_fit_cf_model.py \
    --mode per_subject --roi-a A5 --roi-b FFC --subject 100610 \
    --raw-dir /path/to/raw_ciftis \
    --output-base /path/to/outputs/cf_modeling \
    --sg-filter --psc --gsr \
    --vicsompy-repo /path/to/Vicarious_somatotopy
```

Internal pipeline (matches vicsompy's `analyse_subject` exactly):

1. Load CIFTI (disk) or preprocess on-the-fly (streaming)
2. Split train/test per run with independent z-scoring (Hedger convention)
3. Convert grayordinate → sphere space (118584 verts, medial wall = 0)
4. Load cached Subsurface objects → inject into `CfModel`
5. Build temp YAML → instantiate `CfModel(MssCf)`
6. `make_dm_grayord()` — LBOE design matrix from sphere-space train data
7. `prep_pipeline()` — himalaya `MultipleKernelRidgeCV` with LORO-CV
8. `fit_grayord()` — fit on grayordinate targets (T_train × 59412)
9. `get_params()` — betas, train split R², best alphas
10. `test_xval_grayord()` — test R² (full + per-band split) on 59412 targets
11. `compute_null_r2()` — OLS null model from ROI mean timecourses
12. `save_all_maps()` — R²_nc npy + CIFTI multimap

**Skip logic**: if `R2_{ROI_A}_nc.npy` and `R2_{ROI_B}_nc.npy` already exist,
the run is skipped automatically.

### `viz_cf_modeling.ipynb`

Interactive visualisation notebook.  Edit `ROI_A`, `ROI_B`, `MODE` in the
CONFIG cell and run all.  Produces:

- **1D flatmaps** — all R² maps with individual colourbars (`vicsompy.vis.basic_plot`)
- **2D flatmap** — ROI_A (dim 1) × ROI_B (dim 2) dual-colorbar
  (`vicsompy.vis.Plot.uber_plot`)
- **2D integration flatmap** — modality balance × integration score
- **CIFTI multimap** — all maps in `all_maps_viz.dscalar.nii` for wb_view

## Data space

The HCP 59k_fs_LR sphere has **59292 vertices per hemisphere** (118584 bilateral),
but CIFTI stores only **59412 grayordinates** (medial wall excluded).

Subsurface vertex indices are in full-sphere space (L: 0..59291, R: 59292..118583).
`lib/data_adapter.py::grayord_to_sphere_space()` maps CIFTI 59412→sphere 118584,
filling medial-wall positions with zero.  This matches vicsompy's
`CiftiHandler.decompose_data()` exactly.

For pycortex visualisation, the template CIFTI uses **32k_fs_LR** (32492 verts/hem).
`viz_cf_modeling.ipynb` converts grayordinate → bilateral 32k vertex arrays using
the same BrainModelAxis approach.

## Output layout

```
outputs/cf_modeling/
├── masks/                         # optional CSV masks (00_make_roi_masks.py)
│   ├── {roi}_L_mask.csv
│   └── {roi}_R_mask.csv
├── group_average/
│   └── {ROI_A}_{ROI_B}/
│       ├── subsurfaces/           # sub_{roi}.pkl + {roi}_subsurface.pickle
│       ├── prep/                  # R²/R²_nc/null .npy + band_sizes.npy
│       ├── cifti_maps/            # all_maps.dscalar.nii + individual .dscalar.nii
│       └── results/               # rsa_overlap_*.json (overlap.py)
└── per_subject/
    └── {ROI_A}_{ROI_B}/
        ├── subsurfaces/           # sub_{roi}.pkl + {roi}_subsurface.pickle
        ├── subjects/
        │   └── {sub_id}/          # per-subject R²/R²_nc .npy + pipeline.log
        └── group/
            ├── cifti_maps/        # group avg + integration + stat CIFTIs
            └── stats/             # t/d/p/fdr .npy + stats_summary.json
```

## Viewing in Connectome Workbench

```bash
# Open the combined multimap in wb_view
wb_view \
    /path/to/group_average_sg_psc_gsr_cortex_59k.dtseries.nii \
    /path/to/outputs/cf_modeling/group_average/3b_V1/cifti_maps/all_maps.dscalar.nii

# Or individual maps:
wb_view \
    group_average_sg_psc_gsr_cortex_59k.dtseries.nii \
    cifti_maps/R2_V1_nc.dscalar.nii \
    cifti_maps/R2_3b_nc.dscalar.nii
```

Load `R2_V1_nc` (visual) as blue and `R2_3b_nc` (somatosensory) as red,
both transparent below 0, for a Figure 3a-style dual-colour display.

## Configuration reference (`analysis.sh`)

| Variable | Default | Description |
|---|---|---|
| `VICSOMPY_REPO` | `/path/to/Vicarious_somatotopy` | vicsompy source repo (direct import) |
| `CX_SUB` | `hcp_999999_draw_NH` | pycortex subject |
| `SURF_TYPE` | `fiducial` | pycortex surface type (midthickness) |
| `N_LBOE` | `200` | Max LBOEs per ROI (auto-capped for small ROIs) |
| `STREAM` | `false` | Streaming mode (preprocess raw CIFTIs on-the-fly) |
| `SG_FILTER` | `true` | Savitzky-Golay high-pass filter |
| `PSC` | `true` | Percent signal change (pre-SG mean used for normalisation) |
| `GSR` | `false` | Global signal regression (add `gsr` to suffix when enabled) |
| `BACKEND` | `torch_cuda` | himalaya backend (`torch_cuda` / `torch` / `numpy`) |
| `N_ITER` | `20` | Random-search iterations for α selection |
| `N_TARGETS_BATCH` | `20000` | Targets per GPU batch |
| `BATCH_SIZE` | `8` | Parallel jobs for per-subject pipeline |
| `PERSUBJECT_PAIRS` | `A5:FFC V1:3b TA2:MST` | ROI pairs for per-subject analysis |
| `AVG_PAIRS` | `3b:V1` | ROI pairs for group-average analysis |
