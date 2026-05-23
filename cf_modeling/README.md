# CF Modeling — Connective Field Modeling

Reimplementation of the connective field analysis from **Hedger et al. (2025)**
"Vicarious somatotopy: somatosensory and visual representations converge in human
cortex during naturalistic movie watching."

The pipeline fits banded ridge models predicting cortical fMRI from Laplace-Beltrami
Operator Eigenfunctions (LBOEs) of two Glasser ROI subsurfaces, then derives
audiovisual integration zone maps via null-corrected variance partitioning.

## Attribution

This pipeline builds directly on code and methods from two open-source packages.
See [NOTICE](NOTICE) for full details.

| Component | Source | License |
|---|---|---|
| `vendor/hedger_cf/subsurface.py` — `Subsurface` class | [Hedger et al. (2025) / vicsompy](https://github.com/nicholashedger/vicsompy) | MIT |
| `vendor/hedger_cf/utils.py` — `generate_leave_one_run_out()` | [Hedger et al. (2025) / vicsompy](https://github.com/nicholashedger/vicsompy) (credits [Gallant Lab / voxelwise_tutorials](https://github.com/gallantlab/voxelwise_tutorials)) | MIT |

`Subsurface59k` in `extract_geometry.py` extends `Subsurface` to load 59k_fs_LR
sphere GIFTIs and apply a regularisation shift to the Laplacian eigendecomposition.
`generate_leave_one_run_out` in `vendor/hedger_cf/utils.py` is taken verbatim from
vicsompy, which itself credits the Gallant Lab (documented in the function's docstring).
The CF modeling logic (LBOEs, banded ridge, null-model correction) is unchanged
from Hedger et al. (2025).

If you use this pipeline, please cite:

> Hedger, N. et al. (2025). Vicarious somatotopy: somatosensory and visual
> representations converge in human cortex during naturalistic movie watching.

## Required Preprocessed fMRI

cf_modeling reads the **full-run continuous** output of `preprocess_individual.py`
(SG high-pass → PSC → GSR per run, concatenated). No timing filtering is applied
during preprocessing; the train/test split (last `--n-test-trs` TRs per run = test)
is done inside `run_cfmodeling.py`, following Hedger et al.

### Disk mode (default, `STREAM=false`)

Preprocess once and save per-subject continuous CIFTIs:

```bash
conda activate analysis

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
{sub}_sg_psc_gsr_cortex_59k.dtseries.nii   — (n_cortex, T_total) continuous CIFTI
{sub}_sg_psc_gsr_run_trs.npy               — [T_run1, T_run2, T_run3, T_run4]
group_average_sg_psc_gsr_cortex_59k.dtseries.nii
group_average_sg_psc_gsr_run_trs.npy
```

Set `FMRI_SUFFIX="sg_psc_gsr"` in `run_analysis.sh` (this is the default).

### Streaming mode (`STREAM=true`)

Set `STREAM=true` in `run_analysis.sh` to skip saving preprocessed CIFTIs. Raw
data is preprocessed on-the-fly inside `run_cfmodeling.py`; only the R²_nc maps
are written to disk. Controlled by `SG_FILTER`, `PSC`, `GSR` flags in `run_analysis.sh`.

## Structure

```
cf_modeling/
├── NOTICE                    # third-party code attributions (read this)
├── run_analysis.sh           # master runner — all paths and ROI pairs here
├── extract_geometry.py       # build Subsurfaces + LBOEs (59k sphere)
├── run_cfmodeling.py         # phases 02→04: load data, fit, null-correct
│                             #   disk mode:      --preprocessed-dir + --fmri-suffix
│                             #   streaming mode: --raw-dir (per_subject only)
├── integration_maps.py       # integration_score, modality_balance, bimodal dlabel
├── summary.py                # RSA overlap (group_average) or group stats (per_subject)
├── vendor/
│   └── hedger_cf/
│       ├── subsurface.py     # Subsurface class — verbatim from vicsompy (MIT)
│       ├── utils.py          # generate_leave_one_run_out — verbatim from vicsompy (MIT)
│       ├── ATTRIBUTION.md    # detailed attribution for vendored code
│       └── LICENSE           # MIT License (Hedger / vicsompy)
└── shared/
    └── ridge_utils.py        # build_pipeline, project_onto_lboes, fit_null_r2,
                              # build_sphere_to_grayord_lut
```

## Usage

```bash
conda activate cfmod
cd movie_watching   # run from repo root

bash cf_modeling/run_analysis.sh all           # group-average + per-subject
bash cf_modeling/run_analysis.sh groupaverage  # group-average only
bash cf_modeling/run_analysis.sh persubject    # per-subject only
bash cf_modeling/run_analysis.sh persubject 4  # per-subject, 4 parallel jobs
bash cf_modeling/run_analysis.sh persubject 8 100610  # resume from subject 100610
```

Add or remove ROI pairs by editing `run_all_persubject()` and `run_all_groupaverage()`
in `run_analysis.sh`.

## Script-by-script overview

### `extract_geometry.py`

Builds `Subsurface59k` objects (extending Hedger et al.'s `Subsurface`) and
computes up to 200 LBOEs for two Glasser ROIs. Subsurfaces are cached as `.pkl`
files — re-running skips this step automatically.

**Surface**: 59k_fs_LR sphere (`S1200.{L,R}.sphere.59k_fs_LR.surf.gii`),
following the Hedger et al. (2025) convention.
**Parcellation**: 59k_fs_LR Glasser dlabel.nii for ROI vertex masks.

```bash
python cf_modeling/extract_geometry.py --mode group_average --roi_a A1 --roi_b V1
python cf_modeling/extract_geometry.py --mode per_subject   --roi_a A5 --roi_b FFC
```

### `run_cfmodeling.py` (phases 02→04)

Unified entry point for data preparation, banded ridge fitting, and null
correction. One process per (subject, ROI pair).

**Disk mode** (default — requires pre-saved CIFTIs):
```bash
python cf_modeling/run_cfmodeling.py \
    --mode group_average --roi-a A1 --roi-b V1 \
    --preprocessed-dir /path/to/preprocessed \
    --fmri-suffix sg_psc_gsr \
    --output-base /path/to/outputs/cf_modeling \
    --template-cifti /path/to/group_average_sg_psc_gsr_cortex_59k.dtseries.nii
```

**Streaming mode** (per_subject only — no CIFTI saved):
```bash
python cf_modeling/run_cfmodeling.py \
    --mode per_subject --roi-a A5 --roi-b FFC --subject 100610 \
    --raw-dir /path/to/raw_ciftis \
    --output-base /path/to/outputs/cf_modeling \
    [--sg-filter] [--psc] [--no-gsr]
```

What it does internally:
1. Load CIFTI (disk) or preprocess raw CIFTI on-the-fly (streaming)
2. Build sphere→grayordinate LUT from the BrainModelAxis (maps full 59k sphere
   vertex indices to CIFTI grayordinate rows; critical for correct ROI extraction)
3. Split train/test: last `--n-test-trs` TRs per run = test (default: 103, per Hedger et al.)
4. Project ROI vertex timecourses onto LBOEs → feature matrix X (banded)
5. Fit `himalaya.MultipleKernelRidgeCV` (banded ridge, LORO-CV alpha selection)
6. Compute full-model R², per-band split R², shared R²
7. Subtract OLS null-model R² → null-corrected maps R²_nc
8. Save `.npy` maps (+ `.dscalar.nii` in group_average mode)

Skip logic: if `R2_{ROI_A}_nc.npy` and `R2_{ROI_B}_nc.npy` already exist, the
run is skipped automatically.

### `integration_maps.py`

Derives Figure 3a display maps:
- `integration_score = √(clip(R²_A_nc, 0) × clip(R²_B_nc, 0))`
- `modality_balance = R²_A_nc − R²_B_nc`
- `bimodal_map.dlabel.nii` — 4-category label
- Top-10% integration mask

For `per_subject` mode: nanmeans per-subject R² maps first, then computes maps.

### `summary.py`

**group_average**: Spatial overlap with PE-AV searchlight RSA maps.
Spearman rho + 95% CI (bootstrap) for audio/video/joint RSA;
overlap map = min(integration_norm, rsa_joint_norm); results JSON.

**per_subject**: Group statistics (one-sample t-test, Cohen's d, FDR
Benjamini-Hochberg q<0.05) across subjects for R²_a_nc, R²_b_nc, and
per-subject integration score. Saves `.npy` + `.dscalar.nii` stat maps.

## Configuration

All parameters are set in `run_analysis.sh`:

| Variable | Default | Description |
|---|---|---|
| `STREAM` | `false` | Streaming mode — preprocess raw CIFTIs on-the-fly |
| `SG_FILTER` | `true` | Savitzky-Golay high-pass filter (streaming mode) |
| `PSC` | `true` | Percent signal change normalization (streaming mode) |
| `GSR` | `true` | Global signal regression (streaming mode) |
| `FMRI_SUFFIX` | `sg_psc_gsr` | Preprocessing suffix matching `preprocess_individual.py` output |
| `BATCH_SIZE` | `8` | Parallel jobs for per-subject pipeline |

## Vertex indexing — sphere vs grayordinate space

The HCP 59k_fs_LR sphere has **59292 vertices per hemisphere** (118584 bilateral),
but only **59412 are grayordinates** (medial wall excluded). Subsurface vertex
indices live in full sphere space (L: 0..59291, R: 59292..118583).

`shared/ridge_utils.py::build_sphere_to_grayord_lut(bm_axis)` builds a
118584-element array mapping sphere indices → CIFTI rows (-1 for medial wall).
This LUT is applied before LBOE projection and null-model mean computation so
that only valid grayordinate rows are indexed.

## Output layout

```
outputs/cf_modeling/
├── group_average/
│   └── {ROI_A}_{ROI_B}/
│       ├── subsurfaces/     # sub_{roi}.pkl caches (extract_geometry.py)
│       ├── prep/            # R²/R²_nc/null .npy + band_sizes + run_onsets
│       ├── cifti_maps/      # .dscalar.nii + bimodal_map.dlabel.nii
│       └── results/         # rsa_overlap_*.json (summary.py)
└── per_subject/
    └── {ROI_A}_{ROI_B}/
        ├── subsurfaces/     # sub_{roi}.pkl caches (extract_geometry.py)
        ├── subjects/
        │   └── {sub_id}/    # per-subject R²/R²_nc .npy (run_cfmodeling.py)
        └── group/
            ├── cifti_maps/  # group avg + integration + stat CIFTIs
            └── stats/       # t/d/p/fdr .npy + stats_summary.json
```

## Viewing in Connectome Workbench

```bash
wb_view \
    group_average_sg_psc_gsr_cortex_59k.dtseries.nii \
    cifti_maps/R2_{ROI_B}_nc.dscalar.nii \
    cifti_maps/R2_{ROI_A}_nc.dscalar.nii \
    cifti_maps/bimodal_map.dlabel.nii
```

Load R²_B (e.g. visual) as blue overlay and R²_A (e.g. auditory) as red,
both transparent below 0, for a Figure 3a-style dual-colour display.
