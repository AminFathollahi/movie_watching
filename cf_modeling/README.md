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
| `vendor/hedger_cf/utils.py` — `generate_leave_one_run_out()` | [Hedger et al. (2025) / vicsompy](https://github.com/nicholashedger/vicsompy) (vicsompy itself credits [Gallant Lab / voxelwise_tutorials](https://github.com/gallantlab/voxelwise_tutorials)) | MIT |

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

cf_modeling uses the **full-run** output of `preprocess_individual.py`
(no `--timing-csv`). This mode applies per-run preprocessing (SG high-pass,
PSC, GSR, z-score) and concatenates all 4 runs, producing a CIFTI that
contains every TR in its original order — the train/test split is done
inside `run_cfmodeling.py` (last 103 TRs of each run = test, following Hedger et al.).

```bash
python preprocess_individual.py \
    --raw-dir $CIFTI_DIR \
    --out-dir $FMRI_OUT_DIR \
    --subjects-list subjects.txt \
    --sg-filter --psc \        # full-run mode: SG + PSC + GSR + z-score per run
    --save-individual \        # persist per-subject CIFTIs to disk
    --save-average             # also build + save group-average CIFTI
```

This produces, for each subject and the group average:
```
{sub}_sg_psc_gsr_zscore_cortex_59k.dtseries.nii   — (n_cortex, T_total)
{sub}_sg_psc_gsr_zscore_run_trs.npy               — [T_run1, T_run2, …]
group_average_sg_psc_gsr_zscore_cortex_59k.dtseries.nii
group_average_sg_psc_gsr_zscore_run_trs.npy
```

Set `FMRI_SUFFIX="sg_psc_gsr_zscore"` in `run_analysis.sh` to match.

**Streaming alternative**: set `STREAM=true` in `run_analysis.sh` to skip saving
preprocessed CIFTIs. Raw data is preprocessed on-the-fly inside `run_cfmodeling.py`;
only the final R²_nc maps are written to disk.

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
    └── ridge_utils.py        # build_pipeline, project_onto_lboes, fit_null_r2
```

## Usage

```bash
conda activate cfmod

# Preprocess fMRI first (see above), then:
bash run_analysis.sh all           # group-average + per-subject (all ROI pairs)
bash run_analysis.sh groupaverage  # group-average only
bash run_analysis.sh persubject    # per-subject only
bash run_analysis.sh persubject 4  # per-subject, 4 parallel jobs
```

Edit `run_all_persubject()` and `run_all_groupaverage()` in `run_analysis.sh`
to add or remove ROI pairs.

## Script-by-script overview

### `extract_geometry.py`

Builds `Subsurface59k` objects (extending Hedger et al.'s `Subsurface`) and
computes up to 200 LBOEs for two Glasser ROIs.

**Surface**: 59k_fs_LR sphere (`S1200.{L,R}.sphere.59k_fs_LR.surf.gii`),
following the Hedger et al. (2025) convention.
**Parcellation**: 59k_fs_LR Glasser dlabel.nii for ROI vertex masks.
Both modes share the same geometry; `--mode` only determines the output directory.

```bash
python extract_geometry.py --mode group_average --roi_a A1 --roi_b V1
python extract_geometry.py --mode per_subject   --roi_a A5 --roi_b FFC
```

### `run_cfmodeling.py` (phases 02→04)

Unified entry point for data preparation, banded ridge fitting, and null
correction. Combines what scripts 02–04 do separately, in one parallel-safe
Python process per (subject, ROI pair).

**Disk mode** (default — requires pre-saved CIFTIs):
```bash
# group average
python run_cfmodeling.py \
    --mode group_average --roi-a A1 --roi-b V1 \
    --preprocessed-dir /path/to/preprocessed \
    --fmri-suffix sg_psc_gsr_zscore \
    --output-base /path/to/outputs/cf_modeling \
    --template-cifti /path/to/group_average_..._cortex_59k.dtseries.nii

# per subject
python run_cfmodeling.py \
    --mode per_subject --roi-a A5 --roi-b FFC --subject 100610 \
    --preprocessed-dir /path/to/preprocessed \
    --fmri-suffix sg_psc_gsr_zscore \
    --output-base /path/to/outputs/cf_modeling
```

**Streaming mode** (per_subject only — no CIFTI saved):
```bash
python run_cfmodeling.py \
    --mode per_subject --roi-a A5 --roi-b FFC --subject 100610 \
    --raw-dir /path/to/raw_ciftis \
    --output-base /path/to/outputs/cf_modeling \
    [--sg-filter] [--psc] [--no-gsr] [--no-z-score]
```

What it does internally:
1. Load CIFTI (disk) or preprocess raw CIFTI on-the-fly (streaming)
2. Split train/test: last `--n-test-trs` TRs per run = test (default: 103)
3. Project ROI vertex timecourses onto LBOEs → feature matrix X
4. Fit `himalaya.MultipleKernelRidgeCV` (banded ridge, LORO-CV alpha selection)
5. Compute full-model R², per-band split R², shared R²
6. Subtract OLS null-model R² → null-corrected maps R²_nc
7. Save `.npy` maps (+ `.dscalar.nii` in group_average mode)

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

## Output layout

```
outputs/cf_modeling/
├── group_average/
│   └── {ROI_A}_{ROI_B}/
│       ├── subsurfaces/     # sub_{roi}.pkl caches (script 01)
│       ├── prep/            # R²/R²_nc/null .npy + band_sizes + run_onsets
│       ├── cifti_maps/      # .dscalar.nii + bimodal_map.dlabel.nii
│       └── results/         # rsa_overlap_*.json (script 06)
└── per_subject/
    └── {ROI_A}_{ROI_B}/
        ├── subsurfaces/     # sub_{roi}.pkl caches (script 01)
        ├── subjects/
        │   └── {sub_id}/    # per-subject R²/R²_nc .npy (run_cfmodeling.py)
        └── group/
            ├── cifti_maps/  # group avg + integration + stat CIFTIs
            └── stats/       # t/d/p/fdr .npy + stats_summary.json
```

## Viewing in Connectome Workbench

```bash
wb_view \
    group_average_sg_psc_gsr_zscore_cortex_59k.dtseries.nii \
    cifti_maps/R2_{ROI_B}_nc.dscalar.nii \
    cifti_maps/R2_{ROI_A}_nc.dscalar.nii \
    cifti_maps/bimodal_map.dlabel.nii
```

Load R²_B (e.g. visual) as blue overlay and R²_A (e.g. auditory) as red,
both transparent below 0, for a Figure 3a-style dual-colour display.
