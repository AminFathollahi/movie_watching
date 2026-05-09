# CLAUDE.md — Audiovisual Cortical Mapping Pipeline

## Environment
```bash
conda activate vicsompy_av
cd /home/amin/Research/Representation/Movie/movie_watching/vicsompy_audiovisual
```

## Key Paths
```
DATA_BASE  = /home/amin/Research/Representation/Movie/data/Setareh
OUTPUT_DIR = /home/amin/Research/Representation/Movie/outputs/vicsompy_audiovisual
CIFTI_DIR  = OUTPUT_DIR/cifti_maps
RSA_BASE   = /home/amin/Research/Representation/Movie/outputs/searchlight_rsa_output
TEMPLATE   = DATA_BASE/HCP_S1200_GroupAvg_v1/S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii
```

## Pipeline Scripts — Run in Order
| Script | What it does | ~Time |
|---|---|---|
| `01_extract_geometry.py` | Glasser GIFTI → audio/video subsurfaces + LBOEs | 20 min |
| `02_prep_hcp_timeseries.py` | MAT files → X/Y train/test .npy | 10 min |
| `03_fit_banded_ridge.py` | Himalaya banded ridge → R2_{audio_roi}/R2_{video_roi}/R2_full/Shared | 1-2 hr |
| `04_null_corrected_maps.py` | Null model → R2_{audio_roi}_nc / R2_{video_roi}_nc | 2 min |
| `05_audiovisual_integration_maps.py` | Figure 3a maps: integration_score, modality_balance | 1 min |
| `06_rsa_overlap.py` | CF maps vs PE-AV RSA spatial overlap | 2 min |

Scripts 07 (CF weights) and 08 (RSA geometry of integration zone) require refitting script 03 with weight saving — see below.

## ROI Configuration (modular)
Each script has a `ROI_DEFS` dict at the top of its CONFIG section:

```python
# Keys = ROI short names — drive all output file names (sub_<name>.pkl, R2_<name>.npy, etc.)
# Values = {"L": left_glasser_code, "R": right_glasser_code}
ROI_DEFS = {
    "ta2": {"L": 287, "R": 107},   # audio source ROI (TA2)
    "mst": {"L": 182, "R": 2},     # video source ROI (MST)
}
AUDIO_ROI = "ta2"   # key in ROI_DEFS → audio band
VIDEO_ROI  = "mst"  # key in ROI_DEFS → video band
```

**To switch ROI pair** (e.g. back to A1 × V1): update `ROI_DEFS` and `AUDIO_ROI`/`VIDEO_ROI` in all 6 scripts. File names update automatically — old outputs are not overwritten.

Known ROI Glasser codes:
- A1: L=204, R=24
- V1: L=181, R=1
- TA2: L=287, R=107
- MST: L=182, R=2

Note: `ROI_DEFS` Glasser codes are only *used* in script 01; scripts 02–06 use only the key names for file I/O.

## Current Run: TA2 (audio) × MST (video)
```
sub_ta2.pkl  L_eigenvectors (?, ?)   R_eigenvectors (?, ?)
sub_mst.pkl  L_eigenvectors (?, 200) R_eigenvectors (?, 200)
band_sizes   [n_audio_cols, n_video_cols]  — read from band_sizes.npy
```

## Data Shapes
```
X_train      (2834, n_audio_cols+n_video_cols)
Y_train      (2834, 59412)
X_test       (328,  n_audio_cols+n_video_cols)   4 × 82 TR test clips concatenated
Y_test       (328,  59412)
band_sizes   [n_audio_cols, n_video_cols]        read from prep/band_sizes.npy — never hardcode
```

## Design Matrix Band Structure
```
cols   0 : n_lboe_audio   = Left  audio ROI LBOEs
cols   n_lboe_audio : 2*n_lboe_audio  = Right audio ROI LBOEs
cols   2*n_lboe_audio : ...            = Left  video ROI LBOEs
cols   ...            : end            = Right video ROI LBOEs
```
Always read actual values from `band_sizes.npy`, never hardcode.

## Output File Naming
ROI-specific files use the ROI key names; derived/composite maps use generic names:

| File pattern | Description |
|---|---|
| `subsurfaces/sub_{roi}.pkl` | Subsurface + LBOEs for that ROI |
| `R2_{audio_roi}.npy/.nii` | Audio-band split R² (script 03) |
| `R2_{video_roi}.npy/.nii` | Video-band split R² (script 03) |
| `R2_full.npy/.nii` | Full-model R² (script 03) |
| `Shared_R2.npy/.nii` | Shared variance (script 03) |
| `R2_null_{audio_roi}.npy/.nii` | Null model R² for audio (script 04) |
| `R2_null_{video_roi}.npy/.nii` | Null model R² for video (script 04) |
| `R2_{audio_roi}_nc.npy/.nii` | Null-corrected audio R² (script 04) |
| `R2_{video_roi}_nc.npy/.nii` | Null-corrected video R² (script 04) |
| `integration_score.npy/.nii` | Harmonic mean of both nc maps (script 05) |
| `modality_balance.npy/.nii` | audio_nc − video_nc (script 05) |
| `integration_mask.npy/.nii` | Binary top-10% integration zone (script 05) |
| `overlap_score_{config}.npy/.nii` | CF × RSA convergence map (script 06) |

## CIFTI Saving Pattern (use everywhere)
```python
import nibabel as nib
template = nib.load(TEMPLATE_CIFTI)
cifti = nib.Cifti2Image(arr.reshape(1,-1).astype(np.float32),
                        header=template.header,
                        nifti_header=template.nifti_header)
nib.save(cifti, path)
```

## Pycortex Setup (scripts 01, 05 only)
```python
import os
os.environ["PYCORTEX_FILESTORE"] = DATA_BASE + "/HCP_S1200_GroupAvg_v1"
import cortex, cortex.database
cortex.database.default_filestore = DATA_BASE + "/HCP_S1200_GroupAvg_v1"
CX_SUB = "hcp_999999_draw_NH"
```

## Why Null-Corrected R² (not Shared_R2)
`Shared_R2 = R2_audio + R2_video - R2_full` is typically ≤ 0 with himalaya banded ridge because the solver assigns separate kernel weights per band, minimising overlap. This is a property of the optimiser, not neuroscience. **Do not use Shared_R2 to identify audiovisual integration.**

Instead (mirroring Hedger et al. 2025 Figure 3a):
1. Subtract null-model R² from each split R² → removes global (non-topographic) responsiveness
2. Plot R2_{audio_roi}_nc and R2_{video_roi}_nc as independent axes of a 2D colormap
3. Spatial co-occurrence of both being high = audiovisual integration zone

## RSA File Naming Convention
```
RSA_BASE/pe-av-small-16-frame/
  k100_{norm}_{hrf_dir}_{method}/{bin}/{modality}/{hem}_hemisphere/
    rsa_pe-av-small-16-frame_{modality}_k100_spearman_{norm}_{hrf_tag}_bin{bin}_{hem}_{method}.npy

hrf_dir  : 'hrf'          → hrf_tag in filename: 'hrf'
           'blockdiag_delay5s' or 'global_delay5s' → hrf_tag: 'nohrf'
```
See `RSA_CONFIGS` dict at top of `06_rsa_overlap.py` for all valid combinations.

## Analysis 3 Prerequisite: Save Model Weights
Script 03 does not currently save model weights. To enable Analysis 3 (CF weight maps),
add the following block to `03_fit_banded_ridge.py` inside `fit_and_decompose()` after `pipeline.fit()`:
```python
# Save dual weights for Analysis 3 (CF parameter maps)
mkr = pipeline[-1]  # MultipleKernelRidgeCV
dual_w = backend.to_numpy(mkr.dual_coef_)           # (n_train, n_targets)
np.save(os.path.join(OUTPUT_DIR, 'dual_weights.npy'), dual_w.astype(np.float32))
# Save scaled X_train so primal weights can be reconstructed without re-fitting
X_train_scaled = pipeline[:-1].transform(X_train)
np.save(os.path.join(OUTPUT_DIR, 'X_train_scaled.npy'), backend.to_numpy(X_train_scaled).astype(np.float32))
```
Then run `python 03_fit_banded_ridge.py` again (~1-2 hr).

## Scripts 01-03 Audit Notes
All three scripts are clean and correct. One minor inefficiency in script 01:
`load_sphere_surfaces()` is called inside `build_subsurface()`, which runs twice (for audio and video ROIs).
The sphere GIFTIs are loaded twice — harmless but wasteful if rerunning without cache.
Not worth fixing unless startup time becomes an issue.
