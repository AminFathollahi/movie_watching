# Attribution: Hedger et al. (2025)

## subsurface.py — vendored verbatim

The `Subsurface` class in this directory is taken **verbatim** from the
**vicsompy** package by Nicholas Hedger:
https://github.com/nicholashedger/vicsompy

License: MIT (see LICENSE in this directory)

It implements the Laplace-Beltrami Operator Eigenfunction (LBOE) decomposition
of cortical subsurfaces, which is the core mathematical primitive of the
connective field (CF) modeling approach described in:

> Hedger, N. et al. (2025). *Vicarious somatotopy: somatosensory and visual
> representations converge in human cortex during naturalistic movie watching.*

The file is reproduced here without modification to the mathematical logic.

## utils.py — vendored verbatim

The `generate_leave_one_run_out` function in this directory is taken **verbatim**
from `vicsompy/utils.py`.

Source : https://github.com/nicholashedger/vicsompy
License: MIT (see LICENSE in this directory)

vicsompy itself documents this function as borrowed from the **voxelwise_tutorials**
package by the Gallant Lab (UC Berkeley, MIT License):
https://github.com/gallantlab/voxelwise_tutorials

That attribution is preserved verbatim in the function's docstring.
`shared/ridge_utils.py` now imports it directly from `vicsompy.utils` (the source repo),
not from this vendor copy.  The vendor copy is retained for reference only — see
`vendor/DEPRECATED.md`.

## Overall pipeline fidelity

The CF modeling pipeline (scripts 01–06) is a faithful reimplementation of the
analysis described in Hedger et al. (2025). The following design decisions are
taken directly from vicsompy / the paper:

- **Surface type**: 59k_fs_LR sphere (`S1200.{L,R}.sphere.59k_fs_LR.surf.gii`),
  matching the convention in Hedger et al. (2025).
- **LBOEs**: up to 200 eigenfunctions per hemisphere per ROI (dynamic: capped
  if the ROI has fewer vertices; Hedger et al. used a fixed 200).
- **Train/test split**: last 103 TRs of each run = test set (vicsompy default).
- **Preprocessing**: SG high-pass filter → PSC → GSR → z-score per run,
  applied via `preprocess_individual.py` before script 02 runs.
- **Feature matrix X**: project ROI vertex timecourses onto LBOEs (L and R
  hemispheres separately), concatenate ROI_A + ROI_B → (T, 2·n_lboe_A + 2·n_lboe_B).
- **Model**: himalaya `MultipleKernelRidgeCV` (banded ridge), LORO-CV for alpha.
- **Null model**: single-regressor OLS (mean ROI timecourse on test set).
- **Integration maps**: geometric mean of null-corrected split R² (Figure 3a).

Departures from vicsompy (intentional):
- ROI names are passed as CLI arguments (`--roi_a`, `--roi_b`) rather than
  hardcoded, allowing any pair of Glasser HCP-MMP1 ROIs.
- LBOE count is capped dynamically to `min(200, n_verts_L − 2, n_verts_R − 2)`.
- Data are loaded from preprocessed CIFTIs (preprocess_individual.py output)
  instead of raw CIFTI dtseries.
- The pipeline supports both group-average and per-subject modes via `--mode`.

## Everything else

All other code in `cf_modeling/` (scripts 01–06, `shared/ridge_utils.py` outside
the two attributed functions, and `cf_modeling/run_analysis.sh`) was written by
Mohammad Amin Fathollahi and collaborators.
