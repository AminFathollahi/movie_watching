# Attribution: Hedger et al. (2025)

## subsurface.py

The `Subsurface` class in this directory is taken verbatim from the
**vicsompy** package by Nicholas Hedger:
https://github.com/nicholashedger/vicsompy

It implements the Laplace-Beltrami Operator Eigenfunction (LBOE) decomposition
of cortical subsurfaces, which is the core mathematical primitive of the
connective field (CF) modeling approach described in:

> Hedger, N. et al. (2025). *Vicarious somatotopy: somatosensory and visual
> representations converge in human cortex during naturalistic movie watching.*
> [Journal TBD]

The file is reproduced here under Hedger's MIT License (see LICENSE).
No modifications have been made to the mathematical logic inside `Subsurface`.

## generate_leave_one_run_out (shared/ridge_utils.py)

The `generate_leave_one_run_out` function is adapted from the
**voxelwise_tutorials** package by the Gallant Lab:
https://github.com/gallantlab/voxelwise_tutorials

## Everything else

All other code in this repository (`per_subject/`, `group_average/`, `shared/`
outside of the two attributions above) was written by Amin Fathollahi
and collaborators.
