"""
shared/ridge_utils.py
=====================
Shared ridge regression utilities for the CF modeling pipeline.

generate_leave_one_run_out is imported directly from vicsompy.utils (Hedger et al. 2025,
MIT License), which itself credits the gallantlab/voxelwise_tutorials package and documents
this in the function's docstring.

vicsompy is imported without pip install by inserting the source repo path:
  VICSOMPY_REPO (default: <MOVIE_ROOT>/Vicarious_somatotopy)
This path is already inserted by 01_extract_geometry.py or 02_fit_cf_model.py before
this module is imported.  If called in isolation, we fall back to the default path.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV
from sklearn.model_selection import check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Ensure vicsompy is importable from the source repo (no pip install required)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import ROOT  # noqa: E402

_DEFAULT_VICSOMPY_REPO = os.environ.get("VICSOMPY_REPO", str(ROOT / "Vicarious_somatotopy"))
if _DEFAULT_VICSOMPY_REPO not in sys.path:
    sys.path.insert(0, _DEFAULT_VICSOMPY_REPO)

from vicsompy.utils import generate_leave_one_run_out  # noqa: F401  (re-exported)


def build_pipeline(n_samples_train, run_onsets, band_sizes, roi_names,
                   backend_engine="torch_cuda", solver="random_search",
                   n_iter=20, alpha_min=1, alpha_max=20, alpha_vals=20,
                   n_targets_batch=20000, n_alphas_batch=10,
                   n_targets_batch_refit=20000, with_mean=True, with_std=True):
    """Build the himalaya ColumnKernelizer → MultipleKernelRidgeCV pipeline.

    Replicates vicsompy's MssCf.prep_pipeline() sequence:
        make_cv() → setup_model() → make_preproc() → kernelize() → complete_pipeline()

    Parameters
    ----------
    n_samples_train : int — number of training samples (for LORO-CV)
    run_onsets      : array of int (n_runs,) — training run onset indices
    band_sizes      : list of int — number of columns per band (2 * n_lboe each)
    roi_names       : list of str — band names for ColumnKernelizer
    backend_engine  : str — himalaya backend ('torch', 'numpy', 'cupy')
    ... (himalaya solver params matching vicsompy config.yml defaults)

    Returns
    -------
    pipeline : sklearn Pipeline — ready to .fit(X_train, Y_train)
    backend  : himalaya backend object — for .to_numpy() conversions
    """
    if backend_engine == "torch_cuda":
        torch.set_default_device('cuda')
    backend = set_backend(backend_engine, on_error="warn")

    cv_gen = generate_leave_one_run_out(n_samples_train, run_onsets)
    cv = check_cv(cv_gen)

    # Band slices: [0, band_sizes[0], band_sizes[0]+band_sizes[1], ...]
    starts = np.concatenate([[0], np.cumsum(band_sizes)])
    slices = [slice(int(starts[i]), int(starts[i + 1])) for i in range(len(band_sizes))]

    def make_preproc():
        return make_pipeline(
            StandardScaler(with_mean=with_mean, with_std=with_std),
            Kernelizer(kernel="linear"),
        )

    column_kernelizer = ColumnKernelizer(
        [(name, make_preproc(), sl) for name, sl in zip(roi_names, slices)]
    )

    alphas = np.logspace(alpha_min, alpha_max, alpha_vals)
    solver_params = dict(
        n_iter=n_iter,
        alphas=alphas,
        n_targets_batch=n_targets_batch,
        n_alphas_batch=n_alphas_batch,
        n_targets_batch_refit=n_targets_batch_refit,
    )
    model = MultipleKernelRidgeCV(
        kernels="precomputed",
        solver=solver,
        solver_params=solver_params,
        cv=cv,
    )

    pipeline = make_pipeline(column_kernelizer, model)
    return pipeline, backend


def build_sphere_to_grayord_lut(bm_axis, n_verts_per_hem: int = 59292) -> np.ndarray:
    """Map full-sphere bilateral vertex indices → CIFTI grayordinate positions.

    The HCP 59k_fs_LR CIFTI stores only non-medial-wall vertices (108441 total
    out of 2×59292=118584 full-sphere positions).  Subsurface vertex indices
    (subsurface_verts_L/R from extract_geometry.py) are in full-sphere space;
    this LUT translates them to grayordinate row positions so that the CIFTI
    data array can be indexed correctly.

    Parameters
    ----------
    bm_axis         : nibabel BrainModelAxis — cortex-only slice (from
                      preprocess_individual.preprocess_subject or nib.load+get_axis)
    n_verts_per_hem : int — vertices per hemisphere in the full sphere (59292 for
                      HCP 59k_fs_LR)

    Returns
    -------
    lut : (2 * n_verts_per_hem,) int32
        lut[sphere_vertex_idx] = grayordinate_position, or -1 for medial wall.
        Left hemisphere: indices 0..n_verts_per_hem-1
        Right hemisphere: indices n_verts_per_hem..2*n_verts_per_hem-1
    """
    lut = np.full(2 * n_verts_per_hem, -1, dtype=np.int32)
    pos = 0
    for name, _, model in bm_axis.iter_structures():
        n = len(model.vertex)
        if "CORTEX_LEFT" in name:
            lut[model.vertex] = np.arange(pos, pos + n, dtype=np.int32)
        elif "CORTEX_RIGHT" in name:
            lut[n_verts_per_hem + model.vertex] = np.arange(pos, pos + n, dtype=np.int32)
        pos += n
    return lut


def project_onto_lboes(data, subsurfaces, lut=None):
    """Project BOLD surface data onto LBOEs for a list of Subsurface objects.

    Replicates vicsompy MssCf.make_roi_data() + make_eigs() + make_dm().

    Parameters
    ----------
    data       : array — surface data.
                 • If lut is None   : (118584, T) full bilateral sphere
                                      (L=0:59292, R=59292:118584).
                 • If lut is given  : (n_grayord, T) CIFTI grayordinate data
                                      (medial wall excluded, e.g. 108441 vertices).
    subsurfaces: list of Subsurface — each must have subsurface_verts_L/R,
                 L_eigenvectors, R_eigenvectors, n_lboe
    lut        : (118584,) int32 — from build_sphere_to_grayord_lut(), or None.
                 When provided, subsurface sphere indices are translated to
                 grayordinate positions before indexing data.

    Returns
    -------
    dm         : (T, sum(2*n_lboe)) — LBOE design matrix
    band_sizes : list of int — columns per ROI (2 * n_lboe each)
    """
    dms = []
    band_sizes = []
    for sub in subsurfaces:
        if lut is not None:
            vL = lut[sub.subsurface_verts_L]
            vR = lut[sub.subsurface_verts_R]
            if np.any(vL < 0) or np.any(vR < 0):
                bad_L = int(np.sum(vL < 0))
                bad_R = int(np.sum(vR < 0))
                raise ValueError(
                    f"ROI vertices in medial wall (not in CIFTI grayordinates): "
                    f"L={bad_L}  R={bad_R}. Check ROI/surface alignment.")
        else:
            vL = sub.subsurface_verts_L
            vR = sub.subsurface_verts_R
        dms.append(data[vL, :].T @ sub.L_eigenvectors.real)   # (T, n_lboe)
        dms.append(data[vR, :].T @ sub.R_eigenvectors.real)   # (T, n_lboe)
        band_sizes.append(sub.n_lboe * 2)
    return np.hstack(dms), band_sizes


def fit_null_r2(regressor, Y):
    """Single-regressor OLS null model: R² for all targets.

    Model: Y ~ intercept + beta * regressor
    Matches vicsompy's test_null_model / test_null_model_precomputed OLS math.

    Parameters
    ----------
    regressor : (T,)
    Y         : (T, n_targets)

    Returns
    -------
    R2_null : (n_targets,) float32
    """
    T = Y.shape[0]
    dm = np.column_stack([regressor, np.ones(T)])
    betas, _, _, _ = np.linalg.lstsq(dm, Y, rcond=None)
    Y_hat = dm @ betas
    ss_res = np.sum((Y - Y_hat) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    R2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
    return R2.astype(np.float32)
