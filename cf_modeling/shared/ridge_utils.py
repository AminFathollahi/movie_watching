"""
shared/ridge_utils.py
=====================
Shared ridge regression utilities for the CF modeling pipeline.

generate_leave_one_run_out is imported verbatim from vendor/hedger_cf/utils.py,
which is itself taken verbatim from vicsompy/utils.py (Hedger et al. 2025, MIT License).
vicsompy in turn borrowed the function from the gallantlab/voxelwise_tutorials package
and documents this in its own docstring.
"""

import os
import sys

import numpy as np
from himalaya.backend import set_backend
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV
from sklearn.model_selection import check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vendor.hedger_cf.utils import generate_leave_one_run_out  # noqa: F401  (re-exported)


def build_pipeline(n_samples_train, run_onsets, band_sizes, roi_names,
                   backend_engine="torch", solver="random_search",
                   n_iter=20, alpha_min=1, alpha_max=20, alpha_vals=20,
                   n_targets_batch=400, n_alphas_batch=10,
                   n_targets_batch_refit=400, with_mean=True, with_std=True):
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


def project_onto_lboes(data_118k, subsurfaces):
    """Project BOLD surface data onto LBOEs for a list of Subsurface objects.

    Replicates vicsompy MssCf.make_roi_data() + make_eigs() + make_dm().

    Parameters
    ----------
    data_118k  : (118584, T) — bilateral surface, L=0:59292, R=59292:118584
    subsurfaces: list of Subsurface — each must have subsurface_verts_L/R,
                 L_eigenvectors, R_eigenvectors

    Returns
    -------
    dm         : (T, sum(2*n_lboe)) — LBOE design matrix
    band_sizes : list of int — number of columns per ROI (2 * n_lboe each)
    """
    dms = []
    band_sizes = []
    for sub in subsurfaces:
        vL = sub.subsurface_verts_L
        vR = sub.subsurface_verts_R
        dms.append(data_118k[vL, :].T @ sub.L_eigenvectors.real)   # (T, n_lboe)
        dms.append(data_118k[vR, :].T @ sub.R_eigenvectors.real)   # (T, n_lboe)
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
