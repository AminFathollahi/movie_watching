"""
rsa/shared/residuals.py
==============================
Shared residual-embedding computations. Both RSA and encoding never call
these directly -- generation scripts (e.g.
notebooks/feature_extraction/compute_linear_residual_embeddings.py,
compute_projection_residual_embeddings.py) call these once per model, save
the result as a normal {model}_{modality}.npy file under the standard
embeddings_dir convention (see rsa/shared/model_registry.py::emb_path), and
from then on rsa/searchlight.py, rsa/partial_rsa.py, and encoding/encoding.py
consume it like any other model -- no pipeline changes needed.

Two residual types
-------------------
linear_residual()
    Ridge-regression residual of a joint (AV) embedding w.r.t. one or more
    nuisance embeddings, fit ACROSS ALL SAMPLES (one shared linear map,
    cross-validated regularization). Thin wrapper around
    rsa.multimodal_decomposition.compute_interaction_residual_cv -- that
    function already existed (built for encoding/variance_partition.py's
    Move-6 AVresid band); this module just gives RSA-side generation scripts
    the same import path so both sides share one implementation.

projection_residual()
    PER-SAMPLE orthogonal projection: at each timepoint t, builds a basis for
    the (<=2D) subspace spanned by that timepoint's own paired (a_t, v_t)
    vectors via Gram-Schmidt, projects av_t onto it, and returns av_t minus
    that projection. No regression across samples -- a purely local geometric
    decomposition, independent at every row. Distinct from linear_residual():
    a sample whose av_t happens to lie in span{a_t, v_t} gets (near-)zero
    residual regardless of what any other sample looks like.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def linear_residual(
    J: np.ndarray,
    nuisance_list: list[np.ndarray],
    alpha_grid: np.ndarray | None = None,
    cv_folds: int = 5,
) -> tuple[np.ndarray, float, float]:
    """Cross-validated-ridge residual of J w.r.t. concatenated nuisance embeddings.

    Parameters
    ----------
    J             : (n, d_J) float -- target joint (AV) embedding
    nuisance_list : list of (n, d_i) float -- nuisance embeddings (e.g.
                    [audiomae_a, videomaev2_v]), concatenated and z-scored
    alpha_grid    : ridge alpha candidates (default: logspace(0, 8, 30))
    cv_folds      : CV folds for RidgeCV

    Returns
    -------
    R          : (n, d_J) float64 -- residual (J's component NOT linearly
                 predictable from the nuisance embeddings)
    ms_score   : float -- ||R||_F^2 / ||J||_F^2
    best_alpha : float -- selected ridge alpha
    """
    from rsa.multimodal_decomposition import compute_interaction_residual_cv
    return compute_interaction_residual_cv(
        J, nuisance_list, alpha_grid=alpha_grid, cv_folds=cv_folds
    )


def projection_residual(
    av: np.ndarray,
    a: np.ndarray,
    v: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-sample residual of `av` after orthogonal projection onto span{a, v}.

    For each row t independently: Gram-Schmidt orthonormalize (a_t, v_t) into
    u1, u2 (u2 dropped to the zero vector if v_t is ~parallel to a_t, or both
    dropped if a_t/v_t are ~zero), project av_t onto span{u1, u2}, and return
    av_t minus that projection.

    Parameters
    ----------
    av, a, v : (n, d) float arrays, identical shape -- paired per-timepoint
               embeddings from the SAME model/segment set.

    Returns
    -------
    (n, d) float64 -- residual of av orthogonal to both a_t and v_t at every row.
    """
    if not (av.shape == a.shape == v.shape):
        raise ValueError(f"Shape mismatch: av={av.shape} a={a.shape} v={v.shape}")

    av64, a64, v64 = av.astype(np.float64), a.astype(np.float64), v.astype(np.float64)

    u1_norm = np.linalg.norm(a64, axis=1, keepdims=True)
    u1 = np.divide(a64, u1_norm, out=np.zeros_like(a64), where=u1_norm > eps)

    v_proj1 = np.sum(v64 * u1, axis=1, keepdims=True) * u1
    u2_raw = v64 - v_proj1
    u2_norm = np.linalg.norm(u2_raw, axis=1, keepdims=True)
    u2 = np.divide(u2_raw, u2_norm, out=np.zeros_like(u2_raw), where=u2_norm > eps)

    c1 = np.sum(av64 * u1, axis=1, keepdims=True)
    c2 = np.sum(av64 * u2, axis=1, keepdims=True)
    proj = c1 * u1 + c2 * u2
    return av64 - proj


def _demo():
    """Self-check: residual must be orthogonal to a and v at every row, and
    must vanish (up to fp error) when av is exactly in span{a, v}."""
    rng = np.random.default_rng(0)
    n, d = 50, 16

    a = rng.normal(size=(n, d))
    v = rng.normal(size=(n, d))
    coeffs = rng.normal(size=(n, 2))
    av_in_span = coeffs[:, :1] * a + coeffs[:, 1:] * v

    r_in_span = projection_residual(av_in_span, a, v)
    assert np.allclose(r_in_span, 0, atol=1e-8), \
        f"in-span residual should vanish, max abs = {np.abs(r_in_span).max():.2e}"

    av_random = rng.normal(size=(n, d))
    r = projection_residual(av_random, a, v)
    dot_a = np.sum(r * a, axis=1)
    dot_v = np.sum(r * v, axis=1)
    assert np.allclose(dot_a, 0, atol=1e-8), f"residual not orthogonal to a: max={np.abs(dot_a).max():.2e}"
    assert np.allclose(dot_v, 0, atol=1e-8), f"residual not orthogonal to v: max={np.abs(dot_v).max():.2e}"

    # Degenerate row: a_t is all-zero.
    a_deg = a.copy()
    a_deg[0] = 0.0
    r_deg = projection_residual(av_random, a_deg, v)
    assert np.isfinite(r_deg).all(), "degenerate (zero) row produced non-finite residual"

    print("rsa/shared/residuals.py self-check: OK "
          f"(in-span max|R|={np.abs(r_in_span).max():.2e}, "
          f"orthogonality max|dot|={max(np.abs(dot_a).max(), np.abs(dot_v).max()):.2e})")


if __name__ == "__main__":
    _demo()
