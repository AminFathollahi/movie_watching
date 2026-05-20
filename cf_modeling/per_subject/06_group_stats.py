"""
06_group_stats.py
=================
Group-level statistics and paper-quality figures, following the vicsompy paper
(Hedger et al. 2025) "2 Aggregate Plot" notebook methodology.

For each vertex, computes across subjects:
  - One-sample t-test on null-corrected R² (H0: mean = 0)
  - Cohen's d effect size  (d = t / sqrt(N))
  - FDR-corrected significance mask (Benjamini-Hochberg, q < 0.05)

Additionally computes per-subject integration_score = sqrt(R2_A_nc × R2_B_nc)
and runs the same t-test / Cohen's d on that map.

Outputs
-------
  group/stats/
    t_{ROI_A}_nc.npy, d_{ROI_A}_nc.npy, p_{ROI_A}_nc.npy
    t_{ROI_B}_nc.npy, d_{ROI_B}_nc.npy, p_{ROI_B}_nc.npy
    t_integration.npy, d_integration.npy, p_integration.npy
    fdr_mask_{ROI_A}_nc.npy, fdr_mask_{ROI_B}_nc.npy
    stats_summary.json
  group/cifti_59k/stats/   — t and d maps as dscalar.nii
  group/cifti_32k/stats/   — same maps resampled to 32k
  group/figures/
    d_{ROI_A}_nc_flatmap.png         — Cohen's d, hot colormap  (Fig 1e style)
    d_{ROI_B}_nc_flatmap.png         — Cohen's d, blues colormap
    d_{ROI_A}_nc_fdr_flatmap.png     — FDR-thresholded version
    d_{ROI_B}_nc_fdr_flatmap.png
    d_integration_flatmap.png        — integration effect size
    variance_decomp_2D_flatmap.png   — Vertex2D group avg (Fig 3a style)

Run
---
    conda activate vicsompy_av
    python 06_group_stats.py [--min_subjects N] [--no_pycortex] \\
        --roi_a A5 --roi_b FFC \\
        --output_base /path/to/outputs \\
        --hcp_dir /path/to/HCP_S1200_GroupAvg_v1 \\
        --cifti_dir /path/to/HCP/fMRI_CIFTI
"""

# =============================================================================
# CONFIG — defaults; overridden by argparse in __main__
# =============================================================================
import os
import sys
import json

_DEFAULT_DATA_BASE   = "/home/amin/Research/Representation/Movie/data/Setareh"
_DEFAULT_HCP_DIR     = f"{_DEFAULT_DATA_BASE}/HCP_S1200_GroupAvg_v1"
_DEFAULT_CIFTI_DIR   = "/media/amin/Samsung_T5/HCP/Data/fMRI_CIFTI"
_DEFAULT_ROI_A       = "A5"
_DEFAULT_ROI_B       = "FFC"
_DEFAULT_OUTPUT_BASE = "/home/amin/Research/Representation/Movie/outputs/cf_modeling/per_subject"

ROI_A         = _DEFAULT_ROI_A
ROI_B         = _DEFAULT_ROI_B
HCP_DIR       = _DEFAULT_HCP_DIR
CIFTI_RAW_DIR = _DEFAULT_CIFTI_DIR
OUTPUT_ROOT     = f"{_DEFAULT_OUTPUT_BASE}/{ROI_A}_{ROI_B}"
SUBJECTS_DIR    = f"{OUTPUT_ROOT}/subjects"
GROUP_DIR       = f"{OUTPUT_ROOT}/group"
STATS_DIR       = f"{GROUP_DIR}/stats"
CIFTI_59K_DIR   = f"{GROUP_DIR}/cifti_59k"
CIFTI_32K_DIR   = f"{GROUP_DIR}/cifti_32k"
CIFTI_59K_STATS = f"{CIFTI_59K_DIR}/stats"
CIFTI_32K_STATS = f"{CIFTI_32K_DIR}/stats"
FIGURES_DIR     = f"{GROUP_DIR}/figures"

TEMPLATE_32K = f"{HCP_DIR}/S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii"
SPHERE_59K_L = f"{HCP_DIR}/S1200.L.sphere.59k_fs_LR.surf.gii"
SPHERE_59K_R = f"{HCP_DIR}/S1200.R.sphere.59k_fs_LR.surf.gii"
SPHERE_32K_L = f"{HCP_DIR}/S1200.L.sphere.32k_fs_LR.surf.gii"
SPHERE_32K_R = f"{HCP_DIR}/S1200.R.sphere.32k_fs_LR.surf.gii"

PYCORTEX_FILESTORE = HCP_DIR
CX_SUB             = "hcp_999999_draw_NH"

FDR_Q = 0.05
ALPHA = 0.05

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import logging

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.cifti_io import (
    collect_maps,
    get_59k_bm_axis,
    save_59k_cifti,
    resample_to_32k,
    cifti_59k_to_surface,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# Data loading
# =============================================================================

def load_group_avg(name):
    path = os.path.join(GROUP_DIR, f"{name}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run 05_group_average.py first."
        )
    return np.load(path)


# =============================================================================
# Statistics
# =============================================================================

def one_sample_stats(maps):
    """Per-vertex one-sample t-test (H0: mean=0) + Cohen's d across subjects."""
    N = maps.shape[0]
    t_stat, p_val = scipy_stats.ttest_1samp(maps, popmean=0, axis=0,
                                             nan_policy="omit")
    d_stat = t_stat / np.sqrt(N)
    return t_stat.astype(np.float32), d_stat.astype(np.float32), p_val.astype(np.float32)


def fdr_mask(p_values, q=FDR_Q):
    """Benjamini-Hochberg FDR correction → boolean mask of significant vertices."""
    finite = np.isfinite(p_values)
    mask   = np.zeros(p_values.shape, dtype=bool)
    if not finite.any():
        return mask
    try:
        rejected = scipy_stats.false_discovery_control(
            p_values[finite], axis=0, method="bh"
        ) < q
    except AttributeError:
        p_fin  = p_values[finite]
        n      = len(p_fin)
        order  = np.argsort(p_fin)
        ranked = np.empty(n, dtype=int)
        ranked[order] = np.arange(1, n + 1)
        rejected = p_fin <= (ranked / n) * q
    mask[finite] = rejected
    return mask


# =============================================================================
# MAIN
# =============================================================================

def main(min_subjects=1, no_pycortex=False):
    log.info("=" * 60)
    log.info("Script 06 — Group statistics + figures")
    log.info(f"  ROIs: {ROI_A} × {ROI_B}")
    log.info(f"  FDR q={FDR_Q}  alpha={ALPHA}")
    log.info("=" * 60)

    log.info("\nLoading per-subject null-corrected R² maps …")
    maps_a_nc = collect_maps(SUBJECTS_DIR, f"R2_{ROI_A}_nc", min_subjects)
    maps_b_nc = collect_maps(SUBJECTS_DIR, f"R2_{ROI_B}_nc", min_subjects)
    N = maps_a_nc.shape[0]
    log.info(f"  N subjects: {N}")

    log.info("\nLoading group averages from script 05 …")
    R2_a_nc_avg = load_group_avg(f"R2_{ROI_A}_nc_avg")
    R2_b_nc_avg = load_group_avg(f"R2_{ROI_B}_nc_avg")

    log.info("\nComputing per-subject integration scores …")
    maps_integ = np.sqrt(
        np.clip(maps_a_nc, 0, None) * np.clip(maps_b_nc, 0, None)
    ).astype(np.float32)

    critical_t = scipy_stats.t.ppf(1 - ALPHA / 2, df=N - 1)
    log.info(f"  Critical t (two-tailed α={ALPHA}, df={N-1}): {critical_t:.3f}")

    log.info("\nRunning one-sample t-tests (H0: mean = 0) …")
    t_a, d_a, p_a = one_sample_stats(maps_a_nc)
    t_b, d_b, p_b = one_sample_stats(maps_b_nc)
    t_i, d_i, p_i = one_sample_stats(maps_integ)

    log.info(f"  {ROI_A}: mean_d={d_a.mean():.4f}  "
             f"frac_sig(|t|>{critical_t:.2f}): {np.mean(np.abs(t_a) > critical_t):.1%}")
    log.info(f"  {ROI_B}: mean_d={d_b.mean():.4f}  "
             f"frac_sig(|t|>{critical_t:.2f}): {np.mean(np.abs(t_b) > critical_t):.1%}")
    log.info(f"  Integration: mean_d={d_i.mean():.4f}  "
             f"frac_sig: {np.mean(np.abs(t_i) > critical_t):.1%}")

    log.info("\nApplying FDR correction (Benjamini-Hochberg) …")
    fdr_a = fdr_mask(p_a)
    fdr_b = fdr_mask(p_b)
    log.info(f"  FDR-sig {ROI_A}: {fdr_a.sum()} / {len(fdr_a)} vertices "
             f"({fdr_a.mean():.1%})")
    log.info(f"  FDR-sig {ROI_B}: {fdr_b.sum()} / {len(fdr_b)} vertices "
             f"({fdr_b.mean():.1%})")

    log.info(f"\nSaving stats to {STATS_DIR} …")
    for name, arr in {
        f"t_{ROI_A}_nc":        t_a,
        f"d_{ROI_A}_nc":        d_a,
        f"p_{ROI_A}_nc":        p_a,
        f"t_{ROI_B}_nc":        t_b,
        f"d_{ROI_B}_nc":        d_b,
        f"p_{ROI_B}_nc":        p_b,
        "t_integration":         t_i,
        "d_integration":         d_i,
        "p_integration":         p_i,
        f"fdr_mask_{ROI_A}_nc": fdr_a.astype(np.float32),
        f"fdr_mask_{ROI_B}_nc": fdr_b.astype(np.float32),
    }.items():
        np.save(os.path.join(STATS_DIR, f"{name}.npy"), arr)

    summary = {
        "N_subjects":             N,
        "ROI_A":                  ROI_A,
        "ROI_B":                  ROI_B,
        "critical_t":             float(critical_t),
        "alpha":                  ALPHA,
        "fdr_q":                  FDR_Q,
        f"mean_d_{ROI_A}_nc":     float(d_a[np.isfinite(d_a)].mean()),
        f"mean_d_{ROI_B}_nc":     float(d_b[np.isfinite(d_b)].mean()),
        "mean_d_integration":     float(d_i[np.isfinite(d_i)].mean()),
        f"fdr_sig_{ROI_A}":       int(fdr_a.sum()),
        f"fdr_sig_{ROI_B}":       int(fdr_b.sum()),
        f"uncorr_sig_{ROI_A}":    int((np.abs(t_a) > critical_t).sum()),
        f"uncorr_sig_{ROI_B}":    int((np.abs(t_b) > critical_t).sum()),
    }
    with open(os.path.join(STATS_DIR, "stats_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("  stats_summary.json saved")

    log.info("\nBuilding 59k CIFTI template …")
    try:
        bm_axis = get_59k_bm_axis(SUBJECTS_DIR, CIFTI_RAW_DIR)
        cifti_paths = {}
        for name, arr in [
            (f"t_{ROI_A}_nc",  t_a),
            (f"d_{ROI_A}_nc",  d_a),
            (f"t_{ROI_B}_nc",  t_b),
            (f"d_{ROI_B}_nc",  d_b),
            ("t_integration",   t_i),
            ("d_integration",   d_i),
        ]:
            cifti_paths[name] = save_59k_cifti(arr, name, bm_axis, CIFTI_59K_STATS)

        log.info("\nResampling stats maps 59k → 32k …")
        for name, p59 in cifti_paths.items():
            if p59 and os.path.exists(p59):
                resample_to_32k(p59, name, CIFTI_32K_STATS,
                                TEMPLATE_32K,
                                SPHERE_59K_L, SPHERE_59K_R,
                                SPHERE_32K_L, SPHERE_32K_R)

    except Exception as e:
        log.warning(f"  CIFTI output failed: {e}")
        bm_axis = None

    if no_pycortex:
        log.info("\n  --no_pycortex: skipping flatmap figures.")
        log.info("\nScript 06 complete.")
        return

    if bm_axis is None:
        log.warning("  No BrainModelAxis — skipping pycortex figures.")
        log.info("\nScript 06 complete.")
        return

    log.info("\nGenerating pycortex flatmap figures …")
    import cortex
    import matplotlib.pyplot as plt
    from shared.vis_utils import Plot

    os.environ["PYCORTEX_FILESTORE"] = PYCORTEX_FILESTORE
    cortex.database.default_filestore = PYCORTEX_FILESTORE

    pos_a = R2_a_nc_avg[R2_a_nc_avg > 0]
    pos_b = R2_b_nc_avg[R2_b_nc_avg > 0]
    p95_a = float(np.percentile(pos_a, 95)) if len(pos_a) else 0.05
    p95_b = float(np.percentile(pos_b, 95)) if len(pos_b) else 0.05

    d_max = 0.6

    try:
        # Figure A — Cohen's d flatmaps (1D)
        log.info("  Figure A — Cohen's d flatmaps …")
        for surf_dat, fname, cmap, title in [
            (cifti_59k_to_surface(np.clip(d_a, 0, None), bm_axis),
             f"d_{ROI_A}_nc_flatmap", "hot",
             f"Cohen's d — {ROI_A} null-corrected R² (N={N})"),
            (cifti_59k_to_surface(np.clip(d_b, 0, None), bm_axis),
             f"d_{ROI_B}_nc_flatmap", "Blues",
             f"Cohen's d — {ROI_B} null-corrected R² (N={N})"),
            (cifti_59k_to_surface(np.clip(d_i, 0, None), bm_axis),
             "d_integration_flatmap", "viridis",
             f"Cohen's d — Integration score (N={N})"),
        ]:
            vx  = cortex.Vertex(surf_dat, subject=CX_SUB,
                                cmap=cmap, vmin=0, vmax=d_max)
            fig = cortex.quickflat.make_figure(vx, with_curvature=True,
                                               with_colorbar=True)
            fig.suptitle(title, fontsize=14)
            out = os.path.join(FIGURES_DIR, f"{fname}.png")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            log.info(f"  Figure saved: {out}")

        # Figure B — 2D variance decomposition via shared.vis_utils.Plot
        log.info("  Figure B — 2D variance decomposition (Plot.uber_plot) …")
        surf_a_avg = cifti_59k_to_surface(R2_a_nc_avg, bm_axis)
        surf_b_avg = cifti_59k_to_surface(R2_b_nc_avg, bm_axis)
        p = Plot(
            dat=surf_b_avg,
            vmin=0, vmax=p95_b,
            vmin2=0, vmax2=p95_a,
            subject=CX_SUB,
            cmap="PU_RdBu_covar_alpha",
            x=10, y=8, dpi=150,
            with_curvature=True, with_colorbar=True,
        )
        p.uber_plot(dat2=surf_a_avg)
        p.saveout(outloc=os.path.join(FIGURES_DIR, "variance_decomp_2D_flatmap.png"))
        log.info(f"  Figure saved: {FIGURES_DIR}/variance_decomp_2D_flatmap.png")

        # Figure C — FDR-thresholded Cohen's d
        log.info("  Figure C — FDR-thresholded Cohen's d …")
        for surf_dat, fname, cmap, title in [
            (cifti_59k_to_surface(np.clip(d_a, 0, None) * fdr_a, bm_axis),
             f"d_{ROI_A}_nc_fdr_flatmap", "hot",
             f"Cohen's d (FDR q<{FDR_Q}) — {ROI_A} (N={N})"),
            (cifti_59k_to_surface(np.clip(d_b, 0, None) * fdr_b, bm_axis),
             f"d_{ROI_B}_nc_fdr_flatmap", "Blues",
             f"Cohen's d (FDR q<{FDR_Q}) — {ROI_B} (N={N})"),
        ]:
            vx  = cortex.Vertex(surf_dat, subject=CX_SUB,
                                cmap=cmap, vmin=0, vmax=d_max)
            fig = cortex.quickflat.make_figure(vx, with_curvature=True,
                                               with_colorbar=True)
            fig.suptitle(title, fontsize=14)
            out = os.path.join(FIGURES_DIR, f"{fname}.png")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            log.info(f"  Figure saved: {out}")

    except Exception as e:
        log.warning(f"  Pycortex figure generation failed: {e}")

    log.info("\nScript 06 complete.")
    log.info(f"  Stats    : {STATS_DIR}")
    log.info(f"  CIFTIs   : {CIFTI_59K_STATS}")
    log.info(f"  Figures  : {FIGURES_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Group-level statistics, Cohen's d maps, and flatmap figures."
    )
    parser.add_argument("--min_subjects", type=int, default=1,
                        help="Minimum subjects required (default 1 for pilot)")
    parser.add_argument("--no_pycortex", action="store_true",
                        help="Skip pycortex flatmap figures (for headless servers)")
    parser.add_argument("--roi_a",       default=_DEFAULT_ROI_A,
                        help="First ROI name (default: %(default)s)")
    parser.add_argument("--roi_b",       default=_DEFAULT_ROI_B,
                        help="Second ROI name (default: %(default)s)")
    parser.add_argument("--output_base", default=_DEFAULT_OUTPUT_BASE,
                        help="Root output directory (default: %(default)s)")
    parser.add_argument("--hcp_dir",     default=_DEFAULT_HCP_DIR,
                        help="HCP S1200 group-average atlas directory (default: %(default)s)")
    parser.add_argument("--cifti_dir",   default=_DEFAULT_CIFTI_DIR,
                        help="Directory containing subject CIFTI dtseries files "
                             "(default: %(default)s)")
    args = parser.parse_args()

    ROI_A         = args.roi_a
    ROI_B         = args.roi_b
    HCP_DIR       = args.hcp_dir
    CIFTI_RAW_DIR = args.cifti_dir
    OUTPUT_ROOT     = f"{args.output_base}/{ROI_A}_{ROI_B}"
    SUBJECTS_DIR    = f"{OUTPUT_ROOT}/subjects"
    GROUP_DIR       = f"{OUTPUT_ROOT}/group"
    STATS_DIR       = f"{GROUP_DIR}/stats"
    CIFTI_59K_DIR   = f"{GROUP_DIR}/cifti_59k"
    CIFTI_32K_DIR   = f"{GROUP_DIR}/cifti_32k"
    CIFTI_59K_STATS = f"{CIFTI_59K_DIR}/stats"
    CIFTI_32K_STATS = f"{CIFTI_32K_DIR}/stats"
    FIGURES_DIR     = f"{GROUP_DIR}/figures"
    TEMPLATE_32K    = f"{HCP_DIR}/S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii"
    SPHERE_59K_L    = f"{HCP_DIR}/S1200.L.sphere.59k_fs_LR.surf.gii"
    SPHERE_59K_R    = f"{HCP_DIR}/S1200.R.sphere.59k_fs_LR.surf.gii"
    SPHERE_32K_L    = f"{HCP_DIR}/S1200.L.sphere.32k_fs_LR.surf.gii"
    SPHERE_32K_R    = f"{HCP_DIR}/S1200.R.sphere.32k_fs_LR.surf.gii"
    PYCORTEX_FILESTORE = HCP_DIR

    for d in [STATS_DIR, CIFTI_59K_STATS, CIFTI_32K_STATS, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)

    main(args.min_subjects, args.no_pycortex)
