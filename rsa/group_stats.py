"""
rsa/group_stats.py
======================
Aggregate per-subject searchlight RSA maps into group-level statistics, strictly
following the inference framework of Schütt et al. (2023).

For each vertex across subjects:
  1. Stack per-subject raw ρ maps → (n_subjects, n_grayords)
  2. One-sample t-test on raw ρ (H₀: mean ρ = 0) → one-tailed t/p for ρ > 0
     * Note: Fisher-z is omitted. Inference is performed directly on the
       variance of the performance estimates.
  3. BH-FDR correction on the one-tailed p-values → FDR-corrected sigmap + mask
  4. Cluster border: wb_command traces the contour of the surviving FDR mask
     on the midthickness surface → per-hemisphere .border files

INFERENCE SCOPE
---------------
subject t-test (existing, always run):
    Generalizes to new *subjects*, conditional on the exact movie segments
    used. Valid for ROI localization within this stimulus set.

corrected 2-factor bootstrap (Schütt et al. 2023, Eq. 5; added as extra maps):
    Generalizes to new *subjects AND movie segments* simultaneously.
    Requires --n-blocks > 1 and per-subject block .npy files from
    searchlight.py --n-blocks. Uses non-overlapping temporal blocks as
    a proxy for condition resampling (adapted for naturalistic paradigms
    where full per-bin RDM resampling is not feasible).
    Use this for model-comparison claims that should generalize beyond
    the specific movie content used.

Output: multi-map CIFTI dscalar with 4 continuous maps (always):
  mean_rho  | t_stat | sigmap_uncorr | sigmap_fdr

When block files are present, 3 additional maps are appended:
  mean_rho_c2f | t_c2f | sigmap_c2f

Binary masks are saved as separate standalone dscalar files:
  group_stats_{n}subs_fdr_mask.dscalar.nii
  group_stats_{n}subs_fdr_c2f_mask.dscalar.nii  (if have_blocks)
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.gifti import GiftiDataArray, GiftiImage
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cifti_io import (
    get_bm_axis,
    get_combined_map_names,
    get_cortex_vertex_indices,
    save_cifti_map,
    save_cifti_multimap,
)
from rsa.shared.rsa_utils import corrected_2factor_bootstrap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate per-subject RSA maps to group-level statistics (Schütt et al. 2023).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir",       required=True,
                   help="Root RSA output directory (contains per-subject subdirs).")
    p.add_argument("--model",            required=True)
    p.add_argument("--modality",         required=True, choices=["v", "a", "av", "at", "vt", "avt", "t"])
    p.add_argument("--k",                type=int, required=True)
    p.add_argument("--bin-sec",          type=float, required=True)
    p.add_argument("--skip-sec",         type=float, default=None, dest="skip_sec",
                   help="Window stride in seconds (default: bin-sec).")
    p.add_argument("--delay-sec",        type=float, default=5.0)
    p.add_argument("--method",           required=True, choices=["spearman", "pearson", "rho_a"])
    p.add_argument("--fmri-tag",         required=True,
                   help="Preprocessing tag in per-subject CIFTI filenames.")
    p.add_argument("--template-cifti",   required=True,
                   help="59k CIFTI whose BrainModelAxis defines grayordinate space.")
    p.add_argument("--left-surface",     required=True,
                   help="Left 59k midthickness .surf.gii (for contour tracing).")
    p.add_argument("--right-surface",    required=True,
                   help="Right 59k midthickness .surf.gii (for contour tracing).")
    p.add_argument("--workbench",        default=None,
                   help="Path to wb_command. Used to create .border files.")
    p.add_argument("--alpha",            type=float, default=0.05,
                   help="FDR significance threshold (q < alpha).")
    p.add_argument("--min-cluster-size", type=int, default=10,
                   help="Minimum vertices to form a valid ROI contour in the border file.")
    p.add_argument("--n-blocks",         type=int, default=4, dest="n_blocks",
                   help="Number of temporal blocks expected in per-subject block .npy files "
                        "(from searchlight.py --n-blocks). Set to 1 to skip 2-factor bootstrap.")
    p.add_argument("--n-bootstrap",      type=int, default=2000, dest="n_bootstrap",
                   help="Bootstrap iterations for the corrected 2-factor variance estimate "
                        "(Schütt et al. 2023, Eq. 5).")
    return p.parse_args()


# =============================================================================
# Surface helpers (Retained exclusively for Workbench border drawing)
# =============================================================================

def _load_surface(surf_path: str) -> tuple[np.ndarray, int]:
    surf = nib.load(surf_path)
    n_verts = surf.darrays[0].data.shape[0]
    faces   = surf.darrays[1].data.astype(np.int32)
    return faces, n_verts

def _surface_adjacency(faces: np.ndarray,
                        vertex_idx: np.ndarray,
                        n_cifti_verts: int) -> csr_matrix:
    idx_map = np.full(int(faces.max()) + 1, -1, dtype=np.int32)
    idx_map[vertex_idx] = np.arange(n_cifti_verts, dtype=np.int32)

    f0 = idx_map[faces[:, 0]]
    f1 = idx_map[faces[:, 1]]
    f2 = idx_map[faces[:, 2]]
    valid = (f0 >= 0) & (f1 >= 0) & (f2 >= 0)
    fc = np.column_stack([f0[valid], f1[valid], f2[valid]])

    i = np.concatenate([fc[:, 0], fc[:, 1], fc[:, 2],
                        fc[:, 1], fc[:, 2], fc[:, 0]])
    j = np.concatenate([fc[:, 1], fc[:, 2], fc[:, 0],
                        fc[:, 0], fc[:, 1], fc[:, 2]])
    adj = csr_matrix(
        (np.ones(len(i), dtype=np.uint8), (i, j)),
        shape=(n_cifti_verts, n_cifti_verts),
    )
    adj.data[:] = 1
    return adj


# =============================================================================
# Border & Cluster Filtering
# =============================================================================

def _filter_clusters(sig_mask: np.ndarray,
                     adj: csr_matrix,
                     min_cluster_size: int) -> np.ndarray:
    cluster_mask = np.zeros(len(sig_mask), dtype=np.float32)
    if not sig_mask.any():
        return cluster_mask

    sig_idx = np.where(sig_mask)[0]
    sig_adj = adj[sig_idx][:, sig_idx]
    n_comp, labels = connected_components(sig_adj, directed=False)

    for comp in range(n_comp):
        comp_verts = sig_idx[labels == comp]
        if len(comp_verts) >= min_cluster_size:
            cluster_mask[comp_verts] = 1.0

    return cluster_mask

def _write_border_file(
    cluster_mask: np.ndarray,
    cifti_vertex_indices: np.ndarray,
    n_full_verts: int,
    surface_path: str,
    out_border_path: str,
    workbench: str,
) -> None:
    full_mask = np.zeros(n_full_verts, dtype=np.float32)
    full_mask[cifti_vertex_indices] = cluster_mask

    tmp_metric = Path(out_border_path).with_suffix('.tmp.func.gii')
    arr = GiftiDataArray(data=full_mask, intent=0, datatype='NIFTI_TYPE_FLOAT32')
    nib.save(GiftiImage(darrays=[arr]), str(tmp_metric))

    try:
        subprocess.run(
            [workbench, '-metric-rois-to-border',
             surface_path, str(tmp_metric), 'fdr_roi', out_border_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        log.warning(
            f"wb_command border creation failed for {out_border_path}:\n"
            f"  {e.stderr.strip()}"
        )
    finally:
        tmp_metric.unlink(missing_ok=True)


# =============================================================================
# Significance map helpers
# =============================================================================

def _compute_fdr_maps(
    p_uncorr: np.ndarray,
    mean_rho: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    p_fdr = stats.false_discovery_control(p_uncorr, method="bh")
    eps = np.finfo(np.float32).tiny
    sigmap_fdr = (np.sign(mean_rho) *
                  (-np.log10(np.maximum(p_fdr, eps)))).astype(np.float32)
    fdr_mask = (p_fdr < alpha).astype(np.float32)
    return sigmap_fdr, fdr_mask, int(fdr_mask.sum())


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.skip_sec is None:
        args.skip_sec = args.bin_sec
    bin_sec_int  = int(args.bin_sec)
    skip_int     = int(args.skip_sec)
    delay_tag    = f"delay{int(args.delay_sec)}s"
    config       = f"k{args.k}_{delay_tag}_bin{bin_sec_int}s_skip{skip_int}s_{args.method}"

    out_dir = (Path(args.output_dir) / "group_stats" /
               f"{args.model}_{args.modality}" / config)
    out_dir.mkdir(parents=True, exist_ok=True)

    fname_pattern = (f"rsa_59k_{args.fmri_tag}_k{args.k}_{delay_tag}"
                     f"_bin{bin_sec_int}s_skip{skip_int}s_{args.method}_searchlight.npy")

    model_mod_dir = f"{args.model}_{args.modality}"
    subject_rho_files = sorted(
        Path(args.output_dir).glob(
            f"subject_data/*/{model_mod_dir}/{config}/{fname_pattern}"
        )
    )

    if not subject_rho_files:
        log.error("No per-subject ρ maps found. Run per-subject RSA first.")
        sys.exit(1)

    log.info(f"Found {len(subject_rho_files)} per-subject ρ maps")

    rho_maps = []
    for f in subject_rho_files:
        data = np.load(str(f)).astype(np.float32)
        rho_maps.append(data)

    rho_stack = np.stack(rho_maps, axis=0)
    n_subs, n_grays = rho_stack.shape

    # ── Try to load per-subject block maps for 2-factor bootstrap ────────────
    block_fname_pattern = fname_pattern.replace(
        "_searchlight.npy", f"_searchlight_nblocks{args.n_blocks}.npy"
    )
    block_files = sorted(
        Path(args.output_dir).glob(
            f"subject_data/*/{model_mod_dir}/{config}/{block_fname_pattern}"
        )
    )
    have_blocks = (
        args.n_blocks > 1
        and len(block_files) == len(subject_rho_files)
    )
    if have_blocks:
        log.info(
            f"Block files found ({len(block_files)} subjects, {args.n_blocks} blocks) — "
            f"will run corrected 2-factor bootstrap (Schütt et al. 2023, Eq. 5)"
        )
        block_maps_list = [np.load(str(f)).astype(np.float32) for f in block_files]
        block_stack = np.stack(block_maps_list, axis=0)  # (n_subs, n_blocks, n_grays)
    elif args.n_blocks > 1:
        log.info(
            f"Block files not found for all subjects "
            f"({len(block_files)}/{len(subject_rho_files)}) — "
            f"skipping 2-factor bootstrap. Run searchlight.py --n-blocks {args.n_blocks} first."
        )

    out_path = out_dir / f"group_stats_{n_subs}subs.dscalar.nii"

    existing_map_names = get_combined_map_names(str(out_path)) if out_path.exists() else []
    if "sigmap_fdr" in existing_map_names:
        log.info(f"FDR significance map already present in {out_path.name} — skipping computation.")
        return

    # ── Raw Performance Estimates ─────────────────────────────────────────────
    X = rho_stack.astype(np.float64)
    mean_rho = rho_stack.mean(axis=0).astype(np.float32)

    # ── One-sample t-test (H₀: mean ρ = 0) ────────────────────────────────────
    log.warning("Performing one-sample test against chance (H0: rho = 0). "
                "This evaluates the presence of mutual information for ROI generation, "
                "but does not adjudicate whether this model is the *best* explanation.")
    
    t_vals, p_two = stats.ttest_1samp(X, popmean=0.0, axis=0)
    t_vals = t_vals.astype(np.float32)

    p_uncorr = np.where(t_vals > 0,
                        p_two / 2.0,
                        1.0 - p_two / 2.0).astype(np.float32)

    eps = np.finfo(np.float32).tiny
    sigmap_uncorr = (np.sign(mean_rho) *
                     (-np.log10(np.maximum(p_uncorr, eps)))).astype(np.float32)

    # ── FDR Correction ────────────────────────────────────────────────────────
    sigmap_fdr, fdr_mask, n_sig_fdr = _compute_fdr_maps(p_uncorr, mean_rho, args.alpha)

    log.info(f"mean_rho range: [{mean_rho.min():.4f}, {mean_rho.max():.4f}]")
    log.info(f"Uncorrected p<{args.alpha}: {(p_uncorr < args.alpha).sum():,} / {n_grays:,}")
    log.info(f"BH-FDR p<{args.alpha}: {n_sig_fdr:,} / {n_grays:,}")

    # ── FDR Border Files (For ROI contours in Workbench) ──────────────────────
    fdr_border_lh = fdr_border_rh = None
    
    if args.workbench and args.left_surface and args.right_surface:
        bm_axis = get_bm_axis(args.template_cifti)
        left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
        n_left = len(left_indices)

        faces_lh, n_verts_lh = _load_surface(args.left_surface)
        faces_rh, n_verts_rh = _load_surface(args.right_surface)

        adj_lh = _surface_adjacency(faces_lh, left_indices,  n_left)
        adj_rh = _surface_adjacency(faces_rh, right_indices, n_grays - n_left)

        fdr_cm_lh = _filter_clusters(fdr_mask[:n_left] > 0.5, adj_lh, args.min_cluster_size)
        fdr_cm_rh = _filter_clusters(fdr_mask[n_left:] > 0.5, adj_rh, args.min_cluster_size)
        n_cluster_fdr = int(fdr_cm_lh.sum()) + int(fdr_cm_rh.sum())
        coverage_fdr  = n_cluster_fdr / n_grays if n_grays > 0 else 0.0
        
        if n_cluster_fdr == 0:
            log.info("No FDR cluster vertices; skipping FDR border files.")
        elif coverage_fdr > 0.90:
            log.warning(f"FDR mask covers {coverage_fdr*100:.1f}% of cortex — skipping FDR border files.")
        else:
            log.info("Creating FDR border files ...")
            fdr_border_lh = str(out_dir / f"group_stats_{n_subs}subs_fdr_lh.border")
            fdr_border_rh = str(out_dir / f"group_stats_{n_subs}subs_fdr_rh.border")
            _write_border_file(fdr_cm_lh, left_indices, n_verts_lh,
                               args.left_surface, fdr_border_lh, args.workbench)
            _write_border_file(fdr_cm_rh, right_indices, n_verts_rh,
                               args.right_surface, fdr_border_rh, args.workbench)
            log.info(f"  FDR LH border: {fdr_border_lh}")
            log.info(f"  FDR RH border: {fdr_border_rh}")

    # ── Corrected 2-factor bootstrap (Schütt et al. 2023, Eq. 5) ─────────────
    n_sig_c2f   = 0
    fdr_c2f_mask = None
    if have_blocks:
        log.info(
            f"Running corrected 2-factor bootstrap "
            f"(n_boot={args.n_bootstrap}, n_blocks={args.n_blocks}) ..."
        )
        var_c2f, var_subj_boot, var_block_boot = corrected_2factor_bootstrap(
            rho_stack, block_stack, n_boot=args.n_bootstrap
        )

        mean_rho_c2f = rho_stack.mean(axis=0).astype(np.float32)
        se_c2f = np.sqrt(np.maximum(var_c2f, 0.0)).astype(np.float64)

        # df = min(N_s-1, N_c-1) per Schütt et al. 2023 §5.1.4 (conservative choice)
        df_c2f = max(min(n_subs - 1, args.n_blocks - 1), 1)
        safe_se = np.where(se_c2f > 0, se_c2f, 1.0)  # avoid divide-by-zero; masked below
        t_c2f = np.where(se_c2f > 0, mean_rho_c2f.astype(np.float64) / safe_se, 0.0).astype(np.float32)
        p_two_c2f = stats.t.sf(np.abs(t_c2f.astype(np.float64)), df=df_c2f) * 2.0
        p_c2f = np.where(t_c2f > 0, p_two_c2f / 2.0, 1.0 - p_two_c2f / 2.0).astype(np.float32)

        sigmap_c2f, fdr_c2f_mask, n_sig_c2f = _compute_fdr_maps(p_c2f, mean_rho_c2f, args.alpha)

        log.info(
            f"Corrected 2-factor bootstrap: df={df_c2f}, "
            f"FDR sig verts = {n_sig_c2f:,} / {n_grays:,}"
        )

    # ── Save CIFTI Outputs ────────────────────────────────────────────────────
    maps = [mean_rho, t_vals, sigmap_uncorr, sigmap_fdr]
    map_names = ["mean_rho", "t_stat", "sigmap_uncorr", "sigmap_fdr"]

    if have_blocks:
        maps += [mean_rho_c2f, t_c2f, sigmap_c2f]
        map_names += ["mean_rho_c2f", "t_c2f", "sigmap_c2f"]

    save_cifti_multimap(np.stack(maps, axis=0), map_names, args.template_cifti, str(out_path))
    log.info(f"Saved: {out_path}")

    fdr_mask_path = out_dir / f"group_stats_{n_subs}subs_fdr_mask.dscalar.nii"
    save_cifti_map(fdr_mask, args.template_cifti, str(fdr_mask_path), "fdr_mask")
    log.info(f"Saved FDR mask: {fdr_mask_path.name}")

    if have_blocks and fdr_c2f_mask is not None:
        fdr_c2f_path = out_dir / f"group_stats_{n_subs}subs_fdr_c2f_mask.dscalar.nii"
        save_cifti_map(fdr_c2f_mask, args.template_cifti, str(fdr_c2f_path), "fdr_c2f_mask")
        log.info(f"Saved 2-factor FDR mask: {fdr_c2f_path.name}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "model":                  args.model,
        "modality":               args.modality,
        "config":                 config,
        "fmri_tag":               args.fmri_tag,
        "n_subjects":             n_subs,
        "n_grayordinates":        n_grays,
        "alpha":                  args.alpha,
        "min_cluster_size":       args.min_cluster_size,
        "n_sig_uncorr":           int((p_uncorr < args.alpha).sum()),
        "n_sig_fdr":              n_sig_fdr,
        "max_t_stat":             float(t_vals.max()),
        "max_sigmap_uncorr":      float(sigmap_uncorr.max()),
        "max_sigmap_fdr":         float(sigmap_fdr.max()),
        "mean_rho_range":         [float(mean_rho.min()), float(mean_rho.max())],
        "border_fdr_lh":          fdr_border_lh,
        "border_fdr_rh":          fdr_border_rh,
        "have_blocks":            have_blocks,
        "n_blocks":               args.n_blocks,
        "n_bootstrap":            args.n_bootstrap,
        "n_sig_c2f":              n_sig_c2f,
        "subjects":               [f.parent.parent.parent.name for f in subject_rho_files],
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Summary: {summary_path}")
    log.info(f"Done. n={n_subs} subjects. FDR sig verts = {n_sig_fdr:,}")


if __name__ == "__main__":
    main()