"""
encoding/group_stats.py
============================
Aggregate per-subject encoding Pearson-r maps into group-level statistics.

For each grayordinate vertex across subjects:
  1. Stack per-subject r maps → (n_subjects, n_grayords)
  2. Fisher-z transform:  Z = arctanh(r)
  3. One-sample t-test (H₀: mean Z = 0) → one-tailed t/p for r > 0
  4. Mean r = tanh(mean Z)
  5. Cohen's d = mean(Z) / std(Z, ddof=1)
  6. BH-FDR correction on the one-tailed p-values
  7. TFCE + sign-flipping permutation test (n_permutations, default 5000)
     → FWE-corrected significance mask at p < α (95th percentile of null)
  8. Cluster border: wb_command -metric-rois-to-border traces the contour of
     surviving clusters on the midthickness surface → per-hemisphere .border files

TFCE is computed via mne.stats.permutation_cluster_1samp_test with
threshold=dict(start=0, step=0.2), using full surface adjacency from the
59k midthickness GIFTI meshes.

Output: multi-map CIFTI dscalar with 7 continuous maps:
  mean_r | cohens_d | t_stat | sigmap_uncorr
  | sigmap_fdr | tfce_stat | sigmap_tfce_fwe

Binary masks are saved as separate standalone dscalar files only:
  group_stats_{n}subs_fdr_mask.dscalar.nii
  group_stats_{n}subs_tfce_fwe_mask.dscalar.nii

Note on sigmap_fdr vs sigmap_uncorr: when signal is dense (>99% of vertices
truly positive), BH-FDR p-values converge to uncorrected p-values and the two
maps appear visually identical. This is expected, not a bug. The FDR mask is
still meaningful as a binary significance indicator. For spatial gradients, use
mean_r, cohens_d, or tfce_stat.

And saves the TFCE null distribution for later patching:
  group_stats_{n}subs_h0_null.npy

Skip logic
----------
If the output CIFTI already exists:
  - All 3 significance maps present (sigmap_fdr, fdr_mask, sigmap_tfce_fwe)
    → skip entirely (nothing to do).
  - CIFTI present but sigmap_tfce_fwe missing and h0_null.npy present
    → patch sigmap_tfce_fwe from the saved null without rerunning TFCE.
  - CIFTI present but sigmap_fdr / fdr_mask missing → patch FDR from stored
    t_stat without rerunning the expensive TFCE permutation test.
  - sigmap_tfce_fwe missing and no h0_null.npy → delete the combined CIFTI
    and rerun to regenerate it.
If the output CIFTI does not exist → full computation.

Border contours of the FWE-significant clusters are written as separate
Workbench .border files (one per hemisphere) alongside the dscalar.

Also writes summary.json to the same directory.

Usage:
  python encoding/group_stats.py \\
    --output-dir    /path/to/encoding/outputs \\
    --model         pe-av-small-16-frame \\
    --modality      av \\
    --bin-sec       5.0 \\
    --skip-sec      5.0 \\
    --delay-sec     5.0 \\
    --hrf-mode      delay \\
    --normalize \\
    --template-cifti /path/to/group_average_raw_cortex_59k.dtseries.nii \\
    --left-surface   /path/to/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii \\
    --right-surface  /path/to/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii \\
    --workbench      /opt/workbench/bin_linux64/wb_command \\
    --n-permutations 5000 \\
    --n-jobs -1
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from mne.stats import permutation_cluster_1samp_test
from nibabel.gifti import GiftiDataArray, GiftiImage
from scipy import stats
from scipy.sparse import block_diag as sp_block_diag
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cifti_io import (
    get_bm_axis,
    get_combined_map_names,
    get_cortex_vertex_indices,
    merge_into_combined,
    save_cifti_map,
    save_cifti_multimap,
)

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
        description="Aggregate per-subject encoding r maps to group-level statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir",       required=True,
                   help="Root encoding output directory (contains per-subject subdirs).")
    p.add_argument("--model",            required=True)
    p.add_argument("--modality",         required=True,
                   help="Modality key (e.g. v, a, av).")
    p.add_argument("--bin-sec",          type=float, required=True)
    p.add_argument("--skip-sec",         type=float, default=None, dest="skip_sec",
                   help="Window stride in seconds (default: bin-sec, i.e. no overlap).")
    p.add_argument("--delay-sec",        type=float, default=5.0)
    p.add_argument("--hrf",              action="store_true",
                   help="Set if embeddings were HRF-convolved (changes config label).")
    p.add_argument("--normalize",        action="store_true",
                   help="Set if per-run z-score normalization was used (changes config label).")
    p.add_argument("--template-cifti",   required=True,
                   help="59k CIFTI whose BrainModelAxis defines grayordinate space.")
    p.add_argument("--left-surface",     required=True,
                   help="Left 59k midthickness .surf.gii (for TFCE adjacency).")
    p.add_argument("--right-surface",    required=True,
                   help="Right 59k midthickness .surf.gii (for TFCE adjacency).")
    p.add_argument("--workbench",        default=None,
                   help="Path to wb_command. Used to create .border files from the "
                        "FWE-significant cluster mask. If omitted, border files are skipped.")
    p.add_argument("--alpha",            type=float, default=0.05,
                   help="FWE / FDR significance threshold.")
    p.add_argument("--min-cluster-size", type=int, default=10,
                   help="Minimum cluster size (vertices) to include in FWE mask.")
    p.add_argument("--n-permutations",   type=int, default=5000,
                   help="Number of sign-flip permutations for TFCE null distribution.")
    p.add_argument("--n-jobs",           type=int, default=-1,
                   help="Parallel jobs for permutation test (-1 = all cores).")
    return p.parse_args()


# =============================================================================
# Config label (must match encoding.py _config_label())
# =============================================================================

def _config_label(args) -> str:
    parts = [
        "hrf" if args.hrf else f"delay{args.delay_sec:.0f}s",
        "norm" if args.normalize else "demean",
        f"bin{args.bin_sec:.0f}s_skip{args.skip_sec:.0f}s",
    ]
    return "_".join(parts)


# =============================================================================
# Surface helpers
# =============================================================================

def _load_surface(surf_path: str) -> tuple[np.ndarray, int]:
    """Return (faces, n_verts) from a GIFTI surface file."""
    surf   = nib.load(surf_path)
    n_verts = surf.darrays[0].data.shape[0]
    faces   = surf.darrays[1].data.astype(np.int32)
    return faces, n_verts


def _surface_adjacency(faces: np.ndarray,
                        vertex_idx: np.ndarray,
                        n_cifti_verts: int) -> csr_matrix:
    """Sparse adjacency matrix in CIFTI grayordinate space."""
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
# Cluster filtering
# =============================================================================

def _filter_clusters(sig_mask: np.ndarray,
                     adj: csr_matrix,
                     min_cluster_size: int) -> np.ndarray:
    """Filter a binary significance mask to retain only clusters ≥ min_cluster_size."""
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


# =============================================================================
# Workbench border file
# =============================================================================

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
             surface_path, str(tmp_metric), 'tfce_fwe', out_border_path],
            check=True, capture_output=True, text=True,
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

def _compute_tfce_sigmap(
    tfce_stat_f64: np.ndarray,
    h0: np.ndarray,
    mean_r: np.ndarray,
    n_perms: int,
) -> np.ndarray:
    """Compute sign(mean_r) × −log₁₀(p_fwe_vertex) from the TFCE null distribution.

    p_fwe[v] = fraction of permutations where max-TFCE ≥ observed TFCE[v],
    with a pseudo-count of 1/(n_perms+1) to avoid p=0.
    """
    n_grays = len(tfce_stat_f64)
    p_fwe = np.empty(n_grays, dtype=np.float32)
    chunk = 5000
    for s in range(0, n_grays, chunk):
        e = min(s + chunk, n_grays)
        exceed = (h0[:, None] >= tfce_stat_f64[None, s:e]).sum(axis=0)
        p_fwe[s:e] = exceed.astype(np.float32) / n_perms

    min_p = np.float32(1.0 / (n_perms + 1))
    p_fwe = np.maximum(p_fwe, min_p)
    eps = np.finfo(np.float32).tiny
    return (np.sign(mean_r) *
            (-np.log10(np.maximum(p_fwe, eps)))).astype(np.float32)


def _compute_fdr_maps(
    p_uncorr: np.ndarray,
    mean_r: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply BH-FDR correction; return (sigmap_fdr, fdr_mask, n_significant).

    sigmap_fdr = sign(mean_r) × −log₁₀(p_fdr)  (same convention as sigmap_uncorr)
    fdr_mask   = float32 binary, 1 where BH-corrected p < alpha
    """
    p_fdr = stats.false_discovery_control(p_uncorr, method="bh")
    eps = np.finfo(np.float32).tiny
    sigmap_fdr = (np.sign(mean_r) *
                  (-np.log10(np.maximum(p_fdr, eps)))).astype(np.float32)
    fdr_mask = (p_fdr < alpha).astype(np.float32)
    return sigmap_fdr, fdr_mask, int(fdr_mask.sum())


def _patch_fdr_maps(
    out_path: Path,
    n_subs: int,
    alpha: float,
    template_cifti: str,
    min_cluster_size: int = 10,
    workbench: str | None = None,
    left_surface: str | None = None,
    right_surface: str | None = None,
) -> dict:
    """Append sigmap_fdr and fdr_mask to an existing group-stats CIFTI.

    Also saves standalone mask CIFTIs and, if surface paths + workbench are
    provided, creates FDR border files.

    Returns dict with n_sig_fdr, fdr_border_lh, fdr_border_rh.
    """
    existing = get_combined_map_names(str(out_path))
    img  = nib.load(str(out_path))
    data = img.get_fdata(dtype=np.float32)   # (n_maps, n_grays)
    del img

    t_vals = data[existing.index("t_stat")].astype(np.float64)
    mean_r = data[existing.index("mean_r")].astype(np.float32)
    n_grays = t_vals.shape[0]

    p_two    = stats.t.sf(np.abs(t_vals), df=n_subs - 1) * 2.0
    p_uncorr = np.where(t_vals > 0,
                        p_two / 2.0,
                        1.0 - p_two / 2.0).astype(np.float32)

    sigmap_fdr, fdr_mask, n_sig = _compute_fdr_maps(p_uncorr, mean_r, alpha)

    merge_into_combined(sigmap_fdr, "sigmap_fdr", out_path, template_cifti)
    log.info(f"FDR-significant vertices (p<{alpha}): {n_sig:,}")

    # ── sigmap_tfce_fwe from saved null distribution (if available) ───────────
    h0_path = out_path.parent / out_path.name.replace(".dscalar.nii", "_h0_null.npy")
    h0_path_alt = out_path.parent / (out_path.stem.replace(".dscalar", "") + "_h0_null.npy")
    for hp in (h0_path, h0_path_alt):
        if hp.exists():
            h0 = np.load(str(hp)).astype(np.float64)
            if "tfce_stat" in existing:
                tfce_stat_arr = data[existing.index("tfce_stat")].astype(np.float64)
                mean_r_arr    = data[existing.index("mean_r")].astype(np.float32)
                sig_tfce = _compute_tfce_sigmap(tfce_stat_arr, h0, mean_r_arr, len(h0))
                merge_into_combined(sig_tfce, "sigmap_tfce_fwe", out_path, template_cifti)
                log.info(f"sigmap_tfce_fwe patched from {hp.name}")
            break
    else:
        log.info("h0_null.npy not found — sigmap_tfce_fwe requires a full rerun.")

    # ── Standalone mask CIFTIs ────────────────────────────────────────────────
    stem = out_path.stem.replace(".dscalar", "")
    fdr_mask_path = out_path.parent / f"{stem}_fdr_mask.dscalar.nii"
    save_cifti_map(fdr_mask, template_cifti, str(fdr_mask_path), "fdr_mask")
    log.info(f"Saved FDR mask: {fdr_mask_path.name}")

    if "tfce_fwe_mask" in existing:
        tfce_fwe_mask = data[existing.index("tfce_fwe_mask")]
        tfce_mask_path = out_path.parent / f"{stem}_tfce_fwe_mask.dscalar.nii"
        save_cifti_map(tfce_fwe_mask, template_cifti, str(tfce_mask_path), "tfce_fwe_mask")
        log.info(f"Saved TFCE FWE mask: {tfce_mask_path.name}")

    # ── FDR border files ──────────────────────────────────────────────────────
    fdr_border_lh = fdr_border_rh = None
    if workbench and left_surface and right_surface:
        bm_axis = get_bm_axis(template_cifti)
        lh_idx, rh_idx = get_cortex_vertex_indices(bm_axis)
        n_left = len(lh_idx)
        faces_lh, n_verts_lh = _load_surface(left_surface)
        faces_rh, n_verts_rh = _load_surface(right_surface)
        adj_lh = _surface_adjacency(faces_lh, lh_idx, n_left)
        adj_rh = _surface_adjacency(faces_rh, rh_idx, n_grays - n_left)

        fdr_cm_lh = _filter_clusters(fdr_mask[:n_left] > 0.5, adj_lh, min_cluster_size)
        fdr_cm_rh = _filter_clusters(fdr_mask[n_left:] > 0.5, adj_rh, min_cluster_size)
        n_cluster_fdr = int(fdr_cm_lh.sum()) + int(fdr_cm_rh.sum())
        coverage_fdr  = n_cluster_fdr / n_grays if n_grays > 0 else 0.0

        if n_cluster_fdr == 0:
            log.info("No FDR cluster vertices; skipping FDR border files.")
        elif coverage_fdr > 0.90:
            log.warning(f"FDR mask covers {coverage_fdr*100:.1f}% of cortex — skipping border files.")
        else:
            fdr_border_lh = str(out_path.parent / f"{stem}_fdr_lh.border")
            fdr_border_rh = str(out_path.parent / f"{stem}_fdr_rh.border")
            _write_border_file(fdr_cm_lh, lh_idx, n_verts_lh, left_surface, fdr_border_lh, workbench)
            _write_border_file(fdr_cm_rh, rh_idx, n_verts_rh, right_surface, fdr_border_rh, workbench)
            log.info(f"  FDR LH border: {fdr_border_lh}")
            log.info(f"  FDR RH border: {fdr_border_rh}")

    return {"n_sig_fdr": n_sig, "fdr_border_lh": fdr_border_lh, "fdr_border_rh": fdr_border_rh}


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.skip_sec is None:
        args.skip_sec = args.bin_sec
    config = _config_label(args)

    out_dir = (Path(args.output_dir) / "groupstats" /
               f"{args.model}_{args.modality}" / config)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect per-subject r maps ────────────────────────────────────────────
    fname = f"encoding_r_{args.modality}.dscalar.nii"

    subject_r_files = sorted(
        f for f in Path(args.output_dir).glob(
            f"*/{args.model}/{config}/{fname}"
        )
        if f.parts[-4] not in ("group_average", "groupstats")
    )

    if not subject_r_files:
        log.error(
            f"No per-subject r maps found matching:\n"
            f"  {Path(args.output_dir)}/*/{args.model}/{config}/{fname}\n"
            f"Run per-subject encoding first (analysis.sh persubject)."
        )
        sys.exit(1)

    log.info(f"Found {len(subject_r_files)} per-subject r maps")

    r_maps = []
    for f in subject_r_files:
        img  = nib.load(str(f))
        data = img.get_fdata(dtype=np.float32)
        # dscalar: (1, n_grays) — squeeze to 1-D
        r_maps.append(data.squeeze())
        log.info(f"  Loaded: {f.parts[-4]}  "
                 f"shape={data.squeeze().shape}  max={data.max():.4f}")

    r_stack = np.stack(r_maps, axis=0)   # (n_subjects, n_grayords)
    n_subs, n_grays = r_stack.shape
    log.info(f"Stacked: {r_stack.shape}")

    # ── Output path (needed for skip logic) ──────────────────────────────────
    out_path = out_dir / f"group_stats_{n_subs}subs.dscalar.nii"

    # ── Skip / patch logic ────────────────────────────────────────────────────
    existing_map_names = get_combined_map_names(str(out_path))
    all_maps_present = (
        "sigmap_fdr"      in existing_map_names and
        "sigmap_tfce_fwe" in existing_map_names
    )
    if all_maps_present:
        log.info(f"All significance maps present in {out_path.name} — nothing to do.")
        return

    if out_path.exists() and existing_map_names:
        log.info(
            f"Existing CIFTI lacks FDR maps ({existing_map_names}) — "
            f"patching FDR without rerunning TFCE ..."
        )
        patch = _patch_fdr_maps(
            out_path, n_subs, args.alpha, args.template_cifti,
            min_cluster_size=args.min_cluster_size,
            workbench=args.workbench,
            left_surface=args.left_surface,
            right_surface=args.right_surface,
        )
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary["n_sig_fdr"]     = patch["n_sig_fdr"]
            summary["fdr_border_lh"] = patch["fdr_border_lh"]
            summary["fdr_border_rh"] = patch["fdr_border_rh"]
            summary_path.write_text(json.dumps(summary, indent=2))
            log.info(f"Summary updated: {summary_path}")
        log.info("FDR patch complete.")
        return

    # ── Fisher-z transform ────────────────────────────────────────────────────
    Z = np.arctanh(np.clip(r_stack, -1 + 1e-7, 1 - 1e-7)).astype(np.float64)

    # ── One-sample t-test (H₀: mean Z = 0) ───────────────────────────────────
    t_vals, p_two = stats.ttest_1samp(Z, popmean=0.0, axis=0)
    t_vals = t_vals.astype(np.float32)

    # One-tailed p for r > 0
    p_uncorr = np.where(t_vals > 0,
                        p_two / 2.0,
                        1.0 - p_two / 2.0).astype(np.float32)

    # ── Summary statistics ────────────────────────────────────────────────────
    mean_z = Z.mean(axis=0).astype(np.float32)
    std_z  = Z.std(axis=0, ddof=1).astype(np.float32)
    std_z[std_z == 0] = 1e-10

    mean_r   = np.tanh(mean_z).astype(np.float32)
    cohens_d = (mean_z / std_z).astype(np.float32)

    eps = np.finfo(np.float32).tiny
    sigmap_uncorr = (np.sign(mean_r) *
                     (-np.log10(np.maximum(p_uncorr, eps)))).astype(np.float32)

    # ── BH-FDR correction ─────────────────────────────────────────────────────
    sigmap_fdr, fdr_mask, n_sig_fdr = _compute_fdr_maps(p_uncorr, mean_r, args.alpha)

    log.info(f"mean_r range: [{mean_r.min():.4f}, {mean_r.max():.4f}]")
    log.info(f"Cohen's d range: [{cohens_d.min():.4f}, {cohens_d.max():.4f}]")
    log.info(f"Uncorrected p<{args.alpha}: {(p_uncorr < args.alpha).sum():,} / {n_grays:,}")
    log.info(f"BH-FDR p<{args.alpha}: {n_sig_fdr:,} / {n_grays:,}")

    # ── Surface adjacency for TFCE ────────────────────────────────────────────
    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    n_left = len(left_indices)

    faces_lh, n_verts_lh = _load_surface(args.left_surface)
    faces_rh, n_verts_rh = _load_surface(args.right_surface)

    log.info(f"Building surface adjacency (LH: {n_left} verts, "
             f"RH: {n_grays - n_left} verts) ...")
    adj_lh = _surface_adjacency(faces_lh, left_indices,  n_left)
    adj_rh = _surface_adjacency(faces_rh, right_indices, n_grays - n_left)
    log.info("  Adjacency built.")

    adj_combined = sp_block_diag([adj_lh, adj_rh], format="coo")

    # ── TFCE permutation test ─────────────────────────────────────────────────
    log.info(
        f"Running TFCE permutation test: n_permutations={args.n_permutations}, "
        f"n_jobs={args.n_jobs}  (runtime: 1–4 h for 59k verts)"
    )
    tfce_stat, _, _, h0 = permutation_cluster_1samp_test(
        Z,
        threshold=dict(start=0, step=0.2),
        n_permutations=args.n_permutations,
        adjacency=adj_combined,
        tail=1,
        n_jobs=args.n_jobs,
        seed=42,
        out_type="indices",
        verbose=False,
    )
    tfce_stat_f64 = np.asarray(tfce_stat, dtype=np.float64)
    h0            = np.asarray(h0,        dtype=np.float64)
    n_perms_actual = len(h0)

    h0_path = out_dir / f"group_stats_{n_subs}subs_h0_null.npy"
    np.save(str(h0_path), h0)
    log.info(f"Saved TFCE null distribution: {h0_path.name}")

    if len(h0) == 0:
        log.warning("TFCE null distribution is empty — no positive t-values found. "
                    "Setting threshold to inf; no vertices will be significant.")
        tfce_thresh = np.inf
    else:
        tfce_thresh = float(np.percentile(h0, 100.0 * (1.0 - args.alpha)))
    log.info(f"TFCE FWE threshold (p<{args.alpha}): {tfce_thresh:.4f}")

    n_sig_tfce = int((tfce_stat_f64 >= tfce_thresh).sum())
    log.info(f"TFCE-FWE significant vertices: {n_sig_tfce:,} / {n_grays:,}")

    # ── Cluster mask: filter out small clusters ───────────────────────────────
    sig_lh_tfce = tfce_stat_f64[:n_left] >= tfce_thresh
    sig_rh_tfce = tfce_stat_f64[n_left:] >= tfce_thresh

    cm_lh = _filter_clusters(sig_lh_tfce, adj_lh, args.min_cluster_size)
    cm_rh = _filter_clusters(sig_rh_tfce, adj_rh, args.min_cluster_size)

    tfce_fwe_mask = np.concatenate([cm_lh, cm_rh])

    tfce_stat = tfce_stat_f64.astype(np.float32)

    log.info(f"TFCE-FWE clusters (≥{args.min_cluster_size} verts): "
             f"{int(tfce_fwe_mask.sum()):,} verts")

    # ── TFCE sigmap: sign(mean_r) × −log₁₀(p_fwe_vertex) ────────────────────
    if n_perms_actual > 0:
        sigmap_tfce_fwe = _compute_tfce_sigmap(
            tfce_stat_f64, h0, mean_r, n_perms_actual)
        log.info(f"sigmap_tfce_fwe range: "
                 f"[{sigmap_tfce_fwe.min():.3f}, {sigmap_tfce_fwe.max():.3f}]")
    else:
        sigmap_tfce_fwe = np.zeros(n_grays, dtype=np.float32)

    # ── Workbench border files ────────────────────────────────────────────────
    border_lh_path = border_rh_path = None
    n_cluster = int(tfce_fwe_mask.sum())
    coverage  = n_cluster / n_grays if n_grays > 0 else 0.0
    if not args.workbench:
        log.info("--workbench not provided; skipping border file creation.")
    elif n_cluster == 0:
        log.info("No significant cluster vertices; skipping border file creation.")
    elif coverage > 0.90:
        log.warning(
            f"Cluster covers {coverage*100:.1f}% of cortex ({n_cluster:,}/{n_grays:,} verts). "
            f"The only boundary is the medial-wall edge. Skipping .border file creation."
        )
    else:
        log.info("Creating Workbench border files ...")
        border_lh_path = str(out_dir / f"group_stats_{n_subs}subs_lh.border")
        border_rh_path = str(out_dir / f"group_stats_{n_subs}subs_rh.border")
        _write_border_file(cm_lh, left_indices,  n_verts_lh,
                           args.left_surface,  border_lh_path, args.workbench)
        _write_border_file(cm_rh, right_indices, n_verts_rh,
                           args.right_surface, border_rh_path, args.workbench)
        log.info(f"  LH border: {border_lh_path}")
        log.info(f"  RH border: {border_rh_path}")

    # ── FDR border files ──────────────────────────────────────────────────────
    fdr_border_lh = fdr_border_rh = None
    fdr_cm_lh = _filter_clusters(fdr_mask[:n_left] > 0.5, adj_lh, args.min_cluster_size)
    fdr_cm_rh = _filter_clusters(fdr_mask[n_left:] > 0.5, adj_rh, args.min_cluster_size)
    n_cluster_fdr = int(fdr_cm_lh.sum()) + int(fdr_cm_rh.sum())
    coverage_fdr  = n_cluster_fdr / n_grays if n_grays > 0 else 0.0
    if not args.workbench:
        pass   # already logged above
    elif n_cluster_fdr == 0:
        log.info("No FDR cluster vertices; skipping FDR border files.")
    elif coverage_fdr > 0.90:
        log.warning(
            f"FDR mask covers {coverage_fdr*100:.1f}% of cortex "
            f"({n_cluster_fdr:,}/{n_grays:,} verts) — skipping FDR border files."
        )
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

    # ── Save multi-map CIFTI ──────────────────────────────────────────────────
    maps = np.stack([
        mean_r,
        cohens_d,
        t_vals,
        sigmap_uncorr,
        sigmap_fdr,
        tfce_stat,
        sigmap_tfce_fwe,
    ], axis=0)   # (7, n_grayords)

    map_names = [
        "mean_r",
        "cohens_d",
        "t_stat",
        "sigmap_uncorr",
        "sigmap_fdr",
        "tfce_stat",
        "sigmap_tfce_fwe",
    ]

    save_cifti_multimap(maps, map_names, args.template_cifti, str(out_path))
    log.info(f"Saved: {out_path}")

    # ── Standalone mask CIFTIs ────────────────────────────────────────────────
    tfce_mask_path = out_dir / f"group_stats_{n_subs}subs_tfce_fwe_mask.dscalar.nii"
    save_cifti_map(tfce_fwe_mask, args.template_cifti, str(tfce_mask_path), "tfce_fwe_mask")
    log.info(f"Saved TFCE FWE mask: {tfce_mask_path.name}")

    fdr_mask_path = out_dir / f"group_stats_{n_subs}subs_fdr_mask.dscalar.nii"
    save_cifti_map(fdr_mask, args.template_cifti, str(fdr_mask_path), "fdr_mask")
    log.info(f"Saved FDR mask: {fdr_mask_path.name}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "model":                  args.model,
        "modality":               args.modality,
        "config":                 config,
        "n_subjects":             n_subs,
        "n_grayordinates":        n_grays,
        "alpha":                  args.alpha,
        "min_cluster_size":       args.min_cluster_size,
        "n_permutations":         args.n_permutations,
        "tfce_step":              0.2,
        "tfce_fwe_threshold":     tfce_thresh,
        "n_sig_uncorr":           int((p_uncorr < args.alpha).sum()),
        "n_sig_tfce_fwe":         n_sig_tfce,
        "n_cluster_verts_tfce":   int(tfce_fwe_mask.sum()),
        "n_sig_fdr":              n_sig_fdr,
        "max_cohens_d":           float(cohens_d.max()),
        "max_t_stat":             float(t_vals.max()),
        "max_tfce_stat":          float(tfce_stat.max()),
        "max_sigmap_uncorr":      float(sigmap_uncorr.max()),
        "max_sigmap_fdr":         float(sigmap_fdr.max()),
        "max_sigmap_tfce_fwe":    float(sigmap_tfce_fwe.max()),
        "mean_r_range":           [float(mean_r.min()), float(mean_r.max())],
        "border_tfce_lh":         border_lh_path,
        "border_tfce_rh":         border_rh_path,
        "border_fdr_lh":          fdr_border_lh,
        "border_fdr_rh":          fdr_border_rh,
        "subjects":               [f.parts[-4] for f in subject_r_files],
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Summary: {summary_path}")
    log.info(
        f"Done.  n={n_subs} subjects  "
        f"tfce_fwe={summary['n_cluster_verts_tfce']:,} cluster verts  "
        f"fdr={n_sig_fdr:,} verts"
    )


if __name__ == "__main__":
    main()
