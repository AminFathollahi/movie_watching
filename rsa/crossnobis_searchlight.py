"""
rsa/crossnobis_searchlight.py
==============================
Crossvalidated Mahalanobis (crossnobis) searchlight RSA using the 4 repeated
end-of-run clips in the HCP 7T movie dataset.

Background
----------
Each fMRI run ends with the same ~82-second clip (video5/9/14/18 in
movie_timing.csv).  Having N=4 independent measurements of the same stimulus
allows unbiased distance estimation via the crossnobis formula:

    d̂(i,j) = 1/(N·(N-1)) · Σ_{m≠n} (xᵐᵢ−xᵐⱼ)ᵀ Σ⁻¹ (xⁿᵢ−xⁿⱼ)

where xᵐᵢ is the searchlight-neighbourhood response to bin i in run m, and
Σ is the noise covariance estimated by Ledoit-Wolf shrinkage.  The expected
value of d̂ equals the true neural distance (noise cancels because runs are
independent), unlike the standard Pearson-based distance which is inflated by
noise.  Ref: Walther et al. 2016; Schütt et al. 2023 §5.1.1.

The resulting crossnobis RDM (K×K, where K = bins per clip) is compared to the
model RDM using ρ_a (Kendall's τ_a), the bias-robust comparator recommended
by Schütt et al. 2023 §3.5.

This is a per-subject analysis.  Run it on every subject then aggregate with
group_stats.py (point --method rho_a at the crossnobis output directory).

Usage (disk mode)
-----------------
  python rsa/crossnobis_searchlight.py \\
      --preprocessed-dir /path/to/preprocessed \\
      --fmri-suffix raw \\
      --timing-csv /path/to/movie_timing.csv \\
      --embeddings-dir /path/to/embeddings \\
      --template-cifti /path/to/template.dscalar.nii \\
      --left-surface /path/to/L.surf.gii \\
      --right-surface /path/to/R.surf.gii \\
      --workbench /path/to/wb_command \\
      --output-dir /path/to/output \\
      --subject 100610 \\
      --model pe-av-small-16-frame --modality av \\
      --k 100 --bin-sec 5.0 --delay-sec 5.0 --tr 1.0

Output
------
  {model}_{modality}/k{k}_{delay_tag}_bin{bin}s_skip{skip}s_rho_a/
    crossnobis_rho_a_k{k}_{delay_tag}_bin{bin}s_skip{skip}s.npy   (n_verts,) float32
    crossnobis_rho_a_k{k}_{delay_tag}_bin{bin}s_skip{skip}s.dscalar.nii
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.shared.rsa_utils import load_fmri_cifti, spm_hrf
from rsa.searchlight import get_neighbors, _rho_sigmap, _fdr_sigmap
from cifti_io import get_bm_axis, get_cortex_vertex_indices, save_cifti_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

N_JOBS = int(os.environ.get("_RSA_N_JOBS", -1))


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Crossnobis searchlight RSA on the repeated end-of-run clips "
            "(Walther 2016; Schütt et al. 2023 §5.1.1)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preprocessed-dir", required=True, dest="preprocessed_dir",
                   help="Directory of continuous cleaned CIFTIs from preprocess_individual.py.")
    p.add_argument("--fmri-suffix", default="raw", dest="fmri_suffix",
                   help="Filename suffix encoding preprocessing.")
    p.add_argument("--timing-csv", required=True)
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--template-cifti", required=True)
    p.add_argument("--left-surface", required=True)
    p.add_argument("--right-surface", required=True)
    p.add_argument("--workbench", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subject", default="group_average")
    p.add_argument("--model", required=True)
    p.add_argument("--modality", required=True,
                   choices=["v", "a", "av", "at", "vt", "avt", "t", "caption_t", "transcript_t", "event_t", "transcript_avt", "event_avt"])
    p.add_argument("--k", type=int, required=True,
                   help="Searchlight neighbourhood size (vertices).")
    p.add_argument("--bin-sec", type=float, required=True)
    p.add_argument("--skip-sec", type=float, default=None, dest="skip_sec",
                   help="Window stride in seconds (default: bin-sec, no overlap).")
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--hrf", action="store_true",
                   help="Convolve model embeddings with SPM HRF.")
    p.add_argument("--tr", type=float, required=True)
    p.add_argument("--geodesic-cache-dir", default=None, dest="geodesic_cache_dir")
    p.add_argument("--n-jobs", type=int, default=N_JOBS, dest="n_jobs",
                   help="Joblib parallel workers for CPU searchlight (-1 = all CPUs).")
    return p.parse_args()


# =============================================================================
# Identify the repeated clips
# =============================================================================

def find_repeated_clips(timing_df: pd.DataFrame) -> list[dict]:
    """Return the last clip per run from timing_df.

    Validates that all runs end with clips of the same duration (sanity check
    that they really are the same stimulus repeated).
    """
    run_col = "run" if "run" in timing_df.columns else "run_id"
    clips = []
    for run_id, run_df in timing_df.groupby(run_col, sort=True):
        last = run_df.iloc[-1]
        clips.append({
            "run_id":       int(run_id),
            "video_id":     last["video_id"],
            "onset_sec":    float(last["onset_sec"]),
            "end_sec":      float(last["end_sec"]),
            "duration_sec": float(last["duration_sec"]),
        })
    durations = [c["duration_sec"] for c in clips]
    if len(set(durations)) > 1:
        raise ValueError(
            f"Last clips do not all have the same duration: {durations}. "
            "Crossnobis requires repeated measurements of the same stimulus."
        )
    log.info(
        f"Repeated clips: "
        + ", ".join(f"{c['video_id']} (run {c['run_id']}, {c['duration_sec']:.0f}s)" for c in clips)
    )
    return clips


# =============================================================================
# Brain data extraction
# =============================================================================

def extract_repeated_bins(
    fmri: np.ndarray,
    run_trs: np.ndarray,
    repeated_clips: list[dict],
    bin_sec: float,
    tr: float,
    delay_sec: float,
    skip_sec: float,
) -> np.ndarray:
    """Extract per-run binned fMRI for the repeated clips.

    Parameters
    ----------
    fmri          : (n_verts, T_total) float32 — continuous preprocessed CIFTI
    run_trs       : (n_runs,) int — TRs per run
    repeated_clips: output of find_repeated_clips()
    bin_sec / tr / delay_sec / skip_sec : binning parameters

    Returns
    -------
    (N_runs, K_bins, n_verts) float32
    """
    bin_trs  = max(1, int(np.round(bin_sec  / tr)))
    skip_trs = max(1, int(np.round(skip_sec / tr)))
    n_verts  = fmri.shape[0]
    n_runs   = len(repeated_clips)

    # K_bins from the shared duration
    dur    = repeated_clips[0]["duration_sec"]
    k_bins = max(0, int(np.floor((dur - bin_sec) / skip_sec)) + 1) if dur >= bin_sec else 0
    if k_bins < 2:
        raise ValueError(f"Only {k_bins} bins in the repeated clip ({dur}s, bin={bin_sec}s). "
                         "Need at least 2 to form an RDM.")

    clips_binned = np.zeros((n_runs, k_bins, n_verts), dtype=np.float32)
    run_offset = 0

    for m, clip in enumerate(repeated_clips):
        n_trs_run    = int(run_trs[m])
        run_start_sec = float(np.sum(run_trs[:m])) * tr
        within_run_onset = clip["onset_sec"] - run_start_sec
        start_tr = int(np.round((within_run_onset + delay_sec) / tr))

        run_data = fmri[:, run_offset : run_offset + n_trs_run]  # (n_verts, n_trs)

        for k in range(k_bins):
            w_start = start_tr + k * skip_trs
            w_end   = w_start + bin_trs
            if w_end > n_trs_run:
                log.warning(
                    f"Run {clip['run_id']}: bin {k} [{w_start},{w_end}) exceeds "
                    f"run boundary ({n_trs_run} TRs); truncating K_bins to {k}."
                )
                clips_binned = clips_binned[:, :k, :]
                return clips_binned
            clips_binned[m, k, :] = run_data[:, w_start:w_end].mean(axis=1)

        run_offset += n_trs_run

    return clips_binned


# =============================================================================
# Model embeddings extraction
# =============================================================================

def extract_model_rdm(
    emb_path: str,
    timing_df: pd.DataFrame,
    target_video_id,
    bin_sec: float,
    tr: float,
    delay_sec: float,
    skip_sec: float,
    run_trs: np.ndarray,
    hrf: bool = False,
) -> np.ndarray:
    """Extract the model RDM for the repeated clip.

    Since all 4 repetitions are the same stimulus, we use the first occurrence
    (target_video_id) to build the K×K model RDM.

    Returns
    -------
    (K_bins, K_bins) float64 — correlation-distance model RDM
    """
    bin_trs  = max(1, int(np.round(bin_sec  / tr)))
    skip_trs = max(1, int(np.round(skip_sec / tr)))

    embeddings = np.load(emb_path).astype(np.float64)
    hrf_kernel = spm_hrf(bin_sec) if hrf else None

    run_col = "run" if "run" in timing_df.columns else "run_id"
    seg_idx = 0

    for run_id, run_df in timing_df.groupby(run_col, sort=True):
        run_idx = list(timing_df.groupby(run_col, sort=True).groups.keys()).index(run_id)
        n_trs_run     = int(run_trs[run_idx])
        run_start_sec = float(np.sum(run_trs[:run_idx])) * tr

        for _, row in run_df.iterrows():
            dur    = row["duration_sec"]
            n_wins = (max(0, int(np.floor((dur - bin_sec) / skip_sec)) + 1)
                      if dur >= bin_sec else 0)

            if row["video_id"] == target_video_id and n_wins > 0:
                # Apply same boundary truncation as preprocess_fmri
                within_run_onset = row["onset_sec"] - run_start_sec
                start_tr = int(np.round((within_run_onset + delay_sec) / tr))
                actual_wins = 0
                if 0 <= start_tr < n_trs_run:
                    for i in range(n_wins):
                        w_s = start_tr + i * skip_trs
                        w_e = w_s + bin_trs
                        if w_s >= n_trs_run or w_e > n_trs_run:
                            break
                        actual_wins += 1

                if actual_wins == 0:
                    seg_idx += n_wins
                    continue

                seg = embeddings[seg_idx : seg_idx + actual_wins]
                if hrf_kernel is not None:
                    from scipy.signal import fftconvolve
                    out = np.zeros_like(seg)
                    for j in range(seg.shape[1]):
                        conv = fftconvolve(seg[:, j], hrf_kernel, mode="full")
                        out[:, j] = conv[:len(seg)]
                    seg = out

                # Build model RDM: (K, K) correlation distance
                mu   = seg.mean(axis=1, keepdims=True)
                ec   = seg - mu
                nrms = np.linalg.norm(ec, axis=1, keepdims=True)
                nrms[nrms < 1e-10] = 1.0
                en  = ec / nrms
                sim = en @ en.T
                rdm = np.clip(1.0 - sim, 0.0, 2.0)
                np.fill_diagonal(rdm, 0.0)
                return rdm

            seg_idx += n_wins

    raise ValueError(f"video_id '{target_video_id}' not found in timing_df.")


# =============================================================================
# Per-vertex crossnobis
# =============================================================================

def _crossnobis_vertex(
    surf_v: int,
    clips: np.ndarray,           # (N_runs, K_bins, n_hem_verts) float32
    model_rdm_tril: np.ndarray,  # (n_pairs,) float64 — lower-triangle of model RDM
    neighbors: np.ndarray,       # (n_surf_verts, k) int32
    vertex_to_col: np.ndarray,   # (n_surf_verts,) int32 — -1 for medial wall
    tril_idx: tuple,             # np.tril_indices(K_bins, k=-1)
) -> float:
    """Compute crossnobis RSA at one searchlight vertex.

    Steps
    -----
    1. Gather neighbourhood data: (N_runs, K_bins, n_nb)
    2. Estimate noise residuals by subtracting per-bin run-mean
    3. Fit Ledoit-Wolf precision matrix on the (N*K, n_nb) residual matrix
    4. Compute crossnobis distance for every bin pair using the identity:
         Σ_{m≠n} diff_m @ Σ⁻¹ @ diff_n
         = (Σ_m diff_m) @ Σ⁻¹ @ (Σ_m diff_m) - Σ_m (diff_m @ Σ⁻¹ @ diff_m)
       which avoids an explicit double loop over runs.
    5. Compare the crossnobis RDM to the model RDM with ρ_a (Kendall's τ_a).
    """
    neighbor_surf = neighbors[surf_v]
    neighbor_cols = vertex_to_col[neighbor_surf]
    neighbor_cols = neighbor_cols[neighbor_cols >= 0]
    n_nb = len(neighbor_cols)
    if n_nb < 2:
        return 0.0

    # (N_runs, K_bins, n_nb)
    hood = clips[:, :, neighbor_cols].astype(np.float64)
    N_runs, K_bins, _ = hood.shape

    # Noise residuals: subtract per-bin mean across runs
    # Shape: (N_runs, K_bins, n_nb) → flatten to (N_runs*K_bins, n_nb)
    residuals = (hood - hood.mean(axis=0, keepdims=True)).reshape(-1, n_nb)

    # Ledoit-Wolf precision matrix — handles n_samples < n_features gracefully
    try:
        lw        = LedoitWolf(assume_centered=True).fit(residuals)
        precision = lw.precision_  # (n_nb, n_nb) float64
    except Exception:
        return 0.0

    # Vectorised crossnobis distances
    # diff[m, pair] = hood[m, i, :] - hood[m, j, :]   shape: (N_runs, n_pairs, n_nb)
    rows, cols   = tril_idx
    diff         = hood[:, rows, :] - hood[:, cols, :]  # (N_runs, n_pairs, n_nb)
    sum_diff     = diff.sum(axis=0)                      # (n_pairs, n_nb)

    # Cross term: (Σ_m diff_m) Σ⁻¹ (Σ_m diff_m) per pair
    Pp            = sum_diff @ precision                  # (n_pairs, n_nb)
    cross_term    = (Pp * sum_diff).sum(axis=1)           # (n_pairs,)

    # Self term: Σ_m diff_m Σ⁻¹ diff_m per pair
    Pp_m          = diff @ precision                      # (N_runs, n_pairs, n_nb)
    self_term     = (Pp_m * diff).sum(axis=2).sum(axis=0)  # (n_pairs,)

    xnobis = (cross_term - self_term) / (N_runs * (N_runs - 1))  # (n_pairs,)

    tau, _ = kendalltau(xnobis, model_rdm_tril, method="auto")
    return float(tau) if np.isfinite(tau) else 0.0


# =============================================================================
# Full hemisphere searchlight
# =============================================================================

def run_crossnobis_hemisphere(
    clips_hem: np.ndarray,      # (N_runs, K_bins, n_hem_verts) float32
    model_rdm_tril: np.ndarray, # (n_pairs,) float64
    neighbors: np.ndarray,      # (n_surf_verts, k) int32
    surface_indices: np.ndarray,# (n_hem_verts,) int32 — surface vertex per grayordinate
    vertex_to_col: np.ndarray,  # (n_surf_verts,) int32
    n_jobs: int = -1,
) -> np.ndarray:
    """Run crossnobis searchlight over all grayordinate vertices in one hemisphere."""
    K_bins   = clips_hem.shape[1]
    tril_idx = np.tril_indices(K_bins, k=-1)
    n_verts  = clips_hem.shape[2]

    log.info(f"  Running crossnobis searchlight: {n_verts:,} vertices, {K_bins} bins, "
             f"{clips_hem.shape[0]} runs (n_jobs={n_jobs})")

    corr_map = np.array(
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_crossnobis_vertex)(
                int(surface_indices[v]),
                clips_hem,
                model_rdm_tril,
                neighbors,
                vertex_to_col,
                tril_idx,
            )
            for v in range(n_verts)
        ),
        dtype=np.float32,
    )
    return corr_map


# =============================================================================
# Core analysis
# =============================================================================

def _run_analysis(args, fmri: np.ndarray, run_trs: np.ndarray,
                  timing_df: pd.DataFrame, out_root: Path):

    bin_sec_int = int(args.bin_sec)
    skip_int    = int(args.skip_sec)
    delay_tag   = f"delay{int(args.delay_sec)}s"
    stem        = (f"crossnobis_rho_a_k{args.k}_{delay_tag}"
                   f"_bin{bin_sec_int}s_skip{skip_int}s")
    npy_out     = out_root / f"{stem}.npy"
    cifti_out   = out_root / f"{stem}.dscalar.nii"

    if npy_out.exists() and cifti_out.exists():
        log.info(f"Output already exists — skipping: {npy_out.name}")
        return

    # ── Identify repeated clips ───────────────────────────────────────────────
    repeated_clips = find_repeated_clips(timing_df)
    n_runs = len(repeated_clips)

    # ── Extract brain data for each run's repeated clip ───────────────────────
    log.info("Extracting per-run binned data for repeated clips ...")
    clips_all = extract_repeated_bins(
        fmri, run_trs, repeated_clips,
        args.bin_sec, args.tr, args.delay_sec, args.skip_sec,
    )
    N_runs, K_bins, n_verts_total = clips_all.shape
    n_pairs = K_bins * (K_bins - 1) // 2
    log.info(f"  clips_all: {clips_all.shape}  "
             f"K_bins={K_bins}  n_pairs={n_pairs}  N_runs={N_runs}")

    if N_runs < 2:
        raise RuntimeError(f"Need ≥2 runs for crossnobis; got {N_runs}.")

    # ── Build model RDM ───────────────────────────────────────────────────────
    emb_file = (Path(args.embeddings_dir) / args.model /
                f"bin{bin_sec_int}s_skip{skip_int}s" /
                f"{args.model}_{args.modality}.npy")
    log.info(f"Building model RDM from embeddings: {emb_file.name}")
    target_video_id = repeated_clips[0]["video_id"]

    model_rdm = extract_model_rdm(
        str(emb_file), timing_df, target_video_id,
        args.bin_sec, args.tr, args.delay_sec, args.skip_sec,
        run_trs, hrf=args.hrf,
    )
    tril_idx        = np.tril_indices(K_bins, k=-1)
    model_rdm_tril  = model_rdm[tril_idx].astype(np.float64)
    log.info(f"  Model RDM: {model_rdm.shape}  "
             f"range=[{model_rdm_tril.min():.3f}, {model_rdm_tril.max():.3f}]")

    # ── Per-hemisphere searchlight ────────────────────────────────────────────
    bm_axis = get_bm_axis(args.template_cifti)
    left_indices, right_indices = get_cortex_vertex_indices(bm_axis)
    n_left = len(left_indices)

    if args.geodesic_cache_dir:
        cache_dir = Path(args.geodesic_cache_dir)
    else:
        cache_dir = Path(args.output_dir) / "_geodesic_cache"

    corr_full = np.zeros(n_verts_total, dtype=np.float32)
    offset    = 0

    for hem, surf_path, surf_indices in [
        ("left",  args.left_surface,  left_indices),
        ("right", args.right_surface, right_indices),
    ]:
        n_hem = len(surf_indices)
        log.info(f"  Hemisphere: {hem}  ({n_hem:,} vertices)")

        neighbors = get_neighbors(
            surf_path, args.workbench, args.subject, hem, args.k, cache_dir,
        )
        surf_indices_i32 = surf_indices.astype(np.int32)
        n_surf_verts     = neighbors.shape[0]
        vertex_to_col    = np.full(n_surf_verts, -1, dtype=np.int32)
        vertex_to_col[surf_indices_i32] = np.arange(n_hem, dtype=np.int32)

        clips_hem = clips_all[:, :, offset : offset + n_hem]  # (N_runs, K, n_hem)

        corr_hem = run_crossnobis_hemisphere(
            clips_hem, model_rdm_tril, neighbors,
            surf_indices_i32, vertex_to_col,
            n_jobs=args.n_jobs,
        )
        corr_full[offset : offset + n_hem] = corr_hem

        del neighbors, vertex_to_col, clips_hem, corr_hem
        gc.collect()
        offset += n_hem

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_root.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_out), corr_full)
    log.info(f"  Saved .npy: {npy_out.name}  "
             f"mean={corr_full.mean():.4f}  max={corr_full.max():.4f}")

    save_cifti_map(corr_full, args.template_cifti, str(cifti_out), "crossnobis_rho_a")
    log.info(f"  Saved CIFTI: {cifti_out.name}")

    # Quick significance map using rho approximation (analytical; for group
    # inference use group_stats.py across per-subject maps).
    p_uncorr, sigmap = _rho_sigmap(corr_full, n_pairs)
    sigmap_path = out_root / f"{stem}_sigmap.dscalar.nii"
    save_cifti_map(sigmap, args.template_cifti, str(sigmap_path), "crossnobis_sigmap")
    log.info(f"  Saved sigmap: {sigmap_path.name}")

    del fmri, clips_all
    gc.collect()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.skip_sec is None:
        args.skip_sec = args.bin_sec

    timing_df = pd.read_csv(args.timing_csv)

    cifti_path = (Path(args.preprocessed_dir) /
                  f"{args.subject}_{args.fmri_suffix}_cortex_59k.dtseries.nii")
    trs_path   = (Path(args.preprocessed_dir) /
                  f"{args.subject}_{args.fmri_suffix}_run_trs.npy")

    log.info(f"Crossnobis searchlight RSA: {args.subject} / {args.model} / {args.modality}")
    log.info(f"  fMRI: {cifti_path}")

    fmri    = load_fmri_cifti(str(cifti_path))
    run_trs = np.load(str(trs_path))
    log.info(f"  fMRI loaded: {fmri.shape}  run_trs: {run_trs.tolist()}")

    if args.subject == "group_average":
        sub_dir = Path(args.output_dir) / "group_average"
    else:
        sub_dir = Path(args.output_dir) / "subject_data" / args.subject

    bin_sec_int = int(args.bin_sec)
    skip_int    = int(args.skip_sec)
    out_root    = (sub_dir / f"{args.model}_{args.modality}" /
                   f"k{args.k}_delay{int(args.delay_sec)}s"
                   f"_bin{bin_sec_int}s_skip{skip_int}s_rho_a")

    _run_analysis(args, fmri, run_trs, timing_df, out_root)
    log.info("Done.")


if __name__ == "__main__":
    main()
