"""
connectivity/seed_connectivity.py
==================================
Seed-based functional connectivity from a border-defined ROI.

Takes lh/rh .border files (e.g. produced by rsa/draw_rsa_borders.py with
--threshold-mode percentile), converts them to a grayordinate ROI mask via
`wb_command -border-to-rois`, averages the ROI timeseries into a single seed
timeseries, and Pearson-correlates it against every cortical grayordinate.

Modes (--mode)
---------------
group_average : single continuous group-average CIFTI dtseries.
                Default input: data/preprocessed/average_sub/raw/
                group_average_raw_cortex_59k.dtseries.nii
                (already z-scored per run, no SG/PSC/GSR — "raw" pipeline).

per_subject   : loop over each subject's 4 raw 7T movie runs
                (data/individual-59k/*.dtseries.nii), reusing
                preprocess_individual.preprocess_subject() to apply the same
                "raw" (z-score per run only) pipeline, concatenate the 4
                runs, compute a per-subject connectivity map, then average
                across subjects (Fisher z, mean, inverse Fisher z).
                NOT run automatically — this loops over ~176 subjects x 4
                runs and is a long job. Invoke explicitly when ready.

Window (--window)
------------------
full  (default) : use the entire continuous timeseries (rest + stimulus +
                the small excluded end-credit-like gaps).
rest            : official inter-clip REST blocks only (from
                data/HCP_7T_Movie_Clip_Timing.csv, block_label "0"), minus
                the first --rest-trim-sec seconds of each block (HRF
                carry-over from the preceding clip's offset). See
                official_timing.py.
stim            : the filtered "meaningful AV content" clip windows only
                (data/movie_timing.csv — the same 18 windows used for
                RSA/encoding). Comparing `rest` vs `stim` connectivity for
                the same seed is a useful check for whether the seed's
                connectivity profile is stimulus-driven or intrinsic.
                Global time in both timing sources is cumulative across
                runs and TR = 1s, so seconds index directly into the
                concatenated timeseries (see official_timing.py for the
                run-local -> global conversion of the official sheet).

Usage
-----
python connectivity/seed_connectivity.py \\
    --border-lh outputs/rsa/raw/group_average/pe-av-small-16-frame_av/k100_delay5s_bin5s_skip5s_spearman/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_top5pct_lh.border \\
    --border-rh outputs/rsa/raw/group_average/pe-av-small-16-frame_av/k100_delay5s_bin5s_skip5s_spearman/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_top5pct_rh.border \\
    --mode group_average \\
    --roi-name pe-av-small-16-frame_av_top5pct \\
    --out-dir outputs/connectivity/pe-av-small-16-frame_av_top5pct/group_average
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cifti_io import get_bm_axis, get_cortex_vertex_indices, save_cifti_map
from rsa.glasser import load_glasser_parcels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_BASE = Path("/home/amin/Research/Representation/Movie")
_GROUP_AVG_CIFTI = (_BASE / "data/preprocessed/average_sub/raw"
                     "/group_average_raw_cortex_59k.dtseries.nii")
_RAW_DIR = _BASE / "data/individual-59k"
_SUBJECTS_LIST = _BASE / "data/subjects.txt"
_EXCLUDED_LIST = _BASE / "data/excluded.txt"
_L_SURF = (_BASE / "data/HCP_S1200_GroupAvg_v1/GroupAverage_59k"
           "/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii")
_R_SURF = (_BASE / "data/HCP_S1200_GroupAvg_v1/GroupAverage_59k"
           "/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii")
# NOTE: data/HCP Data/Q1-Q6_RelatedValidation210_{LEFT,RIGHT}.label.gii are the
# 32k-mesh label files (32,492 verts/hem) — WRONG resolution for this 59k-mesh
# pipeline (59,292 verts/hem full surface). Use the combined 59k dlabel instead
# (same one used by rsa/analysis.sh for the ranked_report.csv parcel tables).
_GLASSER_DLABEL_59K = (_BASE / "data/HCP_S1200_GroupAvg_v1"
                        "/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii")
_TIMING_CSV = _BASE / "data/movie_timing.csv"
_WB = Path("/opt/workbench/bin_linux64/wb_command")
_TR = 1.0


# =============================================================================
# Border -> grayordinate ROI mask
# =============================================================================

def _border_to_surface_mask(border_path: str, surface_path: str,
                             workbench: str, n_surf_verts: int) -> np.ndarray:
    """Run wb_command -border-to-rois and return a (n_surf_verts,) bool mask."""
    tmp = Path(border_path).with_suffix(".roi.func.gii")
    subprocess.run(
        [workbench, "-border-to-rois", surface_path, border_path, str(tmp),
         "-include-border"],
        check=True, capture_output=True, text=True,
    )
    gii = nib.load(str(tmp))
    # -border-to-rois writes one column per border in the file; OR them together.
    mask = np.zeros(n_surf_verts, dtype=bool)
    for darr in gii.darrays:
        mask |= (darr.data > 0.5)
    tmp.unlink(missing_ok=True)
    return mask


def load_roi_grayord_mask(border_lh: str, border_rh: str, template_cifti: str,
                           left_surface: str, right_surface: str,
                           workbench: str) -> tuple:
    """Convert lh/rh .border files to a single grayordinate ROI mask.

    Returns
    -------
    roi_mask  : (n_grayord,) bool, in [L..., R...] grayordinate order
    n_left    : int — number of left-hemisphere grayordinates (for reference)
    """
    bm_axis = get_bm_axis(template_cifti)
    left_idx, right_idx = get_cortex_vertex_indices(bm_axis)
    n_surf = 59292  # full-surface vertex count per hemisphere (incl. medial wall)

    mask_L_full = _border_to_surface_mask(border_lh, left_surface, workbench, n_surf)
    mask_R_full = _border_to_surface_mask(border_rh, right_surface, workbench, n_surf)

    mask_L = mask_L_full[left_idx]
    mask_R = mask_R_full[right_idx]
    roi_mask = np.concatenate([mask_L, mask_R])
    log.info(f"ROI grayordinates: L={int(mask_L.sum())}  R={int(mask_R.sum())}  "
             f"total={int(roi_mask.sum())} / {len(roi_mask)}")
    return roi_mask, len(mask_L)


# =============================================================================
# Time-window masks: official rest blocks vs filtered "meaningful AV" stimulus
# =============================================================================
#
# Two independent timing sources, reconciled in official_timing.py:
#   rest window : data/HCP_7T_Movie_Clip_Timing.csv, block_label == "0"
#                 (official 20s inter-clip REST blocks; run-local time,
#                 converted here to global cumulative time via run_trs)
#   stim window : data/movie_timing.csv (the 18 clips actually used for
#                 RSA/encoding — already excludes rest AND some end-credit
#                 tails, e.g. official clip 1.1 runs 20-264.04s but
#                 movie_timing.csv's video1 stops at 232s)
# The small residual gaps between a clip's official end and its
# movie_timing.csv end (end-credit-like content) fall in neither mask and
# are excluded from both "rest" and "stim" windows.

def build_official_rest_mask(n_trs_total: int, official_timing_csv: str,
                              run_trs_npy: str, tr: float,
                              rest_trim_sec: float) -> np.ndarray:
    """Boolean (n_trs_total,) mask, True = usable official rest-block TR.

    Drops the first rest_trim_sec seconds of each rest block (HRF
    carry-over from the preceding clip's offset).
    """
    from official_timing import load_official_timing, add_global_time
    df = add_global_time(load_official_timing(official_timing_csv), run_trs_npy)
    rest = df[df["is_rest"]]
    mask = np.zeros(n_trs_total, dtype=bool)
    trim_trs = int(np.round(rest_trim_sec / tr))
    for _, row in rest.iterrows():
        start_tr = int(np.round(row["global_start"] / tr)) + trim_trs
        end_tr = int(np.round(row["global_end"] / tr))
        if start_tr < end_tr:
            mask[max(0, start_tr):min(n_trs_total, end_tr)] = True
    log.info(f"Official rest window: {int(mask.sum())} / {n_trs_total} TRs "
             f"({100.0 * mask.mean():.1f}%) after trimming {rest_trim_sec}s "
             f"HRF carry-over per block [{len(rest)} blocks, source: {official_timing_csv}]")
    return mask


def build_filtered_stim_mask(n_trs_total: int, timing_csv: str, tr: float) -> np.ndarray:
    """Boolean (n_trs_total,) mask, True = TR inside a movie_timing.csv clip
    window (the same "meaningful AV content" windows used for RSA/encoding).
    onset_sec/end_sec are GLOBAL cumulative seconds (TR=1s convention)."""
    timing = pd.read_csv(timing_csv)
    mask = np.zeros(n_trs_total, dtype=bool)
    for _, row in timing.iterrows():
        onset_tr = int(np.round(row["onset_sec"] / tr))
        end_tr = int(np.round(row["end_sec"] / tr))
        mask[max(0, onset_tr):min(n_trs_total, end_tr)] = True
    log.info(f"Filtered stimulus window: {int(mask.sum())} / {n_trs_total} TRs "
             f"({100.0 * mask.mean():.1f}%) [{len(timing)} clips, source: {timing_csv}]")
    return mask


# =============================================================================
# Seed connectivity core
# =============================================================================

def compute_seed_connectivity(data: np.ndarray, roi_mask: np.ndarray) -> tuple:
    """Pearson-correlate the ROI-mean timeseries against every grayordinate.

    Parameters
    ----------
    data     : (n_grayord, T) float32 — already time-windowed if applicable
    roi_mask : (n_grayord,) bool

    Returns
    -------
    r_map    : (n_grayord,) float32
    seed_ts  : (T,) float32
    """
    seed_ts = data[roi_mask].mean(axis=0)
    seed_c = seed_ts - seed_ts.mean()
    seed_ss = np.sqrt((seed_c ** 2).sum())

    data_c = data - data.mean(axis=1, keepdims=True)
    num = data_c @ seed_c
    denom = np.sqrt((data_c ** 2).sum(axis=1)) * seed_ss
    with np.errstate(invalid="ignore", divide="ignore"):
        r_map = np.where(denom > 0, num / denom, 0.0).astype(np.float32)
    return r_map, seed_ts.astype(np.float32)


def summarize_and_save(r_map: np.ndarray, roi_mask: np.ndarray, template_cifti: str,
                        out_dir: Path, roi_name: str, window: str, n_subjects: int = 1) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{roi_name}_{window}"

    np.save(out_dir / f"connectivity_{tag}.npy", r_map)
    save_cifti_map(r_map, template_cifti, str(out_dir / f"connectivity_{tag}.dscalar.nii"),
                    map_name=f"connectivity_{tag}")

    outside = ~roi_mask
    peak_incl = float(r_map.max())
    peak_excl = float(r_map[outside].max())
    argpeak_excl = np.where(outside)[0][np.argmax(r_map[outside])]
    mean_pos = float(r_map[outside][r_map[outside] > 0].mean())
    pct_pos = float((r_map[outside] > 0).mean() * 100)

    # Glasser parcels (combined 59k dlabel — same one rsa/analysis.sh uses)
    bm_axis = get_bm_axis(template_cifti)
    rows = []
    try:
        all_parcels = load_glasser_parcels(str(_GLASSER_DLABEL_59K), bm_axis)
        for name, indices in all_parcels.items():
            idx = np.array(indices)
            idx = idx[outside[idx]] if len(idx) else idx  # exclude seed-overlapping verts
            if len(idx) == 0:
                continue
            rows.append({"parcel": name, "mean_r": float(r_map[idx].mean()),
                         "n_vertices": int(len(idx)),
                         "seed_overlap_frac": float(1 - len(idx) / len(indices))})
        report = pd.DataFrame(rows).sort_values("mean_r", ascending=False).reset_index(drop=True)
        report["rank"] = range(1, len(report) + 1)
        report.to_csv(out_dir / f"connectivity_{tag}_ranked_parcels.csv", index=False)
    except Exception as exc:
        log.warning(f"Glasser parcel summary failed: {exc}")
        report = pd.DataFrame()

    with open(out_dir / f"connectivity_{tag}_summary.txt", "w") as f:
        summary = (
            f"ROI: {roi_name}   window: {window}   n_subjects_avg: {n_subjects}\n"
            f"ROI grayordinates: {int(roi_mask.sum())} / {len(roi_mask)}\n"
            f"Peak r (incl. seed, trivially ~1): {peak_incl:.4f}\n"
            f"Peak r (excl. seed ROI)          : {peak_excl:.4f}  at grayordinate {argpeak_excl}\n"
            f"Mean r (excl. seed, positive only): {mean_pos:.4f}\n"
            f"%% positive (excl. seed)          : {pct_pos:.1f}%%\n"
        )
        f.write(summary)
    log.info(summary)
    if len(report):
        log.info(f"Top-10 connected parcels:\n{report.head(10).to_string(index=False)}")


def _apply_window(data: np.ndarray, window: str, args, run_trs_source) -> np.ndarray:
    """window: full | rest | stim.  run_trs_source: npy path or array."""
    if window == "full":
        return data
    if window == "rest":
        t_mask = build_official_rest_mask(data.shape[1], args.official_timing_csv,
                                          run_trs_source, _TR, args.rest_trim_sec)
    elif window == "stim":
        t_mask = build_filtered_stim_mask(data.shape[1], args.timing_csv, _TR)
    else:
        raise ValueError(f"Unknown window: {window}")
    return data[:, t_mask]


# =============================================================================
# Group-average mode
# =============================================================================

def run_group_average(args, roi_mask: np.ndarray) -> None:
    log.info(f"Loading group-average CIFTI: {args.group_average_cifti}")
    img = nib.load(args.group_average_cifti)
    data = img.get_fdata(dtype=np.float32).T  # (n_grayord, T)
    log.info(f"  data shape: {data.shape}")

    data = _apply_window(data, args.window, args, args.run_trs_npy)

    r_map, seed_ts = compute_seed_connectivity(data, roi_mask)
    summarize_and_save(r_map, roi_mask, args.group_average_cifti,
                        Path(args.out_dir), args.roi_name, args.window)


# =============================================================================
# Per-subject mode (scaffolded — not run by default; long job)
# =============================================================================

def run_per_subject(args, roi_mask: np.ndarray) -> None:
    import types
    from preprocess_individual import preprocess_subject, load_subjects

    subjects = load_subjects(args.subjects_list)
    if args.excluded_list and Path(args.excluded_list).exists():
        excluded = set(load_subjects(args.excluded_list))
        subjects = [s for s in subjects if s not in excluded]
    log.info(f"Per-subject connectivity: {len(subjects)} subjects "
             f"(this is a long job — {len(subjects)} x 4 runs)")

    prep_args = types.SimpleNamespace(sg_filter=False, psc=False, gsr=False)  # "raw" pipeline
    fisher_z_sum = None
    n_ok = 0
    per_subject_dir = Path(args.out_dir) / "per_subject_maps"
    per_subject_dir.mkdir(parents=True, exist_ok=True)

    for sub in subjects:
        try:
            data, bm_axis, run_trs = preprocess_subject(sub, Path(args.raw_dir), _TR, prep_args)
        except Exception as exc:
            log.warning(f"[{sub}] skipped: {exc}")
            continue

        data = _apply_window(data, args.window, args, run_trs)

        r_map, _ = compute_seed_connectivity(data, roi_mask)
        np.save(per_subject_dir / f"{sub}_connectivity_{args.roi_name}_{args.window}.npy", r_map)

        z = np.arctanh(np.clip(r_map, -0.999999, 0.999999))
        fisher_z_sum = z if fisher_z_sum is None else fisher_z_sum + z
        n_ok += 1
        del data
        log.info(f"[{sub}] done ({n_ok}/{len(subjects)})")

    if n_ok == 0:
        log.error("No subjects processed successfully.")
        return

    mean_r_map = np.tanh(fisher_z_sum / n_ok)
    summarize_and_save(mean_r_map, roi_mask, args.template_cifti_per_subject,
                        Path(args.out_dir), args.roi_name, args.window, n_subjects=n_ok)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--border-lh", required=True, dest="border_lh")
    p.add_argument("--border-rh", required=True, dest="border_rh")
    p.add_argument("--roi-name", required=True, dest="roi_name")
    p.add_argument("--mode", choices=["group_average", "per_subject"], required=True)
    p.add_argument("--window", choices=["full", "rest", "stim"], default="full",
                   help="full=entire run; rest=official inter-clip rest blocks; "
                        "stim=filtered movie_timing.csv clip windows only.")
    p.add_argument("--timing-csv", default=str(_TIMING_CSV), dest="timing_csv",
                   help="Filtered 18-clip 'meaningful AV content' windows (--window stim).")
    p.add_argument("--official-timing-csv", default=str(_BASE / "data/HCP_7T_Movie_Clip_Timing.csv"),
                   dest="official_timing_csv",
                   help="Official per-run clip/rest block sheet (--window rest).")
    p.add_argument("--run-trs-npy", dest="run_trs_npy",
                   default=str(_BASE / "data/preprocessed/average_sub/raw/group_average_raw_run_trs.npy"),
                   help="[group_average mode] per-run TR counts, for official rest-block "
                        "run-local -> global time conversion.")
    p.add_argument("--rest-trim-sec", type=float, default=6.0, dest="rest_trim_sec",
                   help="Seconds of HRF carry-over to drop from the start of each official "
                        "rest block when --window rest.")
    p.add_argument("--left-surface", default=str(_L_SURF), dest="left_surface")
    p.add_argument("--right-surface", default=str(_R_SURF), dest="right_surface")
    p.add_argument("--workbench", default=str(_WB))
    p.add_argument("--out-dir", required=True, dest="out_dir")

    # group_average mode
    p.add_argument("--group-average-cifti", default=str(_GROUP_AVG_CIFTI),
                   dest="group_average_cifti")

    # per_subject mode
    p.add_argument("--raw-dir", default=str(_RAW_DIR), dest="raw_dir")
    p.add_argument("--subjects-list", default=str(_SUBJECTS_LIST), dest="subjects_list")
    p.add_argument("--excluded-list", default=str(_EXCLUDED_LIST), dest="excluded_list")
    p.add_argument("--template-cifti-per-subject", default=str(_GROUP_AVG_CIFTI),
                   dest="template_cifti_per_subject",
                   help="Any CIFTI sharing the per-subject BrainModelAxis, used only "
                        "for the output dscalar header.")
    return p.parse_args()


def main():
    args = parse_args()
    template_for_roi = (args.group_average_cifti if args.mode == "group_average"
                        else args.template_cifti_per_subject)

    roi_mask, n_left = load_roi_grayord_mask(
        args.border_lh, args.border_rh, template_for_roi,
        args.left_surface, args.right_surface, args.workbench,
    )
    if roi_mask.sum() == 0:
        log.error("ROI mask is empty — check border files / template alignment.")
        sys.exit(1)

    if args.mode == "group_average":
        run_group_average(args, roi_mask)
    else:
        run_per_subject(args, roi_mask)

    log.info("Done.")


if __name__ == "__main__":
    main()
