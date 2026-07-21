"""
subcortical/subcortical_rsa.py
================================
Searchlight + nuclei ROI-RSA on subcortical structures. Thin driver: all
geometry-agnostic machinery (binning, embedding alignment, the searchlight
kernel, RDM/correlation, CIFTI I/O, significance maps) is reused verbatim
from rsa/shared/rsa_utils.py, rsa/searchlight.py, and cifti_io.py. The only
new logic is per-structure iteration (subcortical_io.py).

Group-average mode (--subject group_average, the primary first pass):
  1. Build (or load a cached) group-average subcortical timeseries by
     averaging preprocess_subject_subcortical() across --subjects-list.
  2. Bin + align to the model embeddings exactly as the cortical pipeline
     does (rsa_utils.preprocess_fmri / process_model_embeddings).
  3. Run a k=100 within-structure searchlight per anatomical structure,
     concatenate into one subcortical-length rho vector, save + significance
     maps (mirrors rsa/searchlight.py::_save_significance_maps).
  4. Nuclei ROI-RSA (IC/SC/MGN/LGN): one whole-mask RDM per nucleus.

Per-subject mode (--subject <ID>) is the identical code path with fmri =
one subject instead of the running mean; group_stats.py (unmodified) can
then be pointed at the subcortical template for the across-subject t-test.
Per the build order in subcortex.txt, per-subject runs are gated on the
group-average result being promising.
"""

import argparse
import json
import logging
import sys
import types
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rsa"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsa.shared.rsa_utils import (  # noqa: E402
    preprocess_fmri, process_model_embeddings, align_and_assert_bins,
    assert_segment_timing, compute_rdm, correlate_rdms,
)
from cifti_io import get_bm_axis, save_cifti_map, save_cifti_multimap  # noqa: E402
from searchlight import run_searchlight, _rho_sigmap, _fdr_sigmap  # noqa: E402
from subcortical_io import (  # noqa: E402
    SUBCORTICAL_STRUCTURES, STRUCTURE_NEIGHBOR_MODE, NUCLEI_SEEDS,
    struct_slices, build_neighbors, get_cerebellum_neighbors,
    build_nucleus_masks, preprocess_subject_subcortical,
    compute_group_average_subcortical,
)
from preprocess_individual import load_subjects  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

N_JOBS = -1


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--subjects-list", default=None,
                   help="Required for --subject group_average (roster to average).")
    p.add_argument("--subject", default="group_average")
    p.add_argument("--timing-csv", required=True)
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--template-cifti", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--group-average-dir", default=None,
                   help="Where the group-average subcortical dtseries is cached "
                        "(built once, reused across model runs).")
    p.add_argument("--model", required=True)
    p.add_argument("--modality", required=True)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--bin-sec", type=float, required=True)
    p.add_argument("--skip-sec", type=float, default=None)
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--hrf", action="store_true")
    p.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    p.add_argument("--tr", type=float, default=1.0)
    p.add_argument("--nucleus-radius", type=float, default=5.0)
    p.add_argument("--neighbor-cache-dir", default=None)
    p.add_argument("--workbench", default="/opt/workbench/bin_linux64/wb_command")
    p.add_argument("--gpu-batch-size", type=int, default=512)
    prep = p.add_argument_group("streaming preprocessing")
    prep.add_argument("--sg-filter", action="store_true", dest="sg_filter")
    prep.add_argument("--psc", action="store_true")
    prep.add_argument("--gsr", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def _fmri_tag(args) -> str:
    parts = []
    if args.sg_filter: parts.append("sg")
    if args.psc:       parts.append("psc")
    if args.gsr:       parts.append("gsr")
    return "_".join(parts) if parts else "raw"


# =============================================================================
# fMRI source
# =============================================================================

def _get_fmri(args, fmri_tag: str):
    prep_args = types.SimpleNamespace(sg_filter=args.sg_filter, psc=args.psc, gsr=args.gsr)

    if args.subject == "group_average":
        ga_dir = Path(args.group_average_dir or (Path(args.output_dir) / "group_average_cache"))
        ga_cifti = ga_dir / f"group_average_{fmri_tag}_subcortical.dtseries.nii"
        ga_trs = ga_dir / f"group_average_{fmri_tag}_run_trs.npy"
        if ga_cifti.exists() and ga_trs.exists():
            log.info(f"Group-average subcortical timeseries cached: {ga_cifti}")
            img = nib.load(str(ga_cifti))
            data = img.get_fdata(dtype=np.float32).T
            bm_axis = img.header.get_axis(1)
            run_trs = np.load(str(ga_trs))
            return data, bm_axis, run_trs

        if not args.subjects_list:
            raise ValueError("--subjects-list required to build the group average.")
        subjects = load_subjects(args.subjects_list)
        log.info(f"Building group-average subcortical timeseries from {len(subjects)} subjects ...")
        data, bm_axis, run_trs = compute_group_average_subcortical(
            subjects, Path(args.raw_dir), args.tr, prep_args,
            SUBCORTICAL_STRUCTURES, out_path=ga_cifti,
        )
        return data, bm_axis, run_trs

    log.info(f"Streaming preprocessing subject {args.subject} ...")
    return preprocess_subject_subcortical(
        args.subject, Path(args.raw_dir), args.tr, prep_args, SUBCORTICAL_STRUCTURES)


# =============================================================================
# Main analysis
# =============================================================================

def main():
    args = parse_args()
    if args.skip_sec is None:
        args.skip_sec = args.bin_sec
    fmri_tag = _fmri_tag(args)

    bin_int, skip_int, delay_int = int(args.bin_sec), int(args.skip_sec), int(args.delay_sec)
    config = f"k{args.k}_delay{delay_int}s_bin{bin_int}s_skip{skip_int}s_{args.method}"
    sub_dir = (Path(args.output_dir) / "group_average" if args.subject == "group_average"
               else Path(args.output_dir) / "subject_data" / args.subject)
    out_root = sub_dir / f"{args.model}_{args.modality}" / config
    out_root.mkdir(parents=True, exist_ok=True)

    maps_out = out_root / f"rsa_subcortical_{fmri_tag}_{config}_searchlight.npy"
    nuclei_csv = out_root / f"nuclei_roi_rsa_{fmri_tag}_{config}.csv"
    cifti_out = out_root / f"rsa_subcortical_{fmri_tag}_{config}_maps.dscalar.nii"

    if maps_out.exists() and nuclei_csv.exists() and cifti_out.exists():
        log.info("Outputs already exist — skipping.")
        return

    timing_df = pd.read_csv(args.timing_csv)
    neighbor_cache_dir = Path(args.neighbor_cache_dir or
                               (ROOT / "outputs" / "subcortical" / "_neighbor_cache"))

    data, bm_axis, run_trs = _get_fmri(args, fmri_tag)
    log.info(f"fMRI continuous: {data.shape}  run_trs={run_trs.tolist()}")
    slices = struct_slices(bm_axis)

    assert_segment_timing(timing_df, args.bin_sec, args.tr, args.delay_sec, args.skip_sec, run_trs)
    fmri_binned = preprocess_fmri(data, timing_df, run_trs, args.bin_sec, args.tr,
                                   args.delay_sec, skip_sec=args.skip_sec, normalize=True)
    log.info(f"fMRI binned: {fmri_binned.shape}")

    emb_file = (Path(args.embeddings_dir) / args.model /
                f"bin{bin_int}s_skip{skip_int}s" / f"{args.model}_{args.modality}.npy")
    emb = process_model_embeddings(str(emb_file), timing_df, bin_sec=args.bin_sec, tr=args.tr,
                                    run_trs=run_trs, delay_sec=args.delay_sec, hrf=args.hrf,
                                    skip_sec=args.skip_sec, normalize=True)
    log.info(f"Model binned: {emb.shape}")

    fmri_binned, emb = align_and_assert_bins(fmri_binned, emb)
    n_bins = fmri_binned.shape[0]
    n_total = fmri_binned.shape[1]

    # ── Per-structure searchlight ────────────────────────────────────────────
    corr_full = np.zeros(n_total, dtype=np.float32)
    neighbor_modes = {}
    for s in SUBCORTICAL_STRUCTURES:
        info = slices[s]
        start, stop = info["start"], info["stop"]
        n_struct = stop - start
        fmri_struct = fmri_binned[:, start:stop]

        if s.startswith("CEREBELLUM_"):
            hem = "LEFT" if s.endswith("LEFT") else "RIGHT"
            neighbors, mode_used = get_cerebellum_neighbors(
                hem, info["world_xyz"], info["voxel_ijk"], args.k,
                neighbor_cache_dir, args.workbench)
        else:
            mode = STRUCTURE_NEIGHBOR_MODE[s]
            cache_path = neighbor_cache_dir / f"{s}_neighbors_k{args.k}_{mode}.npy"
            neighbors = build_neighbors(info["voxel_ijk"], info["world_xyz"], args.k, mode, cache_path)
            mode_used = mode
        neighbor_modes[s] = mode_used

        idx = np.arange(n_struct, dtype=np.int32)
        corr_struct = run_searchlight(
            fmri_struct, emb, neighbors, surface_indices=idx, vertex_to_col=idx,
            method=args.method, n_jobs=N_JOBS, batch_size=args.gpu_batch_size,
        )
        corr_full[start:stop] = corr_struct
        log.info(f"  {s}: n_vox={n_struct}  mode={mode_used}  mean_rho={corr_struct.mean():.4f}  "
                 f"max_rho={corr_struct.max():.4f}")

    np.save(str(maps_out), corr_full)
    log.info(f"Saved: {maps_out}")

    # ── Significance maps (reuses rsa/searchlight.py's helpers) ─────────────
    p_uncorr, sigmap_uncorr = _rho_sigmap(corr_full, n_bins)
    sigmap_fdr, fdr_mask, n_sig_fdr = _fdr_sigmap(p_uncorr, corr_full)
    log.info(f"FDR significant (p<0.05): {n_sig_fdr:,} / {n_total:,}")

    save_cifti_multimap(
        np.stack([corr_full, sigmap_uncorr, sigmap_fdr, fdr_mask], axis=0),
        ["searchlight_rho", "sigmap_uncorr", "sigmap_fdr", "fdr_mask"],
        args.template_cifti, str(cifti_out),
    )
    log.info(f"Saved: {cifti_out}")

    # ── Per-structure summary ────────────────────────────────────────────────
    struct_summary = {}
    for s in SUBCORTICAL_STRUCTURES:
        info = slices[s]
        seg = corr_full[info["start"]:info["stop"]]
        struct_summary[s] = {
            "n_vox": int(seg.size), "mean_rho": float(seg.mean()), "max_rho": float(seg.max()),
            "n_fdr_sig": int(fdr_mask[info["start"]:info["stop"]].sum()),
            "neighbor_mode": neighbor_modes[s],
        }
    (out_root / "struct_summary.json").write_text(json.dumps(struct_summary, indent=2))

    # ── Nuclei ROI-RSA (IC/SC/MGN/LGN — too small for a within-mask searchlight) ──
    masks = build_nucleus_masks(slices, radius=args.nucleus_radius)
    rows = []
    for nucleus, cols in masks.items():
        rdm_brain = compute_rdm(fmri_binned[:, cols], method="correlation")
        rdm_model = compute_rdm(emb, method="correlation")
        rho, p = correlate_rdms(rdm_brain, rdm_model, method=args.method)
        rows.append({"nucleus": nucleus, "n_vox": len(cols), "rho": rho, "p": p,
                      "parent": NUCLEI_SEEDS[nucleus][1]})
        log.info(f"  nucleus {nucleus}: rho={rho:.4f}  p={p:.4g}  n_vox={len(cols)}")

    pd.DataFrame(rows).to_csv(nuclei_csv, index=False)
    log.info(f"Saved: {nuclei_csv}")
    log.info("Done.")


if __name__ == "__main__":
    main()
