#!/usr/bin/env python3
"""Per-subject CCA-A/CCA-P masks and seed connectivity: real per-subject
.dscalar.nii maps, compact stacked arrays, and group-aggregate CIFTIs.

Reuses (does not reimplement): mask extraction from
persubject_cca_channel_1pct.subject_mask (which itself calls
run_cca_islands.py), preprocessing from preprocess_individual.preprocess_subject,
connectivity from connectivity.seed_connectivity.compute_seed_connectivity, and
CIFTI writing from cifti_io.save_cifti_map / save_cifti_multimap.

Subject list: taken directly from per_subject_contrast.csv (the ground-truth
166-subject set that persubject_cca_channel_1pct.py already produced), not
re-derived, so it is guaranteed to match.

HAZARD GUARD: the per-subject PE-AV searchlight maps this script reads may be
migrated to an HDD while this script runs. Every subject's input path is
resolved and verified to exist up front (`enumerate_inputs`); the exact same
path is re-checked immediately before use in the main loop. Any subject whose
map vanishes between those two checks is a hard error (RuntimeError) that
aborts the whole run before any summary/aggregate file is written — a
disappeared input must never silently degrade into a skipped subject.

Usage
-----
python cf_modeling/deprecated/persubject_cca_maps.py \
    --searchlight-root outputs/rsa/raw/subject_data \
    --persubject-out outputs/cf_modeling/persubject_cca_1pct/persubject

Export one subject's map on demand from the compact stacked arrays (works even
after --persubject-out has been moved off the SSD, since the stacked arrays
are the ones kept on the SSD):
    python cf_modeling/deprecated/persubject_cca_maps.py --export-subject 100610
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cf_modeling.deprecated.persubject_cca_channel_1pct as pcca1pct  # noqa: E402
from cf_modeling.cf_naming import persubject_output_root  # noqa: E402
from cf_modeling.deprecated.run_cca_islands import DEFAULT_TEMPLATE  # noqa: E402
from cifti_io import save_cifti_map, save_cifti_multimap, get_bm_axis  # noqa: E402
from connectivity.seed_connectivity import compute_seed_connectivity  # noqa: E402
from preprocess_individual import preprocess_subject  # noqa: E402
from rsa.glasser import load_glasser_parcels  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("persubject_cca_maps")

MOVIE_ROOT = Path("/home/amin/Research/Representation/Movie")
DEFAULT_SEARCHLIGHT_ROOT = MOVIE_ROOT / "outputs/rsa/raw/subject_data"
SEARCHLIGHT_SUBPATH = (
    "pe-av-small-16-frame_av/k100_delay5s_bin5s_skip5s_spearman/"
    "rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy")
DEFAULT_OUT_DIR = persubject_output_root() / "persubject_cca_1pct"
DEFAULT_CONTRAST_CSV = DEFAULT_OUT_DIR / "per_subject_contrast.csv"
DEFAULT_PERSUBJECT_OUT = DEFAULT_OUT_DIR / "persubject"
N_GRAY = pcca1pct.N_GRAY  # 108441
TR = pcca1pct.TR
RAW_DIR = pcca1pct.RAW_DIR
OVERLAP_UNASSIGNED_FLOOR = 0.05  # modal-assignment: neither island claims below this


def searchlight_path(root: Path, sub: str) -> Path:
    return root / sub / SEARCHLIGHT_SUBPATH


def load_master_subjects(contrast_csv: Path) -> list[str]:
    return pd.read_csv(contrast_csv).subject.astype(str).tolist()


def enumerate_inputs(subjects: list[str], root: Path) -> dict[str, Path]:
    """Resolve and verify every subject's searchlight map before the main loop.

    All `subjects` come from per_subject_contrast.csv, i.e. every one of them
    already succeeded through mask extraction in a prior run — so any miss
    here is an unexpected input-availability problem (e.g. a bulk transfer in
    progress), not ordinary subject attrition, and is fatal.
    """
    resolved, missing = {}, []
    for sub in subjects:
        p = searchlight_path(root, sub)
        if p.exists():
            resolved[sub] = p
        else:
            missing.append((sub, p))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(subjects)} expected searchlight maps missing under "
            f"{root} at enumeration time (subjects are from {DEFAULT_CONTRAST_CSV.name}, "
            f"all previously successful — a concurrent HDD migration is the likely cause). "
            f"First few: {[str(p) for _, p in missing[:5]]}")
    assert len(resolved) == len(subjects), "resolved count must equal input subject count"
    log.info("Enumerated and verified %d/%d searchlight maps under %s",
              len(resolved), len(subjects), root)
    return resolved


def fisher_z_mean(r_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n_subjects, n_gray) Pearson r -> (mean Fisher z, back-transformed mean r).

    Convention: average in Fisher-z space, then inverse-transform (tanh) for
    the reported r map. Any contrast between two such group maps is computed
    in z-space (mean_z_a - mean_z_p), not on the back-transformed r maps.
    """
    z = np.arctanh(np.clip(r_stack.astype(np.float64), -0.999999, 0.999999))
    z_mean = z.mean(axis=0)
    return z_mean, np.tanh(z_mean)


def pack_masks(mask_a_stack: np.ndarray, mask_p_stack: np.ndarray) -> dict:
    """Bit-pack (n_subjects, n_gray) boolean stacks for compact storage."""
    return {
        "mask_a_packed": np.packbits(mask_a_stack, axis=1),
        "mask_p_packed": np.packbits(mask_p_stack, axis=1),
        "n_gray": N_GRAY,
    }


def unpack_masks(packed: np.ndarray, n_subjects: int, n_gray: int) -> np.ndarray:
    bits = np.unpackbits(packed, axis=1, count=n_gray)
    return bits.astype(bool).reshape(n_subjects, n_gray)


def _demo() -> None:
    """Non-trivial-logic self-check: packbits round-trip, fisher-z average."""
    rng = np.random.default_rng(0)
    n_sub, n_gray = 7, 271  # deliberately not a multiple of 8
    masks = rng.random((n_sub, n_gray)) > 0.7
    packed = np.packbits(masks, axis=1)
    restored = unpack_masks(packed, n_sub, n_gray)
    assert np.array_equal(masks, restored), "packbits round-trip failed"

    r = np.full((3, 4), 0.5)
    z_mean, r_mean = fisher_z_mean(r)
    assert np.allclose(z_mean, np.arctanh(0.5)), "fisher-z mean of constant input must equal arctanh(0.5)"
    assert np.allclose(r_mean, 0.5, atol=1e-6), "back-transform of constant fisher-z must recover 0.5"

    mixed = np.array([[0.9, -0.9], [-0.9, 0.9]])
    z_mixed, r_mixed = fisher_z_mean(mixed)
    assert np.allclose(z_mixed, 0.0, atol=1e-9) and np.allclose(r_mixed, 0.0, atol=1e-9), \
        "symmetric +/-0.9 must average to exactly zero in fisher-z space"
    print("demo OK: packbits round-trip + fisher-z average verified")


# =============================================================================
# Disk budget
# =============================================================================

def estimate_footprint_bytes(n_subjects: int) -> int:
    map_bytes = N_GRAY * 4  # float32 single map
    per_subject = 2 * (2 * map_bytes)  # {mask,connectivity} x {cca_a,cca_p}
    packed_masks = 2 * n_subjects * int(np.ceil(N_GRAY / 8))
    stacked_conn = 2 * n_subjects * N_GRAY * 2  # float16
    aggregates = 8 * map_bytes
    return n_subjects * per_subject + packed_masks + stacked_conn + aggregates


def check_disk_budget(out_dir: Path, n_subjects: int, min_free_gb: float) -> int:
    estimated = estimate_footprint_bytes(n_subjects)
    free = shutil.disk_usage(out_dir if out_dir.exists() else out_dir.parent).free
    margin = min_free_gb * 1e9
    log.info("Disk budget: estimated new writes=%.1f MB, free=%.2f GB, required margin=%.2f GB",
              estimated / 1e6, free / 1e9, min_free_gb)
    if free - estimated < margin:
        raise RuntimeError(
            f"Aborting: writing an estimated {estimated / 1e6:.1f} MB would leave "
            f"{(free - estimated) / 1e9:.2f} GB free, below the required {min_free_gb} GB margin "
            f"(currently {free / 1e9:.2f} GB free).")
    return estimated


# =============================================================================
# Main per-subject loop
# =============================================================================

def run_main(args) -> None:
    subjects = load_master_subjects(args.contrast_csv)
    log.info("Master subject list: %d subjects from %s", len(subjects), args.contrast_csv)
    resolved_paths = enumerate_inputs(subjects, args.searchlight_root)

    args.persubject_out.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    estimated_bytes = check_disk_budget(args.out_dir, len(subjects), args.min_free_gb)

    pcca1pct.SEARCHLIGHT_TMPL = str(args.searchlight_root / "{sub}" / SEARCHLIGHT_SUBPATH)
    prep_args = types.SimpleNamespace(sg_filter=True, psc=True, gsr=False)
    tmp_base = Path(tempfile.mkdtemp(prefix="persubject_cca_maps_"))

    mask_a_stack = np.zeros((len(subjects), N_GRAY), dtype=bool)
    mask_p_stack = np.zeros((len(subjects), N_GRAY), dtype=bool)
    conn_a_stack = np.zeros((len(subjects), N_GRAY), dtype=np.float16)
    conn_p_stack = np.zeros((len(subjects), N_GRAY), dtype=np.float16)

    try:
        for i, sub in enumerate(subjects, 1):
            map_path = resolved_paths[sub]
            if not map_path.exists():
                raise RuntimeError(
                    f"HAZARD: searchlight map for subject {sub} vanished mid-run "
                    f"(present at enumeration, missing now): {map_path}. Aborting — "
                    f"{i - 1}/{len(subjects)} subjects completed; refusing to write partial "
                    f"summary/aggregate outputs.")
            masks = pcca1pct.subject_mask(sub, tmp_base)
            if masks is None:
                raise RuntimeError(
                    f"Subject {sub} is in the trusted {len(subjects)}-subject list (already "
                    f"succeeded mask extraction once) but failed this time. Aborting rather "
                    f"than silently dropping a subject that should be reproducible.")
            mask_a, mask_p = masks["mask_a"], masks["mask_p"]

            save_cifti_multimap(
                np.stack([mask_a, mask_p]).astype(np.float32), ["cca_a", "cca_p"],
                str(DEFAULT_TEMPLATE), str(args.persubject_out / f"{sub}_cca_masks.dscalar.nii"))

            data, _bm, _run_trs = preprocess_subject(sub, RAW_DIR, TR, prep_args)
            r_map_a, _ = compute_seed_connectivity(data, mask_a)
            r_map_p, _ = compute_seed_connectivity(data, mask_p)
            del data

            save_cifti_multimap(
                np.stack([r_map_a, r_map_p]), ["cca_a", "cca_p"],
                str(DEFAULT_TEMPLATE),
                str(args.persubject_out / f"{sub}_seed_connectivity.dscalar.nii"))

            mask_a_stack[i - 1], mask_p_stack[i - 1] = mask_a, mask_p
            conn_a_stack[i - 1] = r_map_a.astype(np.float16)
            conn_p_stack[i - 1] = r_map_p.astype(np.float16)
            log.info("[%d/%d] %s done", i, len(subjects), sub)
    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)

    n_processed = len(subjects)
    if n_processed != len(pd.read_csv(args.contrast_csv)):
        log.error("MISMATCH: processed %d subjects but per_subject_contrast.csv has %d",
                  n_processed, len(pd.read_csv(args.contrast_csv)))
    else:
        log.info("Subject count check OK: processed %d, matches per_subject_contrast.csv (%d)",
                  n_processed, len(pd.read_csv(args.contrast_csv)))

    np.savez_compressed(
        args.out_dir / "masks_packed.npz", subjects=np.array(subjects),
        **pack_masks(mask_a_stack, mask_p_stack))
    np.savez_compressed(
        args.out_dir / "connectivity_stacked.npz", subjects=np.array(subjects),
        conn_a=conn_a_stack, conn_p=conn_p_stack)
    log.info("Wrote stacked arrays to %s", args.out_dir)

    write_aggregates(mask_a_stack, mask_p_stack, conn_a_stack, conn_p_stack, args.out_dir)

    du_bytes = sum(f.stat().st_size for f in args.persubject_out.rglob("*") if f.is_file())
    du_bytes += sum(f.stat().st_size for f in args.out_dir.glob("*") if f.is_file())
    free_after = shutil.disk_usage(args.out_dir).free
    log.info("Actual footprint written: %.1f MB (estimated %.1f MB). Free space now: %.2f GB",
              du_bytes / 1e6, estimated_bytes / 1e6, free_after / 1e9)


def write_aggregates(mask_a_stack, mask_p_stack, conn_a_stack, conn_p_stack, out_dir: Path) -> None:
    p_a = mask_a_stack.mean(axis=0).astype(np.float32)
    p_p = mask_p_stack.mean(axis=0).astype(np.float32)
    save_cifti_map(p_a, str(DEFAULT_TEMPLATE), str(out_dir / "cca_a_overlap_probability.dscalar.nii"),
                    "cca_a_overlap_probability")
    save_cifti_map(p_p, str(DEFAULT_TEMPLATE), str(out_dir / "cca_p_overlap_probability.dscalar.nii"),
                    "cca_p_overlap_probability")
    save_cifti_map(p_a - p_p, str(DEFAULT_TEMPLATE),
                    str(out_dir / "cca_preference_probability_diff.dscalar.nii"),
                    "p_cca_a_minus_p_cca_p")

    keys = np.zeros(N_GRAY, dtype=np.int32)
    keys[(p_a > p_p) & (p_a >= OVERLAP_UNASSIGNED_FLOOR)] = 1
    keys[(p_p > p_a) & (p_p >= OVERLAP_UNASSIGNED_FLOOR)] = 2
    import nibabel as nib
    bm_axis = get_bm_axis(str(DEFAULT_TEMPLATE))
    labels = {0: ("UNASSIGNED", (0.0, 0.0, 0.0, 0.0)),
              1: ("cca_a_modal", (0.85, 0.16, 0.16, 1.0)),
              2: ("cca_p_modal", (0.16, 0.34, 0.85, 1.0))}
    label_axis = nib.cifti2.LabelAxis(["CCA modal assignment"], [labels])
    header = nib.cifti2.Cifti2Header.from_axes((label_axis, bm_axis))
    nib.save(nib.Cifti2Image(keys[None, :], header=header), str(out_dir / "cca_modal_assignment.dlabel.nii"))

    z_a, r_a_mean = fisher_z_mean(conn_a_stack)
    z_p, r_p_mean = fisher_z_mean(conn_p_stack)
    save_cifti_map(r_a_mean.astype(np.float32), str(DEFAULT_TEMPLATE),
                    str(out_dir / "connectivity_cca_a_mean.dscalar.nii"), "connectivity_cca_a_mean")
    save_cifti_map(r_p_mean.astype(np.float32), str(DEFAULT_TEMPLATE),
                    str(out_dir / "connectivity_cca_p_mean.dscalar.nii"), "connectivity_cca_p_mean")

    z_contrast = (z_a - z_p).astype(np.float32)
    save_cifti_map(z_contrast, str(DEFAULT_TEMPLATE),
                    str(out_dir / "connectivity_contrast_a_minus_p.dscalar.nii"),
                    "fisher_z_contrast_cca_a_minus_cca_p")

    per_subject_contrast_z = (
        np.arctanh(np.clip(conn_a_stack.astype(np.float64), -0.999999, 0.999999))
        - np.arctanh(np.clip(conn_p_stack.astype(np.float64), -0.999999, 0.999999)))
    group_sign = np.sign(z_contrast.astype(np.float64))
    same_sign = np.sign(per_subject_contrast_z) == group_sign
    consistency = same_sign.mean(axis=0).astype(np.float32)
    save_cifti_map(consistency, str(DEFAULT_TEMPLATE),
                    str(out_dir / "connectivity_contrast_consistency.dscalar.nii"),
                    "consistency_fraction_same_sign_as_group")

    glasser_report(p_a, p_p, out_dir)

    metadata = {
        "n_subjects": int(mask_a_stack.shape[0]),
        "cca_a_peak_overlap_probability": float(p_a.max()),
        "cca_p_peak_overlap_probability": float(p_p.max()),
        "modal_assignment_unassigned_floor": OVERLAP_UNASSIGNED_FLOOR,
        "connectivity_averaging": "Fisher-z mean across subjects, then tanh back-transform for the reported r maps",
        "connectivity_contrast_definition": "mean_z(CCA-A) - mean_z(CCA-P), computed in Fisher-z space (not back-transformed)",
        "consistency_definition": "fraction of subjects whose own (z_a - z_p) has the same sign as the group mean contrast",
        "statistical_caveat_1_factor": (
            "No cortex-wide significance test was computed on these aggregate maps. If one is "
            "run later, it is a 1-factor test across subjects who all watched the identical movie "
            "stimulus: valid as a claim of generalization across people for this stimulus, "
            "anticonservative as a claim about movies/stimuli in general. Do not present such a "
            "test as a stimulus-general cortex-wide significance claim."),
    }
    (out_dir / "persubject_cca_maps_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def glasser_report(p_a: np.ndarray, p_p: np.ndarray, out_dir: Path) -> None:
    dlabel = pcca1pct.GLASSER_DLABEL
    bm_axis = get_bm_axis(str(DEFAULT_TEMPLATE))
    parcels = load_glasser_parcels(str(dlabel), bm_axis)
    vertex_to_parcel = {idx: name for name, indices in parcels.items() for idx in indices}
    rows = []
    for label, prob in (("cca_a", p_a), ("cca_p", p_p)):
        peak_idx = int(np.argmax(prob))
        parcel_means = sorted(
            ((name, float(prob[indices].mean())) for name, indices in parcels.items()),
            key=lambda kv: kv[1], reverse=True)[:5]
        rows.append({
            "island": label, "peak_probability": float(prob[peak_idx]),
            "peak_grayordinate": peak_idx,
            "peak_parcel": vertex_to_parcel.get(peak_idx, "unknown"),
            "top5_parcels_by_mean_overlap_probability": parcel_means,
        })
        log.info("%s peak overlap %.3f at grayordinate %d, parcel %s; top parcels: %s",
                  label, prob[peak_idx], peak_idx, vertex_to_parcel.get(peak_idx, "unknown"),
                  parcel_means)
    (out_dir / "cca_overlap_glasser_report.json").write_text(json.dumps(rows, indent=2) + "\n")


# =============================================================================
# --export-subject
# =============================================================================

def export_subject(sub: str, out_dir: Path, export_out: Path) -> None:
    masks_npz = np.load(out_dir / "masks_packed.npz")
    conn_npz = np.load(out_dir / "connectivity_stacked.npz")
    subjects = list(masks_npz["subjects"])
    if sub not in subjects:
        raise ValueError(f"{sub} not found in {out_dir / 'masks_packed.npz'} "
                          f"({len(subjects)} subjects on file)")
    i = subjects.index(sub)
    n_gray = int(masks_npz["n_gray"])
    mask_a = unpack_masks(masks_npz["mask_a_packed"][i:i + 1], 1, n_gray)[0]
    mask_p = unpack_masks(masks_npz["mask_p_packed"][i:i + 1], 1, n_gray)[0]
    export_out.mkdir(parents=True, exist_ok=True)
    save_cifti_multimap(np.stack([mask_a, mask_p]).astype(np.float32), ["cca_a", "cca_p"],
                        str(DEFAULT_TEMPLATE), str(export_out / f"{sub}_cca_masks.dscalar.nii"))
    save_cifti_multimap(
        np.stack([conn_npz["conn_a"][i], conn_npz["conn_p"][i]]).astype(np.float32),
        ["cca_a", "cca_p"], str(DEFAULT_TEMPLATE),
        str(export_out / f"{sub}_seed_connectivity.dscalar.nii"))
    log.info("Exported %s masks + connectivity to %s", sub, export_out)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--searchlight-root", type=Path, default=DEFAULT_SEARCHLIGHT_ROOT,
                        help="Root of per-subject PE-AV searchlight maps (subject-id subdirs). "
                             "Point this at the HDD path once the maps are migrated there.")
    parser.add_argument("--persubject-out", type=Path, default=DEFAULT_PERSUBJECT_OUT,
                        help="Where real per-subject .dscalar.nii files are written. Single "
                             "directory, safe to bulk-mv to an HDD afterward.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Where stacked arrays and aggregate CIFTIs are written (stays on SSD).")
    parser.add_argument("--contrast-csv", type=Path, default=DEFAULT_CONTRAST_CSV,
                        help="Ground-truth subject list (per_subject_contrast.csv).")
    parser.add_argument("--min-free-gb", type=float, default=1.0,
                        help="Abort before writing if free disk after the estimated footprint "
                             "would fall below this margin.")
    parser.add_argument("--export-subject", default=None,
                        help="Skip the main run; export one subject's mask+connectivity "
                             "dscalar.nii from the stacked arrays in --out-dir.")
    parser.add_argument("--export-out", type=Path, default=None,
                        help="Destination for --export-subject (default: --persubject-out).")
    parser.add_argument("--demo", action="store_true", help="Run the self-check and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        _demo()
        return
    if args.export_subject:
        export_subject(args.export_subject, args.out_dir, args.export_out or args.persubject_out)
        return
    run_main(args)


if __name__ == "__main__":
    main()
