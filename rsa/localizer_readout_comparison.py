"""
rsa/localizer_readout_comparison.py
==========================
Quantitative comparison of the 8 topoomni localizer-readout RSA searchlight
runs: 2 localizer kinds (auditory / integration) x 4 readout tags
(text / av / clsav_from_a / clsav_from_v).

For each of the 8 combined dscalar.nii outputs (already computed by
run_localizer_readout_rsa.sh), reports:
  - peak Spearman rho and its Glasser parcel + hemisphere
  - % FDR-significant vertices (p<0.05)
  - top-5 Glasser parcels by mean rho within the FDR-significant mask
Then, within each localizer kind, reports pairwise Dice overlap of the
FDR-significant masks across the 4 readout tags.

No figures are produced -- exact wb_view paths + map names are printed
for manual inspection, per standing project convention.
"""

import os
import sys
from pathlib import Path
from itertools import combinations

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsa.glasser import load_glasser_parcels
from cifti_io import get_bm_axis
from rsa.shared.naming import DEFAULT_MODEL_NORM
from paths import DATA, OUTPUTS

GROUP_DIR = OUTPUTS / "rsa/raw/group_average"
GLASSER_DLABEL = str(DATA / "HCP_S1200_GroupAvg_v1") + "/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii"
CONFIG_TAG = f"k100_delay5s_bin5s_skip5s_spearman_{os.environ.get('MODEL_NORM', DEFAULT_MODEL_NORM)}"

KINDS = ["auditory", "integration"]
TAGS = ["text", "av", "clsav_from_a", "clsav_from_v"]

RHO_MAP = "searchlight_spearman_rho"
FDR_MAP = "searchlight_spearman_rho_sigmap_fdr"


def combined_path(kind: str, tag: str) -> Path:
    model = f"topoomni_{kind}_localizer_{tag}_av"
    return GROUP_DIR / model / f"rsa_59k_raw_{CONFIG_TAG}_maps.dscalar.nii"


def load_maps(path: Path):
    img = nib.load(str(path))
    names = list(img.header.get_axis(0).name)
    data = img.get_fdata(dtype=np.float32)
    bm_axis = img.header.get_axis(1)
    out = {n: data[i] for i, n in enumerate(names)}
    return out, bm_axis


def vertex_to_hem_and_local(bm_axis, flat_idx: int):
    for name, sl, struct in bm_axis.iter_structures():
        stop = sl.stop if sl.stop is not None else sl.start + len(struct.vertex)
        if sl.start <= flat_idx < stop:
            local_i = flat_idx - sl.start
            vidx = int(struct.vertex[local_i])
            hem = "L" if "LEFT" in name else ("R" if "RIGHT" in name else name)
            return hem, vidx
    return "?", -1


def parcel_name_for_index(parcels: dict, idx: int) -> str:
    for name, indices in parcels.items():
        # indices sorted; use searchsorted for speed
        pos = np.searchsorted(indices, idx)
        if pos < len(indices) and indices[pos] == idx:
            return name
    return "(no parcel / medial wall)"


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return 2.0 * inter / denom


def main():
    results = {}
    print("=" * 100)
    print("LOCALIZER-READOUT RSA SEARCHLIGHT COMPARISON")
    print("2 localizer kinds x 4 readout tags, k=100, delay=5s, bin=5s, skip=5s, spearman")
    print("=" * 100)

    parcels_cache = None

    for kind in KINDS:
        for tag in TAGS:
            key = f"{kind}/{tag}"
            path = combined_path(kind, tag)
            if not path.exists():
                print(f"[MISSING] {key}: {path}")
                continue
            maps, bm_axis = load_maps(path)
            if parcels_cache is None:
                parcels_cache = load_glasser_parcels(GLASSER_DLABEL, bm_axis)
            rho = maps[RHO_MAP]
            fdr = maps[FDR_MAP]

            peak_idx = int(np.nanargmax(rho))
            peak_rho = float(rho[peak_idx])
            hem, vidx = vertex_to_hem_and_local(bm_axis, peak_idx)
            peak_parcel = parcel_name_for_index(parcels_cache, peak_idx)

            n_total = rho.shape[0]
            fdr_mask = fdr > 0
            n_sig = int(fdr_mask.sum())
            pct_sig = 100.0 * n_sig / n_total

            # top-5 parcels by mean rho within FDR-significant vertices
            parcel_means = []
            for pname, indices in parcels_cache.items():
                sig_in_parcel = fdr_mask[indices]
                if sig_in_parcel.sum() == 0:
                    continue
                mean_rho = float(np.mean(rho[indices[sig_in_parcel]]))
                n_parcel_sig = int(sig_in_parcel.sum())
                parcel_means.append((pname, mean_rho, n_parcel_sig))
            parcel_means.sort(key=lambda x: -x[1])
            top5 = parcel_means[:5]

            # top-1% strongest vertices by rho (more discriminative than the
            # FDR mask, which saturates near ceiling -- see note below)
            n_top1pct = max(1, int(round(0.01 * n_total)))
            top1pct_idx = np.argsort(rho)[-n_top1pct:]
            top1pct_mask = np.zeros(n_total, dtype=bool)
            top1pct_mask[top1pct_idx] = True

            results[key] = dict(
                path=path, rho=rho, fdr_mask=fdr_mask, top1pct_mask=top1pct_mask,
                peak_rho=peak_rho, peak_hem=hem, peak_vidx=vidx, peak_parcel=peak_parcel,
                pct_sig=pct_sig, n_sig=n_sig, n_total=n_total, top5=top5,
            )

            print(f"\n--- {key} ---")
            print(f"  file: {path}")
            print(f"  maps: {RHO_MAP} (map 1), {FDR_MAP} (map 3)")
            print(f"  peak rho = {peak_rho:.4f}  @ {hem} vertex {vidx}  parcel = {peak_parcel}")
            print(f"  FDR-significant (p<0.05): {n_sig} / {n_total} ({pct_sig:.2f}%)")
            print(f"  top-5 Glasser parcels by mean rho (within FDR-sig vertices):")
            for pname, mean_rho, n_parcel_sig in top5:
                print(f"    {pname:40s}  mean_rho={mean_rho:.4f}  n_sig_vertices={n_parcel_sig}")

    # pairwise Dice overlap of FDR masks across the 4 tags, within each kind
    print("\n" + "=" * 100)
    print("PAIRWISE FDR-MASK OVERLAP (Dice coefficient) WITHIN EACH LOCALIZER KIND")
    print("NOTE: FDR-significant (p<0.05) covers ~99.9% of cortex in every map here")
    print("(large effective n_bins=626 makes even rho~0.003 pass FDR), so this")
    print("metric saturates near 1.0 and is NOT discriminative -- reported for")
    print("completeness only. See top-1%-vertex overlap below for the metric")
    print("that actually differentiates the readout variants.")
    print("=" * 100)
    for kind in KINDS:
        print(f"\n--- {kind} localizer ---")
        keys = [f"{kind}/{tag}" for tag in TAGS if f"{kind}/{tag}" in results]
        for a, b in combinations(keys, 2):
            d = dice(results[a]["fdr_mask"], results[b]["fdr_mask"])
            print(f"  Dice({a.split('/')[1]:16s}, {b.split('/')[1]:16s}) = {d:.4f}")

    print("\n" + "=" * 100)
    print("PAIRWISE TOP-1%-VERTEX OVERLAP (Dice, ~1084 strongest-rho vertices each)")
    print("=" * 100)
    for kind in KINDS:
        print(f"\n--- {kind} localizer ---")
        keys = [f"{kind}/{tag}" for tag in TAGS if f"{kind}/{tag}" in results]
        for a, b in combinations(keys, 2):
            d = dice(results[a]["top1pct_mask"], results[b]["top1pct_mask"])
            print(f"  Dice({a.split('/')[1]:16s}, {b.split('/')[1]:16s}) = {d:.4f}")

    print("\n" + "=" * 100)
    print("EXACT wb_view PATHS FOR MANUAL INSPECTION (no auto-generated figures)")
    print("=" * 100)
    for kind in KINDS:
        for tag in TAGS:
            key = f"{kind}/{tag}"
            if key not in results:
                continue
            print(f"\n{key}:")
            print(f"  {results[key]['path']}")
            print(f"    map 'searchlight_spearman_rho'            -> continuous Spearman rho")
            print(f"    map 'searchlight_spearman_rho_sigmap_uncorr' -> uncorrected p<0.05 mask")
            print(f"    map 'searchlight_spearman_rho_sigmap_fdr'  -> FDR p<0.05 mask")


if __name__ == "__main__":
    main()
