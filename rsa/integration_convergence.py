"""
rsa/integration_convergence.py
================================
Cross-architecture integration convergence map.

Combines the best-additive integration maps (rsa/partial_rsa.py,
kind="integration") across every native-AV model that succeeded, to show
that the integration territory is a property of the BRAIN, not an artifact
of any one embedding. Tests the hypothesis that integration lives in the
INFORMATION content, not the training objective, so it should be visible
across architecturally distinct native-AV models (see
rsa/multimodal_decomposition.py's docstring for the underlying decomposition).

Two convergence layers, both saved and reported:
  convergence_mean_z    : per-vertex mean of z-scored (across-cortex)
                          integration maps -- the primary convergence signal.
  convergence_sign_count: per-vertex count (0..n_models) of models with a
                          POSITIVE integration value. This is a descriptive
                          sign-agreement count, NOT a formal FDR-significance
                          count -- per-model FDR would need per-subject or
                          permutation-based null distributions, which this
                          script does not compute. Labelled honestly rather
                          than mislabelled as "FDR-significant".

Usage
-----
python rsa/integration_convergence.py \\
    --models pe-av-small-16-frame cav-mae-sync omni3b_layer18_mp topoomni_layer18_mp \\
    --rsa-root /home/amin/Research/Representation/Movie/outputs/rsa/raw/group_average \\
    --config k100_delay5s_bin5s_skip5s_spearman \\
    --template-cifti /home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii \\
    --glasser-dlabel /home/amin/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii \\
    --out-dir /home/amin/Research/Representation/Movie/outputs/rsa/raw/group_average/_integration_convergence
"""

import argparse
import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cifti_io import get_bm_axis, save_cifti_multimap
from rsa.glasser import load_glasser_parcels

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True,
                   help="Native-AV model names (must have a Move-1 integration map).")
    p.add_argument("--rsa-root", required=True, dest="rsa_root",
                   help="outputs/rsa/raw/group_average")
    p.add_argument("--config", required=True,
                   help="e.g. k100_delay5s_bin5s_skip5s_spearman")
    p.add_argument("--template-cifti", required=True, dest="template_cifti")
    p.add_argument("--glasser-dlabel", required=True, dest="glasser_dlabel")
    p.add_argument("--out-dir", required=True, dest="out_dir")
    return p.parse_args()


def main():
    args = parse_args()
    rsa_root = Path(args.rsa_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maps = {}
    for model in args.models:
        npy_path = (rsa_root / f"{model}_av_INTEGRATION" / args.config /
                    "integration_partial_r_searchlight.npy")
        dscalar_path = npy_path.with_suffix(".dscalar.nii")
        if npy_path.exists():
            d = np.load(npy_path).astype(np.float32)
        elif dscalar_path.exists():
            d = nib.load(str(dscalar_path)).get_fdata().squeeze().astype(np.float32)
        else:
            log.warning(f"SKIP {model}: no integration map found at {npy_path}")
            continue
        if not np.isfinite(d).all():
            log.warning(f"SKIP {model}: non-finite values in integration map")
            continue
        maps[model] = d
        log.info(f"  [{model}] loaded shape={d.shape} peak={d.max():.4f} mean={d.mean():.4f}")

    if len(maps) < 2:
        log.error(f"Need >=2 models with valid integration maps; got {list(maps.keys())}")
        sys.exit(1)

    model_names = list(maps.keys())
    stack = np.stack([maps[m] for m in model_names], axis=0)  # (n_models, n_grayord)
    n_models = stack.shape[0]
    log.info(f"Convergence over {n_models} models: {model_names}")

    # ── Z-score each model's map across cortex, then average ────────────────
    mu = stack.mean(axis=1, keepdims=True)
    sd = stack.std(axis=1, keepdims=True)
    sd[sd < 1e-10] = 1.0
    z = (stack - mu) / sd
    convergence_mean_z = z.mean(axis=0).astype(np.float32)

    # ── Sign-agreement count (descriptive, NOT an FDR-significance count) ───
    sign_count = (stack > 0).sum(axis=0).astype(np.float32)

    log.info(
        f"convergence_mean_z: peak={convergence_mean_z.max():.4f} "
        f"mean_pos={convergence_mean_z[convergence_mean_z > 0].mean():.4f} "
        f"pct_pos={100*(convergence_mean_z > 0).mean():.1f}%"
    )
    log.info(
        f"convergence_sign_count: max={sign_count.max():.0f}/{n_models}  "
        f"pct_all_agree={100*(sign_count == n_models).mean():.1f}%  "
        f"pct_none_positive={100*(sign_count == 0).mean():.1f}%"
    )

    out_path = out_dir / "convergence_map.dscalar.nii"
    save_cifti_multimap(
        np.stack([convergence_mean_z, sign_count], axis=0),
        ["convergence_mean_z", "convergence_sign_count"],
        args.template_cifti,
        str(out_path),
    )
    log.info(f"Saved: {out_path}")

    # ── Top Glasser parcels (ranked by convergence_mean_z) ───────────────────
    bm_axis = get_bm_axis(args.template_cifti)
    parcels = load_glasser_parcels(args.glasser_dlabel, bm_axis)
    rows = []
    for name, indices in parcels.items():
        idx = np.array(indices)
        rows.append({
            "parcel": name,
            "mean_convergence_z": float(convergence_mean_z[idx].mean()),
            "mean_sign_count": float(sign_count[idx].mean()),
            "n_vertices": int(len(idx)),
        })
    report = pd.DataFrame(rows).sort_values("mean_convergence_z", ascending=False).reset_index(drop=True)
    report["rank"] = range(1, len(report) + 1)
    report.to_csv(out_dir / "convergence_ranked_parcels.csv", index=False)
    log.info(f"Top-10 Glasser parcels by convergence_mean_z:\n{report.head(10).to_string(index=False)}")

    summary = {
        "models": model_names,
        "n_models": n_models,
        "convergence_mean_z_peak": float(convergence_mean_z.max()),
        "convergence_mean_z_mean_positive": float(convergence_mean_z[convergence_mean_z > 0].mean()),
        "convergence_mean_z_pct_positive": float(100 * (convergence_mean_z > 0).mean()),
        "sign_count_pct_all_agree": float(100 * (sign_count == n_models).mean()),
        "sign_count_pct_none_positive": float(100 * (sign_count == 0).mean()),
        "note": ("convergence_sign_count is a descriptive per-vertex count of models "
                 "with positive integration rho, NOT a formal FDR-significance count "
                 "(that would need per-subject or permutation null distributions, "
                 "which this script does not compute)."),
    }
    import json
    json.dump(summary, open(out_dir / "convergence_summary.json", "w"), indent=2)
    log.info("Done.")


if __name__ == "__main__":
    main()
