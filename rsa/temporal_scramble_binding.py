"""
rsa/temporal_scramble_binding.py
==================================
Move 3 — Temporal-scramble binding control: per-vertex binding maps.

For each native-AV model with both an INTACT and a SCRAMBLED (Move-3,
video[i] paired with audio[perm(i)], seed=42) best-additive integration map
(rsa/partial_rsa.py, kind="integration"), computes

    binding[v] = integration_intact[v] - integration_scrambled[v]

A positive binding value at vertex v means the model's fusion signal there
depends on audio and video being correctly TEMPORALLY paired -- i.e. genuine
cross-modal binding, not just co-occurrence of unimodal content. This is the
complement to Move 1 (which shows fusion is non-additive) and Move 4 (which
shows fusion is architecture-general): Move 3 shows the fusion is actually
PAIRING-sensitive, ruling out the trivial "same average content" explanation.

Also computes a Move-4-style cross-model convergence summary over the
per-model binding maps (mean of per-model z-scored binding maps, and a
descriptive sign-agreement count) -- same caveat as integration_convergence.py:
sign_count is descriptive, not an FDR-significance count.

Usage
-----
python rsa/temporal_scramble_binding.py \\
    --models pe-av-small-16-frame cav-mae-sync \\
             omni3b_layer9 omni3b_layer18 omni3b_layer27 \\
             omni3b_layer9_lasttoken omni3b_layer18_lasttoken omni3b_layer27_lasttoken \\
             topoomni_layer9 topoomni_layer18 topoomni_layer27 \\
             topoomni_layer9_lasttoken topoomni_layer18_lasttoken topoomni_layer27_lasttoken \\
    --rsa-root /home/amin/Research/Representation/Movie/outputs/rsa/raw/group_average \\
    --config k100_delay5s_bin5s_skip5s_spearman \\
    --template-cifti /home/amin/Research/Representation/Movie/data/preprocessed/average_sub/raw/group_average_raw_cortex_59k.dtseries.nii \\
    --glasser-dlabel /home/amin/Research/Representation/Movie/data/HCP_S1200_GroupAvg_v1/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.59k_fs_LR.dlabel.nii \\
    --out-dir /home/amin/Research/Representation/Movie/outputs/rsa/raw/group_average/_temporal_scramble_binding
"""

import argparse
import json
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
                   help="Base model names (must have BOTH {model}_av_INTEGRATION and "
                        "{model}_avscramble_av_INTEGRATION Move-1 maps).")
    p.add_argument("--rsa-root", required=True, dest="rsa_root")
    p.add_argument("--config", required=True)
    p.add_argument("--template-cifti", required=True, dest="template_cifti")
    p.add_argument("--glasser-dlabel", required=True, dest="glasser_dlabel")
    p.add_argument("--out-dir", required=True, dest="out_dir")
    return p.parse_args()


def _load_map(rsa_root: Path, label: str, config: str):
    npy_path = rsa_root / label / config / "integration_partial_r_searchlight.npy"
    dscalar_path = npy_path.with_suffix(".dscalar.nii")
    if npy_path.exists():
        return np.load(npy_path).astype(np.float32)
    if dscalar_path.exists():
        return nib.load(str(dscalar_path)).get_fdata().squeeze().astype(np.float32)
    return None


def main():
    args = parse_args()
    rsa_root = Path(args.rsa_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    binding = {}
    for model in args.models:
        intact = _load_map(rsa_root, f"{model}_av_INTEGRATION", args.config)
        scrambled = _load_map(rsa_root, f"{model}_avscramble_av_INTEGRATION", args.config)
        if intact is None:
            log.warning(f"SKIP {model}: no intact integration map ({model}_av_INTEGRATION)")
            continue
        if scrambled is None:
            log.warning(f"SKIP {model}: no scrambled integration map ({model}_avscramble_av_INTEGRATION)")
            continue
        if not (np.isfinite(intact).all() and np.isfinite(scrambled).all()):
            log.warning(f"SKIP {model}: non-finite values in intact or scrambled map")
            continue
        diff = intact - scrambled
        binding[model] = diff
        log.info(f"  [{model}] intact peak={intact.max():.4f} mean={intact.mean():.4f}  "
                  f"scrambled peak={scrambled.max():.4f} mean={scrambled.mean():.4f}  "
                  f"binding peak={diff.max():.4f} mean={diff.mean():.4f} pct_pos={100*(diff>0).mean():.1f}%")

    if len(binding) < 1:
        log.error("No models had both intact and scrambled integration maps.")
        sys.exit(1)

    model_names = list(binding.keys())

    # ── Per-model binding maps, saved as one multimap CIFTI ─────────────────
    stack = np.stack([binding[m] for m in model_names], axis=0)  # (n_models, n_grayord)
    out_path = out_dir / "binding_maps_per_model.dscalar.nii"
    save_cifti_multimap(stack, [f"binding_{m}" for m in model_names],
                         args.template_cifti, str(out_path))
    log.info(f"Saved: {out_path}")

    # ── Cross-model convergence of the binding effect (Move-4-style) ────────
    n_models = stack.shape[0]
    mu = stack.mean(axis=1, keepdims=True)
    sd = stack.std(axis=1, keepdims=True)
    sd[sd < 1e-10] = 1.0
    z = (stack - mu) / sd
    binding_convergence_mean_z = z.mean(axis=0).astype(np.float32)
    binding_sign_count = (stack > 0).sum(axis=0).astype(np.float32)

    log.info(
        f"binding_convergence_mean_z: peak={binding_convergence_mean_z.max():.4f} "
        f"mean_pos={binding_convergence_mean_z[binding_convergence_mean_z > 0].mean():.4f} "
        f"pct_pos={100*(binding_convergence_mean_z > 0).mean():.1f}%"
    )
    log.info(
        f"binding_sign_count: max={binding_sign_count.max():.0f}/{n_models}  "
        f"pct_all_agree={100*(binding_sign_count == n_models).mean():.1f}%  "
        f"pct_none_positive={100*(binding_sign_count == 0).mean():.1f}%"
    )

    conv_path = out_dir / "binding_convergence_map.dscalar.nii"
    save_cifti_multimap(
        np.stack([binding_convergence_mean_z, binding_sign_count], axis=0),
        ["binding_convergence_mean_z", "binding_sign_count"],
        args.template_cifti, str(conv_path),
    )
    log.info(f"Saved: {conv_path}")

    # ── Top Glasser parcels by convergent binding ────────────────────────────
    bm_axis = get_bm_axis(args.template_cifti)
    parcels = load_glasser_parcels(args.glasser_dlabel, bm_axis)
    rows = []
    for name, indices in parcels.items():
        idx = np.array(indices)
        rows.append({
            "parcel": name,
            "mean_binding_convergence_z": float(binding_convergence_mean_z[idx].mean()),
            "mean_binding_sign_count": float(binding_sign_count[idx].mean()),
            "n_vertices": int(len(idx)),
        })
    report = pd.DataFrame(rows).sort_values("mean_binding_convergence_z", ascending=False).reset_index(drop=True)
    report["rank"] = range(1, len(report) + 1)
    report.to_csv(out_dir / "binding_ranked_parcels.csv", index=False)
    log.info(f"Top-10 Glasser parcels by binding convergence:\n{report.head(10).to_string(index=False)}")

    summary = {
        "models": model_names,
        "n_models": n_models,
        "per_model": {
            m: {
                "peak": float(binding[m].max()),
                "mean": float(binding[m].mean()),
                "pct_positive": float(100 * (binding[m] > 0).mean()),
            } for m in model_names
        },
        "binding_convergence_mean_z_peak": float(binding_convergence_mean_z.max()),
        "binding_convergence_mean_z_pct_positive": float(100 * (binding_convergence_mean_z > 0).mean()),
        "binding_sign_count_pct_all_agree": float(100 * (binding_sign_count == n_models).mean()),
        "binding_sign_count_pct_none_positive": float(100 * (binding_sign_count == 0).mean()),
        "note": ("binding[v] = integration_intact[v] - integration_scrambled[v] (Move-1 "
                 "best-additive partial-r maps, intact vs seed-42 A-V temporal-scramble "
                 "pairing). binding_sign_count is a descriptive per-vertex count of models "
                 "with positive binding, NOT a formal FDR-significance count."),
    }
    json.dump(summary, open(out_dir / "binding_summary.json", "w"), indent=2)
    log.info("Done.")


if __name__ == "__main__":
    main()
