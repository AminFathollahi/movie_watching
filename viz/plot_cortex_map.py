"""
viz/plot_cortex_map.py
=======================
Render a grayordinate-space 1-D map (.npy or single-map .dscalar.nii) as a
4-panel cortical surface figure (LH lateral/medial, RH lateral/medial) using
nilearn on the HCP 59k inflated group-average surfaces.

Usage
-----
python viz/plot_cortex_map.py \
    --map-file <path.npy | path.dscalar.nii> \
    --template-cifti <path used to build the map, for BrainModelAxis> \
    --out-png <path.png> \
    --title "PE-AV av RSA (raw, k100, 5s bins)" \
    --cmap viridis --vmin 0 --vmax 0.2 --symmetric-cmap false
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cifti_io import get_bm_axis, get_cortex_vertex_indices
from paths import ROOT as _BASE

_L_INFL = (_BASE / "data/HCP_S1200_GroupAvg_v1/GroupAverage_59k"
           "/CohortAvg.L.inflated_MSMAll.59k_fs_LR.surf.gii")
_R_INFL = (_BASE / "data/HCP_S1200_GroupAvg_v1/GroupAverage_59k"
           "/CohortAvg.R.inflated_MSMAll.59k_fs_LR.surf.gii")
_N_SURF = 59292


def load_map(map_file: str) -> np.ndarray:
    if map_file.endswith(".npy"):
        return np.load(map_file).astype(np.float32)
    img = nib.load(map_file)
    data = img.get_fdata(dtype=np.float32)
    return data[0] if data.ndim == 2 else data


def to_full_surface(grayord_map: np.ndarray, template_cifti: str) -> tuple:
    bm_axis = get_bm_axis(template_cifti)
    left_idx, right_idx = get_cortex_vertex_indices(bm_axis)
    n_left = len(left_idx)

    full_L = np.full(_N_SURF, np.nan, dtype=np.float32)
    full_R = np.full(_N_SURF, np.nan, dtype=np.float32)
    full_L[left_idx] = grayord_map[:n_left]
    full_R[right_idx] = grayord_map[n_left:]
    return full_L, full_R


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map-file", required=True)
    p.add_argument("--template-cifti", required=True)
    p.add_argument("--out-png", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()

    grayord_map = load_map(args.map_file)
    full_L, full_R = to_full_surface(grayord_map, args.template_cifti)

    l_surf = str(_L_INFL)
    r_surf = str(_R_INFL)

    vmax = args.vmax if args.vmax is not None else float(np.nanmax([full_L, full_R]))
    vmin = args.vmin if args.vmin is not None else float(np.nanmin([full_L, full_R]))

    fig, axes = plt.subplots(2, 2, subplot_kw={"projection": "3d"}, figsize=(10, 9))
    panels = [
        (full_L, l_surf, "left", "lateral", axes[0, 0]),
        (full_L, l_surf, "left", "medial", axes[0, 1]),
        (full_R, r_surf, "right", "lateral", axes[1, 0]),
        (full_R, r_surf, "right", "medial", axes[1, 1]),
    ]
    for data, surf, hemi, view, ax in panels:
        plotting.plot_surf_stat_map(
            surf, data, hemi=hemi, view=view, colorbar=False,
            cmap=args.cmap, vmin=vmin, vmax=vmax, threshold=args.threshold,
            bg_on_data=False, axes=ax, figure=fig,
        )
        ax.set_title(f"{hemi.upper()} {view}", fontsize=9)

    fig.suptitle(args.title, fontsize=12)
    sm = plt.cm.ScalarMappable(cmap=args.cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out_png}")


if __name__ == "__main__":
    main()
