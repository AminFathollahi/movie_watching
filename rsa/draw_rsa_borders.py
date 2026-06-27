"""
rsa/draw_rsa_borders.py
=======================
Draw surface borders around high-RSA islands on the PE-AV searchlight map.

Threshold: vertices with ρ > (map_mean + N_SD * map_SD), with connected
components smaller than MIN_VERTS vertices removed.  Borders are written
per hemisphere using wb_command -metric-rois-to-border.

Usage
-----
python rsa/draw_rsa_borders.py \
    --rsa-npy  <path>/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy \
    --out-dir  <same dir or other> \
    --n-sd     2 \
    --min-verts 10

Default paths hard-coded to the main PE-AV k100 5s-bin group-average map.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from collections import deque

import nibabel as nib
from nibabel.gifti import GiftiDataArray, GiftiImage
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cifti_io import get_bm_axis, get_cortex_vertex_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── defaults ─────────────────────────────────────────────────────────────────
_BASE = Path("/home/amin/Research/Representation/Movie")
_RSA_NPY = (_BASE / "outputs/rsa/raw/group_average/pe-av-small-16-frame_av"
             "/k100_delay5s_bin5s_skip5s_spearman"
             "/rsa_59k_raw_k100_delay5s_bin5s_skip5s_spearman_searchlight.npy")
_TEMPLATE = (_BASE / "data/preprocessed/average_sub/raw"
             "/group_average_raw_cortex_59k.dtseries.nii")
_L_SURF = (_BASE / "data/HCP_S1200_GroupAvg_v1/GroupAverage_59k"
           "/CohortAvg.L.midthickness_MSMAll.59k_fs_LR.surf.gii")
_R_SURF = (_BASE / "data/HCP_S1200_GroupAvg_v1/GroupAverage_59k"
           "/CohortAvg.R.midthickness_MSMAll.59k_fs_LR.surf.gii")
_WB = Path("/opt/workbench/bin_linux64/wb_command")


# =============================================================================
# Connected components on surface mesh
# =============================================================================

def _build_adjacency(surf_gii: nib.gifti.GiftiImage, n_verts: int) -> list[set]:
    """Return adjacency list built from mesh triangles."""
    adj = [set() for _ in range(n_verts)]
    faces = surf_gii.darrays[1].data  # (n_faces, 3) int32
    for a, b, c in faces:
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    return adj


def _connected_components(mask: np.ndarray, adj: list[set]) -> np.ndarray:
    """Label connected components on surface.  mask: (n_full_verts,) bool.
    Returns labels array same shape; 0 = not in mask."""
    labels = np.zeros(len(mask), dtype=np.int32)
    current_label = 0
    for start in np.where(mask)[0]:
        if labels[start] != 0:
            continue
        current_label += 1
        q = deque([start])
        labels[start] = current_label
        while q:
            v = q.popleft()
            for nb in adj[v]:
                if mask[nb] and labels[nb] == 0:
                    labels[nb] = current_label
                    q.append(nb)
    return labels


def _filter_small_islands(full_mask: np.ndarray, adj: list[set],
                           min_verts: int) -> np.ndarray:
    """Remove connected components with fewer than min_verts vertices."""
    labels = _connected_components(full_mask.astype(bool), adj)
    kept = np.zeros_like(full_mask)
    for lbl in range(1, labels.max() + 1):
        comp = (labels == lbl)
        if comp.sum() >= min_verts:
            kept[comp] = 1.0
    return kept


# =============================================================================
# Border writing (mirrors searchlight.py _write_border_file)
# =============================================================================

def _write_border(full_mask: np.ndarray, surface_path: str,
                  out_path: str, workbench: str, class_name: str = "2sd") -> None:
    tmp = Path(out_path).with_suffix(".tmp.func.gii")
    arr = GiftiDataArray(data=full_mask.astype(np.float32),
                         intent=0, datatype="NIFTI_TYPE_FLOAT32")
    nib.save(GiftiImage(darrays=[arr]), str(tmp))
    try:
        subprocess.run(
            [workbench, "-metric-rois-to-border",
             surface_path, str(tmp), class_name, out_path],
            check=True, capture_output=True, text=True,
        )
        log.info(f"  Saved border: {Path(out_path).name}")
    except subprocess.CalledProcessError as e:
        log.warning(f"wb_command failed: {e.stderr.strip()}")
    finally:
        tmp.unlink(missing_ok=True)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rsa-npy",    default=str(_RSA_NPY))
    p.add_argument("--out-dir",    default=None)
    p.add_argument("--template-cifti", default=str(_TEMPLATE))
    p.add_argument("--left-surface",   default=str(_L_SURF))
    p.add_argument("--right-surface",  default=str(_R_SURF))
    p.add_argument("--workbench",      default=str(_WB))
    p.add_argument("--n-sd",      type=float, default=2.0, dest="n_sd",
                   help="Threshold = mean + n_sd * SD")
    p.add_argument("--min-verts", type=int,   default=10,  dest="min_verts",
                   help="Drop islands smaller than this")
    return p.parse_args()


def main():
    args = parse_args()

    rsa_npy = Path(args.rsa_npy)
    out_dir = Path(args.out_dir) if args.out_dir else rsa_npy.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load RSA map ──────────────────────────────────────────────────────────
    rsa = np.load(str(rsa_npy)).astype(np.float32)
    log.info(f"RSA map: {rsa_npy.name}  shape={rsa.shape}  "
             f"mean={rsa.mean():.4f}  SD={rsa.std():.4f}")

    thresh = float(rsa.mean() + args.n_sd * rsa.std())
    log.info(f"Threshold ({args.n_sd} SD above mean): {thresh:.4f}  "
             f"→ {int((rsa > thresh).sum())} raw vertices ({(rsa>thresh).mean()*100:.1f}%)")

    # ── CIFTI vertex index split ──────────────────────────────────────────────
    bm_axis = get_bm_axis(args.template_cifti)
    left_idx, right_idx = get_cortex_vertex_indices(bm_axis)
    n_left_cifti  = len(left_idx)
    n_right_cifti = len(right_idx)

    rsa_L = rsa[:n_left_cifti]
    rsa_R = rsa[n_left_cifti:]

    # ── Process each hemisphere ───────────────────────────────────────────────
    for tag, rsa_hem, cifti_idx, surf_path in [
        ("lh", rsa_L, left_idx,  args.left_surface),
        ("rh", rsa_R, right_idx, args.right_surface),
    ]:
        n_surf = 59292  # full 59k surface vertices (both hemispheres same)

        # Expand to full surface
        full_mask = np.zeros(n_surf, dtype=np.float32)
        full_mask[cifti_idx] = (rsa_hem > thresh).astype(np.float32)

        log.info(f"  [{tag}] {int(full_mask.sum())} vertices above threshold before filtering")

        # Load surface and filter small islands
        surf_img = nib.load(surf_path)
        adj = _build_adjacency(surf_img, n_surf)
        full_mask = _filter_small_islands(full_mask, adj, args.min_verts)

        log.info(f"  [{tag}] {int(full_mask.sum())} vertices in islands ≥{args.min_verts} verts")

        if full_mask.sum() == 0:
            log.warning(f"  [{tag}] No islands remain after filtering — skipping border.")
            continue

        # Build output stem from input filename
        stem = rsa_npy.stem.replace("_searchlight", "")
        border_path = out_dir / f"{stem}_2sd_{tag}.border"
        _write_border(full_mask, surf_path, str(border_path), args.workbench,
                      class_name=f"2sd_n{args.n_sd}")

    log.info("Done.")


if __name__ == "__main__":
    main()
