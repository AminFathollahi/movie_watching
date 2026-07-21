"""
subcortical/precompute_neighbors.py
====================================
Pre-compute + cache k-NN neighbor arrays for every subcortical structure.

The subcortical voxel grid is subject-invariant (verified in subcortical_io
self-check), so this runs ONCE — not per subject, unlike the cortical
geodesic cache. All structures get voxel-graph geodesic; cerebellum
additionally gets the purist SUIT surface geodesic (with voxel-graph
fallback if SUITPy/wb_command are unavailable).

Usage
-----
  python subcortical/precompute_neighbors.py \\
      --raw-dir /home/amin/Research/Representation/Movie/data/individual-59k \\
      --subject 132118 --k 100 \\
      --cache-dir /home/amin/Research/Representation/Movie/outputs/subcortical/_neighbor_cache \\
      --workbench /opt/workbench/bin_linux64/wb_command
"""

import argparse
import logging
import sys
import types
from pathlib import Path

import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from subcortical_io import (  # noqa: E402
    SUBCORTICAL_STRUCTURES, STRUCTURE_NEIGHBOR_MODE,
    struct_slices, build_neighbors, get_cerebellum_neighbors,
    extract_subcortical,
)
from preprocess_individual import get_run_path, RUN_IDS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--subject", default="132118",
                   help="Any subject — the voxel grid is identical across subjects.")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--workbench", default="/opt/workbench/bin_linux64/wb_command")
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(get_run_path(Path(args.raw_dir), args.subject, RUN_IDS[0])))
    _, bm_axis = extract_subcortical(img, SUBCORTICAL_STRUCTURES)
    slices = struct_slices(bm_axis)

    for s in SUBCORTICAL_STRUCTURES:
        info = slices[s]
        n_vox = info["stop"] - info["start"]
        log.info(f"{s}: {n_vox} voxels")

        if s.startswith("CEREBELLUM_"):
            hem = "LEFT" if s.endswith("LEFT") else "RIGHT"
            cache_path = cache_dir / f"{s}_neighbors_k{args.k}_geodesic.npy"
            surf_cache_path = cache_dir / f"CEREBELLUM_{hem}" / f"cerebellum_neighbors_k{args.k}_surface_geodesic.npy"
            if cache_path.exists() and surf_cache_path.exists():
                log.info(f"  cached (voxel-graph + surface geodesic) — skipping")
                continue
            neighbors, mode_used = get_cerebellum_neighbors(
                hem, info["world_xyz"], info["voxel_ijk"], args.k, cache_dir, args.workbench)
            log.info(f"  purist mode available: surface_geodesic used for this run={mode_used}")
            # Always also cache the plain voxel-graph geodesic for comparison (§2 policy: both saved).
            build_neighbors(info["voxel_ijk"], info["world_xyz"], args.k, "geodesic", cache_path)
            continue

        mode = STRUCTURE_NEIGHBOR_MODE[s]
        cache_path = cache_dir / f"{s}_neighbors_k{args.k}_{mode}.npy"
        if cache_path.exists():
            log.info(f"  cached — skipping")
            continue
        build_neighbors(info["voxel_ijk"], info["world_xyz"], args.k, mode, cache_path)

    log.info("Neighbor precompute complete.")


if __name__ == "__main__":
    main()
