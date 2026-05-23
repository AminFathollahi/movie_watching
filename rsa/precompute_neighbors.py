"""
rsa/precompute_neighbors.py
============================
Pre-compute geodesic k-NN neighbour caches for one subject.

Run this BEFORE per-subject searchlight RSA to front-load the expensive
``wb_command -surface-geodesic-distance-all-to-all`` step.  Once the two
``.npy`` files (one per hemisphere) are present in ``cache-dir``, any
searchlight run for the same subject will skip geodesic computation entirely.

The k=150 cache is also usable for any k' ≤ 150: ``run_searchlight.py``
will automatically derive the k'-NN cache from the k=150 cache without
re-running wb_command.

Usage
-----
  python rsa/precompute_neighbors.py \\
      --subject       100610 \\
      --left-surface  /path/to/100610.L.midthickness.surf.gii \\
      --right-surface /path/to/100610.R.midthickness.surf.gii \\
      --workbench     /opt/workbench/bin_linux64/wb_command \\
      --cache-dir     /path/to/outputs/rsa/sg_psc_gsr/_geodesic_cache \\
      --k             150

Called from ``run_analysis.sh neighbors`` mode (via GNU parallel).
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsa.run_searchlight import get_neighbors   # noqa: E402 — after sys.path patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s   %(message)s",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-compute geodesic k-NN caches for one subject.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--subject",       required=True,
                   help="Subject ID (used in cache filename).")
    p.add_argument("--left-surface",  required=True, dest="left_surface",
                   help="Left hemisphere 59k midthickness .surf.gii.")
    p.add_argument("--right-surface", required=True, dest="right_surface",
                   help="Right hemisphere 59k midthickness .surf.gii.")
    p.add_argument("--workbench",     required=True,
                   help="Path to wb_command binary.")
    p.add_argument("--cache-dir",     required=True, dest="cache_dir",
                   help="Directory where .npy neighbour files are stored.")
    p.add_argument("--k",             required=True, type=int,
                   help="Number of nearest neighbours to extract.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    surfaces = [
        ("left",  args.left_surface),
        ("right", args.right_surface),
    ]

    for hem, surf_path in surfaces:
        npy = cache_dir / f"{args.subject}_{hem}_neighbors_k{args.k}.npy"
        if npy.exists():
            log.info(f"Already cached — skipping: {npy.name}")
            continue
        log.info(f"Computing {hem} hemisphere for subject {args.subject} ...")
        get_neighbors(surf_path, args.workbench, args.subject, hem, args.k, cache_dir)

    log.info(f"Neighbor precompute complete: {args.subject}")


if __name__ == "__main__":
    main()
