"""
subcortical/subcortical_noise_ceiling.py
==========================================
Vertex-wise inter-subject noise ceiling for subcortical searchlight RSA.
Mandatory reliability gate (subcortex.txt §4) — subcortical BOLD SNR is
lower than cortex and must be interpreted relative to its own ceiling, not
compared to cortex-grade rho.

Reuses rsa/noise_ceiling.py's GPU/CPU kernels (_compute_nc_gpu/_compute_nc_cpu)
verbatim — they are geometry-agnostic given neighbors + surface_indices +
vertex_to_col, exactly like run_searchlight. The only new logic is looping
over subcortical structures instead of two cortical hemispheres.

Streaming mode only (reads raw CIFTIs directly per subject) — mirrors how
the cortical noise ceiling is typically run on a subject subset, no
preprocessed-CIFTI dependency needed.
"""

import argparse
import gc
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rsa"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsa.shared.rsa_utils import preprocess_fmri  # noqa: E402
from cifti_io import save_cifti_multimap  # noqa: E402
from noise_ceiling import _compute_nc_gpu, _compute_nc_cpu  # noqa: E402
from subcortical_io import (  # noqa: E402
    SUBCORTICAL_STRUCTURES, STRUCTURE_NEIGHBOR_MODE,
    struct_slices, build_neighbors, get_cerebellum_neighbors,
    preprocess_subject_subcortical,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--timing-csv", required=True)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--bin-sec", type=float, required=True)
    p.add_argument("--skip-sec", type=float, default=None)
    p.add_argument("--delay-sec", type=float, default=5.0)
    p.add_argument("--tr", type=float, default=1.0)
    p.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    p.add_argument("--template-cifti", required=True)
    p.add_argument("--neighbor-cache-dir", required=True)
    p.add_argument("--workbench", default="/opt/workbench/bin_linux64/wb_command")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    if args.skip_sec is None:
        args.skip_sec = args.bin_sec
    N = len(args.subjects)
    config = f"k{args.k}_delay{int(args.delay_sec)}s_bin{int(args.bin_sec)}s_skip{int(args.skip_sec)}s_{args.method}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"noise_ceiling_subcortical_{N}subs_{config}.dscalar.nii"
    if out_path.exists():
        log.info(f"Already exists: {out_path.name} — skipping.")
        return

    timing_df = pd.read_csv(args.timing_csv)
    prep_args = types.SimpleNamespace(sg_filter=False, psc=False, gsr=False)

    fmri_all = []
    bm_axis = None
    for sub in args.subjects:
        log.info(f"Loading + binning {sub} ...")
        data, bm_ax, run_trs = preprocess_subject_subcortical(
            sub, Path(args.raw_dir), args.tr, prep_args, SUBCORTICAL_STRUCTURES)
        if bm_axis is None:
            bm_axis = bm_ax
        binned = preprocess_fmri(data, timing_df, run_trs, args.bin_sec, args.tr,
                                  args.delay_sec, skip_sec=args.skip_sec, normalize=True)
        log.info(f"  {sub}: binned shape={binned.shape}")
        fmri_all.append(binned.astype(np.float32))
        del data
        gc.collect()

    n_total = fmri_all[0].shape[1]
    slices = struct_slices(bm_axis)
    neighbor_cache_dir = Path(args.neighbor_cache_dir)

    nc_upper_full = np.zeros(n_total, dtype=np.float32)
    nc_lower_full = np.zeros(n_total, dtype=np.float32)

    for s in SUBCORTICAL_STRUCTURES:
        info = slices[s]
        start, stop = info["start"], info["stop"]
        n_struct = stop - start
        fmri_struct = [f[:, start:stop] for f in fmri_all]

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

        idx = np.arange(n_struct, dtype=np.int32)
        try:
            import torch
            if torch.cuda.is_available():
                nc_up, nc_lo = _compute_nc_gpu(
                    fmri_struct, neighbors, idx, idx,
                    method=args.method, batch_size=args.batch_size, device="cuda")
            else:
                raise ImportError
        except (ImportError, RuntimeError):
            nc_up, nc_lo = _compute_nc_cpu(fmri_struct, neighbors, idx, idx, method=args.method)

        nc_upper_full[start:stop] = nc_up
        nc_lower_full[start:stop] = nc_lo
        log.info(f"  {s}: mode={mode_used}  NC_upper=[{nc_up.min():.3f},{nc_up.max():.3f}] "
                 f"mean={nc_up.mean():.3f}  NC_lower mean={nc_lo.mean():.3f}")
        del fmri_struct, nc_up, nc_lo
        gc.collect()

    save_cifti_multimap(np.stack([nc_lower_full, nc_upper_full], axis=0),
                         ["nc_lower", "nc_upper"], args.template_cifti, str(out_path))
    log.info(f"Saved: {out_path}")
    log.info(f"Done. NC_lower mean={nc_lower_full.mean():.4f}  NC_upper mean={nc_upper_full.mean():.4f}")


if __name__ == "__main__":
    main()
